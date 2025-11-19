# =========================================================================
# I. KURULUM VE KÜTÜPHANELER
# =========================================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
import operator
import time
import math
from stmol import showmol
import py3Dmol
# Optimizasyon için DEAP kütüphanesi
import deap.base as base
import deap.creator as creator
import deap.tools as tools
from deap import algorithms

# Kimya kütüphaneleri
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies as sf
from datasets import load_dataset
# import stmol as showmol # 3D görselleştirme kütüphanesi (varsa)

RDLogger.DisableLog('rdApp.*')

# --- SABİTLER ---
N_BITS = 2048 # Morgan Fingerprint boyutu

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_critic_models():
    """Tüm Eleştirmen (Critic) modellerini yükler."""
    models = {}
    try:
        models['Tg'] = joblib.load('xgb_tg.joblib')
        models['Td'] = joblib.load('xgb_td.joblib')
        models['EPS'] = joblib.load('rf_eps.joblib')
        # DİĞER MODELLERİNİZİ BURAYA EKLEYİN
        models['Tm'] = joblib.load('xgb_tm.joblib')
        models['BandgapBulk'] = joblib.load('xgb_band gap bulk.joblib')
        models['BandgapChain'] = joblib.load('xgb_band gap chain.joblib')
        models['BandgapCrystal'] = joblib.load('xgb_bandgap-crystal.joblib')
        models['GasPerma'] = joblib.load('xgb_gaspermability.joblib')
        models['Refractive'] = joblib.load('rf_refractive_index.joblib')



        return models
    except Exception as e:
        st.error(f"⚠️ Model Yükleme Hatası! Lütfen 'tg_model.joblib', 'td_model.joblib' ve 'eps_model.joblib' dosyalarının mevcut olduğundan emin olun. Hata: {e}")
        return None

# --- YARDIMCI KİMYA FONKSİYONLARI (Değişmedi) ---

def smiles_to_selfies_safe(smiles):
    if not smiles: return None
    clean_smi = smiles.replace('*', '[H]').replace('(*)', '[H]').replace('[*]', '[H]')
    try:
        selfies_string = sf.encoder(clean_smi)
        return selfies_string.replace('[H]', '[*]')
    except:
        return None

def selfies_to_smiles_safe(selfes_string):
    if not selfes_string: return None
    try:
        temp_selfies = selfes_string.replace('[*]', '[H]')
        smiles = sf.decoder(temp_selfies)
        return smiles.replace('[H]', '*')
    except:
        return None

def get_morgan_fp(p_smiles):
    smi_clean = str(p_smiles).replace('*', '[H]').replace('(*)', '[H]').replace('[*]', '[H]')
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, N_BITS)
    return np.array([fp])

def cxSelfies(ind1, ind2):
    # Çaprazlama fonksiyonu (Değişmedi)
    tokens1 = list(sf.split_selfies(ind1[0]))
    tokens2 = list(sf.split_selfies(ind2[0]))
    min_len = min(len(tokens1), len(tokens2))
    if min_len <= 1: return ind1, ind2
    k = random.randint(1, min_len - 1)
    ind1[0] = "".join(tokens1[:k] + tokens2[k:])
    ind2[0] = "".join(tokens2[:k] + tokens1[k:])
    return ind1, ind2

def mutSelfies(individual):
    # Mutasyon fonksiyonu (Değişmedi)
    tokens = list(sf.split_selfies(individual[0]))
    if not tokens: return individual,
    if random.random() < 0.6 and len(tokens) > 1:
        idx = random.randint(0, len(tokens) - 1)
        del tokens[idx]
    if random.random() < 0.4:
        idx = random.randint(0, len(tokens))
        new_token = random.choice(['[C]', '[N]', '[O]', '[F]', '[Cl]', '[S]', '[*]'])
        tokens.insert(idx, new_token)
    individual[0] = "".join(tokens)
    return individual,

# =========================================================================
# II. DİNAMİK DEĞERLENDİRME ÇEKİRDEĞİ (DYNAMIC EVALUATE)
# =========================================================================

def evaluate_individual_single_obj(individual, models, targets, active_props):
    """
    Seçilen hedeflere (active_props) olan toplam mesafeye (hata) göre değerlendirir.
    """
    s_selfies = individual[0]
    s_smiles = selfies_to_smiles_safe(s_selfies)
    
    if s_smiles is None:
        return (1000.0,) 

    fp = get_morgan_fp(s_smiles)
    if fp is None:
        return (1000.0,)

    # 1. Tahminleri Al
    preds = {}
    for prop in active_props:
        if prop in models:
             preds[prop] = models[prop].predict(fp)[0]
    
    # 2. Toplam Hatayı Hesapla
    total_error = 0.0
    
    if not active_props:
        # Hiçbir hedef seçilmezse ceza
        return (1000.0,) 

    for prop in active_props:
        # Hata = |Tahmin - Hedef|
        if prop in preds:
            error = abs(preds[prop] - targets[prop])
            total_error += error
    
    # Seçilen hiçbir özellik hesaplanamazsa büyük ceza
    if total_error == 0.0 and len(active_props) > 0:
         return (1000.0,) 
         
    return (total_error,)

# =========================================================================
# III. ANA GENETİK ALGORİTMA AKIŞI
# =========================================================================

# DEAP Yapısını Tanımlama (Minimizasyon için)
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimizasyon için
    creator.create("Individual", list, fitness=creator.FitnessMin)


def run_single_objective_flow(models, generations, targets, active_props, initial_pop):
    
    # 1. Toolbox'ı Tanımlama
    toolbox = base.Toolbox()
    toolbox.register("attr_selfies", random.choice, initial_pop)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_selfies, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operatörleri Kaydetme
    # Evaluate fonksiyonuna hedefleri ve aktif özellikleri geçir
    toolbox.register("evaluate", evaluate_individual_single_obj, models=models, targets=targets, active_props=active_props)
    toolbox.register("mate", cxSelfies)
    toolbox.register("mutate", mutSelfies)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # 2. GA Parametreleri
    pop_size = 100
    cxpb = 0.7 
    mutpb = 0.3 
    pop = toolbox.population(n=pop_size)
    best_history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Başlangıç popülasyonunu değerlendir
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # 3. Akış
    for gen in range(generations):
        offspring = toolbox.select(pop, pop_size)
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(ind1, ind2)
            if random.random() < mutpb:
                toolbox.mutate(ind1)
            del ind1.fitness.values, ind2.fitness.values

        # Yeni bireyleri değerlendir
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
            
        pop = offspring
        
        # En iyi bireyi bul ve tarihçeye kaydet
        best_of_gen = tools.selBest(pop, 1)[0]
        best_history.append(best_of_gen.fitness.values[0])
        
        # Streamlit durumunu güncelle
        progress_bar.progress((gen + 1) / generations)
        status_text.text(f"Nesil {gen+1}/{generations}: En İyi Toplam Hata: {best_of_gen.fitness.values[0]:.2f}")
        time.sleep(0.02)
        
    # Sonuç
    best_ind = tools.selBest(pop, 1)[0]
    best_smiles = selfies_to_smiles_safe(best_ind[0])
    
    if best_smiles:
        fp = get_morgan_fp(best_smiles)
        preds = {prop: models[prop].predict(fp)[0] for prop in models.keys()}
        return {'smiles': best_smiles, 'preds': preds, 'total_error': best_ind.fitness.values[0]}, best_history
    else:
        return None, best_history
def make_3d_view(smiles):
    try:
        clean_smi = str(smiles).replace('*', '[H]')
        mol = Chem.MolFromSmiles(clean_smi)
        mol = Chem.AddHs(mol) # 3D için hidrojenler şart
        AllChem.EmbedMolecule(mol) # 3D koordinatları hesapla
        AllChem.MMFFOptimizeMolecule(mol) # Yapıyı enerji açısından düzelt
        
        mblock = Chem.MolToMolBlock(mol)
        
        view = py3Dmol.view(width=400, height=400)
        view.addModel(mblock, 'mol')
        # Görünüm stili: Stick (Çubuk) ve renkler elemente göre
        view.setStyle({'stick':{'colorscheme':'Jmol'}})
        view.zoomTo()
        view.spin(True) # Otomatik dönsün mü? Evet, havalı durur.
        return view
    except:
        return None
# =========================================================================
# IV. STREAMLIT ANA KISIM
# =========================================================================

st.title("🎯 PolimerX - Dinamik Optimizasyon")

models = load_critic_models()
ALL_PROPS = list(models.keys()) # Yüklenen modellerin anahtarları: ['Tg', 'Td', 'EPS']

if models:
    st.sidebar.header("⚙️ Hedef Seçimi")
    
    # 1. Optimizasyona Dahil Edilecek Özelliklerin Seçimi
    active_props = []
    
    st.sidebar.markdown("### Dahil Edilecek Özellikler")
    # Her özellik için onay kutusu oluştur
    if st.sidebar.checkbox("Tg (Camsı Geçiş Sıcaklığı)", value=True):
        active_props.append('Tg')
    if st.sidebar.checkbox("Td (Bozunma Sıcaklığı)"):
        active_props.append('Td')
    if st.sidebar.checkbox("EPS (Dielektrik Sabiti)"):
        active_props.append('EPS')
    if st.sidebar.checkbox("Tm (Erime Sıcaklığı)"):
        active_props.append('Tm')
    if st.sidebar.checkbox("Bandgap Bulk (Elektriksel Band Aralığı - Bulk)"):
        active_props.append('BandgapBulk')
    if st.sidebar.checkbox("Bandgap Chain (Elektriksel Band Aralığı - Zincir)"):
        active_props.append('BandgapChain')
    if st.sidebar.checkbox("Bandgap Crystal (Elektriksel Band Aralığı - Kristal)"):
        active_props.append('BandgapCrystal')
    if st.sidebar.checkbox("Gas Permeability (Gaz Geçirgenliği)"):
        active_props.append('GasPerma')
    if st.sidebar.checkbox("Refractive Index (Kırılma İndeksi)"):
        active_props.append('Refractive')
    
     # En az bir hedef seçilmemişse uyarı ver
        
    if not active_props:
        st.sidebar.warning("Lütfen optimize edilecek en az bir hedef seçin.")
        st.stop()

    # 2. Hedef Değerler (Sadece seçilenler için giriş alanı göster)
    st.sidebar.markdown("### Hedef Değerler")
    targets = {}
    
    # Tg Hedefi
    if 'Tg' in active_props:
        targets['Tg'] = st.sidebar.number_input("Hedef Tg (°C):", value=200.0, step=5.0)
    else:
        targets['Tg'] = 0.0 # Kullanılmayacağı için sıfır tutabiliriz

    # Td Hedefi
    if 'Td' in active_props:
        targets['Td'] = st.sidebar.number_input("Hedef Td (°C):", value=350.0, step=5.0)
    else:
        targets['Td'] = 0.0

    # EPS Hedefi
    if 'EPS' in active_props:
        targets['EPS'] = st.sidebar.number_input("Hedef EPS:", value=2.5, step=0.1)
    else:
        targets['EPS'] = 0.0
    
    if 'Tm' in active_props:
        targets['Tm'] = st.sidebar.number_input("Hedef Tm:", value=2.5, step=0.1)
    else:
        targets['Tm'] = 0.0   

    if 'BandgapBulk' in active_props:
        targets['BandgapBulk'] = st.sidebar.number_input("Hedef BandgapBulk:", value=2.5, step=0.1)
    else:
        targets['BandgapBulk'] = 0.0  

    if 'BandgapChain' in active_props:
        targets['BandgapChain'] = st.sidebar.number_input("Hedef BandgapChain:", value=2.5, step=0.1)
    else:
        targets['BandgapChain'] = 0.0  
    
    if 'BandgapCrystal' in active_props:
        targets['BandgapCrystal'] = st.sidebar.number_input("Hedef BandgapCrystal:", value=2.5, step=0.1)
    else:
        targets['BandgapCrystal'] = 0.0  
    
    if 'GasPerma' in active_props:
        targets['GasPerma'] = st.sidebar.number_input("Hedef GasPerma:", value=2.5, step=0.1)
    else:
        targets['GasPerma'] = 0.0 

    if 'Refractive' in active_props:
        targets['Refractive'] = st.sidebar.number_input("Hedef Refractive:", value=2.5, step=0.1)
    else:
        targets['Refractive'] = 0.0 
    # Diğer hedefleriniz için buraya kod ekleyin (Örn: Tm, Bandgap)

    # 3. GA Parametreleri
    generations = st.sidebar.slider("Evrim Nesli Sayısı", 10, 100, 30)

    # Başlangıç popülasyonu (Gerçek verinizi buraya koyun)
    repo_id = "OsBaran/Polimer-Ozellik-Tahmini"
    tg_data = load_dataset(repo_id,split="Tg")
    df = tg_data.to_pandas()
            # Sütun adının 'p_smiles' veya 'smiles' olduğundan emin ol
    col_name = 'p_smiles' if 'p_smiles' in df.columns else 'smiles'
    initial_pop = df[col_name].tolist()
    # initial_pop_simulated = ["C[N+](C)(C)CC(=O)[O-]", "CCCCCC(=O)NCCCCCCNCC(=O)O", "c1ccc(C)c(C)c1", "CC(=O)Oc1ccccc1C(=O)O"] * 50
    initial_selfies = [smiles_to_selfies_safe(s) for s in initial_pop if smiles_to_selfies_safe(s)]

    if st.sidebar.button("🚀 Hedefi Ara", type="primary"):
        
        if not initial_selfies:
            st.error("Başlangıç popülasyonu boş veya geçersiz.")
            st.stop()
            
        with st.spinner(f'Genetik Algoritma Çalışıyor... Hedefler: {", ".join(active_props)}'):
            best_poly_data, history = run_single_objective_flow(models, generations, targets, active_props, initial_selfies)

        if best_poly_data:
            preds = best_poly_data['preds']
            
            st.success(f"✅ Optimizasyon Tamamlandı! En Yakın Polimer Bulundu.")
            
            # 1. Metrikleri Ferah Gösterme (Grid Layout)
            st.markdown("### 🔬 Tahmin Sonuçları")
            
            # Toplam Hata Metriği (En Üste, Büyük)
            st.metric(
                "🎯 Toplam Sapma (Minimize)", 
                f"{best_poly_data['total_error']:.2f}", 
                help=f"Seçilen özelliklere ({', '.join(active_props)}) olan toplam mutlak mesafe."
            )
            
            st.divider()
            
            # Diğer Özellikleri 3'erli Satırlar Halinde Göster
            cols_per_row = 3 
            current_col_idx = 0
            row_cols = st.columns(cols_per_row) # İlk satırı oluştur
            
            for prop in ALL_PROPS:
                # Sütun dolduysa yeni satıra geç
                if current_col_idx >= cols_per_row:
                    current_col_idx = 0
                    row_cols = st.columns(cols_per_row)
                
                col = row_cols[current_col_idx]
                
                is_active = prop in active_props
                
                # Birim belirleme (Sıcaklık için °C, diğerleri birimsiz)
                unit = " °C" if prop in ['Tg', 'Td', 'Tm'] else ""
                
                # Hedef Metni
                target_val = targets.get(prop, 0.0)
                delta_text = f"Hedef: {target_val}{unit}" if is_active else "Optimizasyon Dışı"
                
                # Renklendirme (Aktifse yeşilimsi, değilse gri)
                val_str = f"{preds[prop]:.2f}{unit}"
                
                # Metriği yazdır
                col.metric(
                    label=f"{prop}",
                    value=val_str,
                    delta=delta_text,
                    delta_color="off"  # Kırmızı/Yeşil okları kapat, sadece metni göster
                )
                
                current_col_idx += 1

            st.divider()

            # 2. Evrim Grafiği ve Yapı (Yan Yana)
            col_graph, col_struct = st.columns([1, 1])
            
            with col_graph:
                st.markdown("### 📈 Hata Evrimi")
                df_history = pd.DataFrame({"Toplam Hata": history})
                st.line_chart(df_history)
                
            with col_struct:
                st.markdown("### 🧬 Yeni Polimer Yapısı")
                st.text_area("p-SMILES Kodu:", best_poly_data['smiles'], height=100)

            # 3. 3D Görüntüleme (Tam Genişlik)
            st.markdown("### 🧊 3D Molekül Yapısı")
            view = make_3d_view(best_poly_data["smiles"])
            if view:
                # Mobilde taşmasın diye responsive genişlik
                showmol(view, height=400, width=800) 
            else:
                st.warning("3D model oluşturulamadı.")
        
        else:
            st.error("Belirlenen hedeflere yakın geçerli bir polimer bulunamadı. Lütfen nesil sayısını veya hedef değerleri değiştirin.")