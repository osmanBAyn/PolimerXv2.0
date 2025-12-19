# =========================================================================
# I. KURULUM VE KÜTÜPHANELER
# =========================================================================
import google.generativeai as genai
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
import pubchempy as pcp
# Optimizasyon için DEAP kütüphanesi
import deap.base as base
import deap.creator as creator
import deap.tools as tools
from deap import algorithms
import lightgbm as lgbm
# Kimya kütüphaneleri
from rdkit import Chem
import torch
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies as sf
from datasets import load_dataset
import rdkit.Chem.rdChemReactions as rdChemReactions
# import stmol as showmol # 3D görselleştirme kütüphanesi (varsa)
import sys
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Draw
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
# --- YEREL RETROSENTEZ MODELİ ENTEGRASYONU ---

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- MODELİ ÖNBELLEĞE AL (Sadece 1 kere yüklenir) ---
@st.cache_resource
def load_my_trained_model():
# Buraya az önce oluşturduğun Hugging Face modelinin adını yaz
    model_path = "OsBaran/POLISEN_T5_Model"
    try:
        print("--- [LOG] Model yükleme fonksiyonu başladı...", flush=True) # EKLE

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("--- [LOG] Tokenizer yüklendi. Model yükleniyor...", flush=True) # EKLE

        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()  # Modeli değerlendirme moduna al
        print("--- [LOG] Model Quantization işlemi yapılıyor...", flush=True) # EKLE

    # Modeli sıkıştır (Quantization)
    # Bu işlem CPU'da çalışmayı İNANILMAZ hızlandırır
        model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        print("--- [LOG] Model başarıyla hafızaya alındı! Hazır.", flush=True) # EKLE

        return tokenizer, model
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None, None

def predict_monomers_local(polymer_smiles):
    """
    Önce eğitilmiş T5 modelini kullanır. 
    Eğer sonuç başarısızsa kural tabanlı motoru devreye sokar.
    """
    # 1. MODEL TAHMİNİ
    print(f"--- [LOG] {time.strftime('%X')} - Girdi (Input) hazırlanıyor...", flush=True)

    tokenizer, model = load_my_trained_model()
    ai_prediction = ""
    
    if model:
        try:
            # Eğitimde kullandığımız "retrosynthesis: " ön ekini unutmuyoruz!
            input_text = "retrosynthesis: " + polymer_smiles
            inputs = tokenizer(input_text, return_tensors="pt")
            print(f"--- [LOG] {time.strftime('%X')} - MODEL GENERATE BAŞLIYOR (Burada bekleyebilir)...", flush=True)
            start_time = time.time()

            # Tahmin üret
            outputs = model.generate(
                inputs["input_ids"], 
                max_length=64, 
                num_beams=1,
                do_sample=False,# En iyi 5 yolu ara
                early_stopping=True,
                num_return_sequences=1
            )
            ai_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            end_time = time.time()
            print(f"--- [LOG] {time.strftime('%X')} - MODEL GENERATE BİTTİ! Geçen süre: {round(end_time - start_time, 2)} saniye.", flush=True)
            
            # 3. Decode İşlemi
            print(f"--- [LOG] {time.strftime('%X')} - Çıktı okunuyor (Decode)...", flush=True)
            # result = tokenizer.decode(...) kodun burada
            
            print("--- [LOG] İşlem başarıyla tamamlandı.", flush=True)
            st.success("Bitti!")
        except:
            ai_prediction = ""

    # 2. SONUÇ KONTROLÜ VE HİBRİT KARAR
    # Model mantıklı bir şey (örneğin nokta ile ayrılmış iki parça) döndürdü mü?
    if ai_prediction and " . " in ai_prediction:
        return f"{ai_prediction} (Yerel AI Modeli)"
    
    # Model başarısızsa veya emin değilse KURAL MOTORUNU çağır
    else:
        rules = decompose_polymer(polymer_smiles) # Mevcut fonksiyonun
        if rules:
            monomers = rules[0]['monomers']
            return f"{' . '.join(monomers)} (Kural Tabanlı - Yedek)"
        else:
            # Model bir şey buldu ama nokta yoksa yine de gösterelim (belki tek monomerdir)
            if ai_prediction:
                return f"{ai_prediction} (AI Modeli - Tek Parça)"
            return "Ayrıştırılamadı"
# --- YAYGIN ÇÖZÜCÜLER REFERANS LİSTESİ ---
COMMON_SOLVENTS = {
    "n-Heksan (Apolar)": 7.3,
    "Dietil Eter": 7.4,
    "Toluen (Aromatik)": 8.9,
    "Etil Asetat": 9.1,
    "Kloroform": 9.3,
    "Aseton (Polar Aprotik)": 9.9,
    "Diklorometan (DCM)": 9.7,
    "THF (Tetrahidrofuran)": 9.1,
    "Etanol (Alkol)": 12.7,
    "Metanol": 14.5,
    "Su (Çok Polar)": 23.4
}
def get_soluble_solvents(pred_val):
    """Tahmin edilen Hildebrand değerine göre uygun çözücüleri bulur."""
    soluble_list = []
    swelling_list = [] # Kısmi çözünme / Şişme
    
    for solvent, s_val in COMMON_SOLVENTS.items():
        diff = abs(pred_val - s_val)
        
        if diff <= 1.8: # İyi çözücü
            soluble_list.append(solvent)
        elif diff <= 2.5: # Sınırda (Isıtarak çözünebilir veya şişer)
            swelling_list.append(solvent)
            
    return soluble_list, swelling_list
def draw_2d_molecule(smiles):
    """SMILES kodundan yüksek kaliteli 2D resim oluşturur."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Görüntü kalitesini artır
            dopts = Draw.MolDrawOptions()
            dopts.addAtomIndices = False
            dopts.bondLineWidth = 2
            return Draw.MolToImage(mol, size=(500, 400), options=dopts)
    except:
        return None
def inject_custom_css():
    st.markdown("""
    <style>
        /* Ana Başlık Stili */
        .main-title {
            font-size: 3rem;
            color: #4A90E2;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
        }
        /* Alt Başlık */
        .sub-title {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        /* Kart Tasarımı (Sonuçlar için) */
        .metric-card {
            background-color: #f9f9f9;
            border-left: 5px solid #4A90E2;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
        /* Dark Mode Uyumu için Kart Rengi */
        @media (prefers-color-scheme: dark) {
            .metric-card {
                background-color: #262730;
                border-left: 5px solid #4A90E2;
                color: white;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# Uygulamanın en başında çağırın:
inject_custom_css()
# --- SABİTLER ---
N_BITS = 2048 # Morgan Fingerprint boyutu
@st.cache_data
def get_initial_population():
    """Verisetini sadece bir kez indirir ve önbelleğe alır."""
    repo_id = "OsBaran/Polimer-Ozellik-Tahmini"
    tg_data = load_dataset(repo_id, split="Tg")
    df = tg_data.to_pandas()
    col_name = 'p_smiles' if 'p_smiles' in df.columns else 'smiles'
    # Sadece geçerli SELFIES'leri filtrele ve listeye çevir
    raw_smiles = df[col_name].tolist()
    valid_selfies = []
    for s in raw_smiles:
        sf_str = smiles_to_selfies_safe(s)
        if sf_str:
            valid_selfies.append(sf_str)
    return valid_selfies, raw_smiles # İkisini de döndür
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
        models['GasPerma'] = joblib.load('lgbm_gas_pipeline.joblib')
        models['Refractive'] = joblib.load('rf_refractive_index.joblib')

        models['LOI'] = joblib.load('xgb_loi.joblib')               # Yanıcılık
        models['Solubility'] = joblib.load('xgb_solubility.joblib') # Çözünürlük
        models['ThermalCond'] = joblib.load('xgb_thermal_cond.joblib') # Isıl İletkenlik
        models['CTE'] = joblib.load('xgb_cte.joblib')

        return models
    except Exception as e:
        st.error(f"⚠️ Model Yükleme Hatası! Lütfen 'tg_model.joblib', 'td_model.joblib' ve 'eps_model.joblib' dosyalarının mevcut olduğundan emin olun. Hata: {e}")
        return None

def run_ga_silent(models, generations, targets, active_props, initial_pop, ranges_dict):
    """
    GA'yı grafik çizmeden (sessizce) çalıştırır. Çoklu testler için optimize edilmiştir.
    """
    # DEAP Kurulumu (Mevcut kodunuzdakiyle aynı)
    toolbox = base.Toolbox()
    toolbox.register("attr_selfies", random.choice, initial_pop)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_selfies, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual_optimized, models=models, targets=targets, active_props=active_props, ranges=ranges_dict)
    toolbox.register("mate", cxSelfies)
    toolbox.register("select", tools.selTournament, tournsize=7) # Turnuva boyutu 7 (Önerilen)

    pop_size = 100
    pop = toolbox.population(n=pop_size)
    
    # Sadece en iyilerin geçmişini tutacağız
    best_fitness_history = []
    
    # Parametreler (Optimize ettiğimiz değerler)
    cxpb, mutpb, extendpb, newpb, chempb = 0.8, 0.05, 0.05, 0.01, 0.05

    # --- HIZLI DÖNGÜ ---
    for gen in range(generations):
        # Seçilim & Klonlama
        offspring = toolbox.select(pop, pop_size)
        offspring = list(map(toolbox.clone, offspring))

        # Çaprazlama
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        # Mutasyon
        for i in range(len(offspring)):
            if not offspring[i].fitness.valid: pass
            offspring[i] = generate_offspring(offspring[i], initial_pop, mutpb=mutpb, extendpb=extendpb, newpb=newpb, chempb=chempb)
            del offspring[i].fitness.values

        # Değerlendirme
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = offspring
        
        # En iyiyi kaydet
        fits = [ind.fitness.values[0] for ind in pop]
        best_fitness_history.append(min(fits))

    return best_fitness_history
# --- YARDIMCI KİMYA FONKSİYONLARI (Değişmedi) --
def run_mass_random_test(models, generations, initial_pop, ranges_dict, num_trials=100):
    """
    Rastgele hedeflerle 100 kez stres testi yapar.
    """
    results = []
    all_props_list = list(ranges_dict.keys())
    
    # İlerleme Göstergeleri
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_trials):
        # 1. RASTGELE SENARYO OLUŞTURMA
        # Kaç özellik optimize edilecek? (2 ile 5 arası rastgele)
        n_active = random.randint(2, 5)
        # Hangi özellikler olacak?
        active_props = random.sample(all_props_list, n_active)
        
        # Hedefleri belirle (Rastgele)
        current_targets = {}
        target_descriptions = []
        for prop in active_props:
            r = ranges_dict[prop]
            # Min ve Max arasında rastgele bir değer seç
            val = random.uniform(r['min'], r['max'])
            
            # Bazı değerleri tam sayıya yuvarla (Sıcaklıklar gibi)
            if r.get('is_int', False) or prop in ['Tg', 'Td', 'Tm', 'LOI']:
                val = round(val, 0)
            else:
                val = round(val, 2)
                
            current_targets[prop] = val
            target_descriptions.append(f"{prop}={val}")

        # 2. GA'YI ÇALIŞTIR (Sessiz Modda)
        # run_ga_silent fonksiyonunu önceki adımdan aldığınızı varsayıyorum
        history = run_ga_silent(models, generations, current_targets, active_props, initial_pop, ranges_dict)
        
        # 3. SONUCU KAYDET
        final_score = history[-1] # En son jenerasyonun en iyi skoru
        
        results.append({
            "Deneme No": i + 1,
            "Hedef Sayısı": n_active,
            "Hedefler": ", ".join(target_descriptions),
            "Final Hata Skoru": final_score
        })
        
        # İlerlemeyi Güncelle
        progress_bar.progress((i + 1) / num_trials)
        status_text.text(f"Test {i+1}/{num_trials} | Son Hata: {final_score:.4f} | Hedefler: {', '.join(target_descriptions)[:50]}...")
    
    status_text.success(f"{num_trials} Farklı Senaryo Testi Tamamlandı!")
    return pd.DataFrame(results)
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
# --- MEVCUT IMPORTLARIN ALTINA EKLE ---
from rdkit.Chem import Descriptors

# --- YENİ ÖZELLİK FONKSİYONU ---
def get_gas_features_combined(smiles):
    """
    Gaz geçirgenliği LGBM modeli için hem Morgan FP hem de
    Fiziksel Deskriptörleri birleştirir.
    """
    try:
        mol = Chem.MolFromSmiles(smiles.replace('*', '[H]'))
        if mol is None: return None
        
        # 1. Morgan Fingerprint (2048 bit)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
        
        # 2. Fiziksel Deskriptörler
        desc = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HallKierAlpha(mol)
        ])
        
        # İkisini birleştir
        return np.concatenate((fp, desc)).reshape(1, -1)
    except:
        return None
def cxSelfies(ind1, ind2):
    t1 = list(sf.split_selfies(ind1[0]))
    t2 = list(sf.split_selfies(ind2[0]))
    min_len = min(len(t1), len(t2))
    if min_len < 2: return ind1, ind2

    # Segmentleri belirle
    split1 = random.randint(1, min_len-1)
    split2 = random.randint(1, min_len-1)
    
    new1 = t1[:split1] + t2[split2:]
    new2 = t2[:split2] + t1[split1:]

    # Valid bireyleri seç
    new1_str = "".join(new1)
    new2_str = "".join(new2)
    
    if is_valid_polymer(new1_str):
        ind1[0] = new1_str
    if is_valid_polymer(new2_str):
        ind2[0] = new2_str
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
# II. DİNAMİK DEĞERLENDİRME ÇEKİRDEĞİ kısmına ekleyin

# Global önbellek sözlüğü (Uygulama yeniden başlayana kadar tutulur)
# Key: SELFIES string, Value: (Fitness Score,)
FITNESS_CACHE = {}

def evaluate_individual_optimized(individual, models, targets, active_props, ranges):
    s_selfies = individual[0]
    
    # Önbellek Kontrolü
    if s_selfies in FITNESS_CACHE:
        return FITNESS_CACHE[s_selfies]

    s_smiles = selfies_to_smiles_safe(s_selfies)
    if s_smiles is None: return (1000.0,)

    # Standart Fingerprint (Diğer modeller için)
    fp = get_morgan_fp(s_smiles)
    
    # Gas Model Özellikleri (Sadece GasPerma aktifse hesapla)
    gas_features = None
    if 'GasPerma' in active_props:
        gas_features = get_gas_features_combined(s_smiles)

    if fp is None: return (1000.0,)

    preds = {}
    
    # --- TAHMİN DÖNGÜSÜ ---
    for prop in active_props:
        if prop in models:
            # ÖZEL DURUM: GasPerma modeli için özel özellikleri kullan
            if prop == 'GasPerma':
                if gas_features is not None:
                    # Model log10 tahmini yapıyor, bunu gerçek değere çeviriyoruz (10^x)
                    log_pred = models[prop].predict(gas_features)[0]
                    preds[prop] = 10 ** log_pred 
                else:
                    preds[prop] = 0.0 # Hata durumunda
            
            # DİĞERLERİ: Standart Fingerprint kullanır
            else:
                preds[prop] = models[prop].predict(fp)[0]
    
    # --- HATA HESAPLAMA ---
    total_error = 0.0
    if not active_props: return (1000.0,)

    for prop in active_props:
        if prop in preds:
            norm_error = abs(preds[prop] - targets[prop]) / (ranges[prop]['max'] - ranges[prop]['min'])
            total_error += np.exp(norm_error * 10) - 1
    
    # ... (Geri kalan SA Score ve return kısmı aynı kalacak) ...
    sa_score = get_sa_score_local(s_smiles)
    total_error += sa_score * 2.0
    
    result = (total_error,)
    FITNESS_CACHE[s_selfies] = result
    return result
def run_random_benchmark(models, targets, active_props, initial_pop, ranges_dict, total_budget, batch_size=100):
    """
    GA ile adil kıyaslama için Rastgele Arama (Random Search) yapar.
    total_budget: Toplam değerlendirme sayısı (GA'daki pop_size * generations)
    batch_size: Grafik çizimi için her kaç adımda bir kayıt alınacağı (GA'daki pop_size kadar olmalı)
    """
    history_random = []
    best_so_far = float('inf')
    
    # İlerleme çubuğu (kullanıcı beklerken sıkılmasın)
    progress_text = st.empty()
    bar = st.progress(0)
    
    for i in range(0, total_budget, batch_size):
        # Batch (Grup) halindeki rastgele bireyler
        # initial_pop listesinden rastgele seç
        candidates = random.sample(initial_pop, batch_size) 
        
        scores = []
        for ind_selfies in candidates:
            # Mevcut evaluate fonksiyonunu kullanıyoruz (Adil olması için)
            # individual formatı liste olduğu için [ind_selfies] şeklinde veriyoruz
            fit = evaluate_individual_optimized([ind_selfies], models, targets, active_props, ranges_dict)
            
            # Ceza puanı alanları (1000) filtreleyebiliriz veya olduğu gibi alabiliriz
            # Random search genelde çok hata yapar, olduğu gibi alalım.
            scores.append(fit[0])
        
        # Bu batch'teki en iyiyi bul
        current_batch_best = min(scores)
        
        # Genel en iyiyi güncelle
        if current_batch_best < best_so_far:
            best_so_far = current_batch_best
            
        history_random.append(best_so_far)
        
        # İlerlemeyi güncelle
        progress = (i + batch_size) / total_budget
        if progress > 1.0: progress = 1.0
        bar.progress(progress)
        progress_text.text(f"Rastgele Arama: {i}/{total_budget} tamamlandı. En iyi skor: {best_so_far:.4f}")
        
    bar.empty()
    progress_text.empty()
    return history_random

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

            norm_error = abs(preds[prop] - targets[prop]) / (ranges[prop]['max'] - ranges[prop]['min'])

            total_error += np.exp(norm_error * 10) - 1  # Küçük farklar neredeyse lineer, büyük farklar çok ağır

   

    # Seçilen hiçbir özellik hesaplanamazsa büyük ceza

    if total_error == 0.0 and len(active_props) > 0:

         return (1000.0,)
    
    total_error += get_sa_score_local(s_smiles) / 10.0 # SA Score ekle
         

    return (total_error,)

# =========================================================================
# III. ANA GENETİK ALGORİTMA AKIŞI
# =========================================================================

# DEAP Yapısını Tanımlama (Minimizasyon için)
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimizasyon için
    creator.create("Individual", list, fitness=creator.FitnessMin)

# =========================
# 1. Sentezlenebilirlik Kontrolü
# =========================
def is_valid_polymer(selfies_str):
    """
    Hem kimyasal geçerliliği hem de polimer olma şartını (bağlantı noktaları) kontrol eder.
    """
    # 1. SELFIES -> SMILES dönüşümü
    smiles = selfies_to_smiles_safe(selfies_str)
    if smiles is None: 
        return False

    # ==========================================================
    # KONTROL 1: Bağlantı Noktası (Star Atom) Kontrolü
    # ==========================================================
    # Bir polimerin tekrar eden birim (monomer) olması için 
    # en az 2 ucunun açık olması gerekir (* işareti).
    # Lineer polimerler için genellikle tam 2 adet istenir.
    # Ağ yapılı (cross-linked) polimerler için >2 olabilir.
    
    star_count = smiles.count('*')
    if star_count < 2:
        return False  # Zincir kopmuş, bu artık bir polimer değil.

    # ==========================================================
    # KONTROL 2: Çok Küçük Moleküllerin Engellenmesi
    # ==========================================================
    # GA bazen "*C*" gibi çok anlamsız küçük şeyler üretebilir.
    # Yıldızlar hariç atom sayısına bakabiliriz.
    
    clean_smi = smiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(clean_smi)
    
    if mol is None:
        return False # Kimyasal olarak bozuk
        
    # Yıldızlar (Hidrojen oldu) hariç ağır atom sayısı (C, O, N vs.) en az 4 olsun
    if mol.GetNumHeavyAtoms() < 4:
        return False

    return True


MUTATION_TOKENS = ['[C]', '[N]', '[O]', '[F]', '[Cl]', '[S]', '[*]', 'c', 'n', 'o']

# =========================
# 2. Mutasyon (küçük token değişiklikleri)
# =========================
def mutSelfies(individual, max_attempts=5):
    tokens = list(sf.split_selfies(individual[0]))
    if not tokens: 
        return individual

    for _ in range(max_attempts):
        temp_tokens = tokens.copy()
        # Token silme
        if random.random() < 0.3 and len(temp_tokens) > 1:
            idx = random.randint(0, len(temp_tokens) - 1)
            del temp_tokens[idx]
        # Token ekleme
        if random.random() < 0.3:
            idx = random.randint(0, len(temp_tokens))
            new_token = random.choice(MUTATION_TOKENS)
            temp_tokens.insert(idx, new_token)
        # Token değiştirme
        if random.random() < 0.3:
            idx = random.randint(0, len(temp_tokens) - 1)
            temp_tokens[idx] = random.choice(MUTATION_TOKENS)
        
        candidate = "".join(temp_tokens)
        if is_valid_polymer(candidate):
            individual[0] = candidate
            return individual
    
    # Max deneme sonrası geçerli değilse rastgele valid birey ata
    individual[0] = random.choice(initial_selfies)
    return individual


# =========================
# 3. Zincir Uzatma
# =========================
def extendPolymer(individual, max_add=3):
    tokens = list(sf.split_selfies(individual[0]))
    for _ in range(random.randint(1, max_add)):
        tokens.append(random.choice(['[C]', '[N]', '[O]', '[F]', '[Cl]', '[S]']))
    candidate = "".join(tokens)
    return candidate if is_valid_polymer(candidate) else individual[0]


# =========================
# 4. Reaction tabanlı mutasyon 
# =========================
import rdkit.Chem.rdChemReactions as rdChemReactions
from rdkit.Chem import rdmolops

# Örnek reaction havuzu (kendi ihtiyacına göre genişletilebilir)
REACTION_SMARTS = [
    "[C:1][H:2]>>[C:1]Cl",
    "[C:1][H:2]>>[C:1]O",
    "[C:1](=O)[O;H1].[O;H1][C:2]>>[C:1](=O)O[C:2]",
    "[C:1](=O)Cl.[N:2]>>[C:1](=O)N",
    "[O:1][H].[C:2]Br>>[O:1][C:2]",
    "c1ccccc1>>c1([N+](=O)[O-])ccccc1",
    "[C:1]=[C:2]>>[C:1]-[C:2]"
]

RDKit_REACTIONS = [rdChemReactions.ReactionFromSmarts(s) for s in REACTION_SMARTS]

def chemically_valid_mutate(p_smi: str, reactions=RDKit_REACTIONS, attempts=6):
    """Reaction tabanlı mutasyon uygular; başarısızsa fallback döner."""
    def sanitize_and_canonicalize(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            rdmolops.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None

    def replace_star_with_H(smi: str):
        return str(smi).replace('*', '[H]')

    def restore_H_to_star(smi: str):
        return str(smi).replace('[H]', '*')

    def is_reasonable_product(prod_smiles, max_atoms=120, min_atoms=4):
        if prod_smiles is None: return False
        try:
            m = Chem.MolFromSmiles(prod_smiles)
            if m is None: return False
            n = m.GetNumAtoms()
            if n > max_atoms or n < min_atoms: return False
            try: rdmolops.SanitizeMol(m)
            except: return False
            return True
        except: return False

    # 1. Prepare
    base = replace_star_with_H(p_smi)
    base_mol = Chem.MolFromSmiles(base)
    if base_mol is None: return p_smi

    # 2. Reaction denemeleri
    candidate_products = []
    for _ in range(attempts):
        rxn = random.choice(reactions)
        try:
            ps = rxn.RunReactants((base_mol,))
        except:
            ps = ()
        for prod_tuple in ps:
            for prod_mol in prod_tuple:
                try:
                    prod_smiles = Chem.MolToSmiles(prod_mol, canonical=True)
                except: prod_smiles = None
                prod_restored = restore_H_to_star(prod_smiles) if prod_smiles else None
                if is_reasonable_product(prod_restored):
                    candidate_products.append(prod_restored)

    # 3. Sonuç
    if candidate_products:
        out = random.choice(candidate_products)
        if out == p_smi or len(out) < max(4, len(p_smi)//2):
            return p_smi
        return out
    return p_smi

# =========================
# 5. Offspring Üretim Fonksiyonu
# =========================

mutation_stats = {'SELFIES':0, 'REACTION':0, 'EXTEND':0, 'NEW':0}

def generate_offspring(individual, initial_selfies, mutpb=0.05, extendpb=0.05, newpb=0.01, chempb=0.05):
    """Mutasyon, zincir uzatma, yeni birey ve reaction mutasyonunu uygular."""
    # 1. SELFIES mutasyonu
    if random.random() < mutpb:
        individual = mutSelfies(individual)
        mutation_stats['SELFIES'] += 1

    # 2. Reaction tabanlı mutasyon
    if random.random() < chempb:
        smi = selfies_to_smiles_safe(individual[0])
        if smi:
            mutated = chemically_valid_mutate(smi)
            ind_selfies = smiles_to_selfies_safe(mutated)
            if ind_selfies:
                individual[0] = ind_selfies
                mutation_stats['REACTION'] += 1

    # 3. Zincir uzatma
    if random.random() < extendpb:
        individual[0] = extendPolymer(individual)
        mutation_stats['EXTEND'] += 1

    # 4. Rastgele yeni birey
    if random.random() < newpb:
        individual[0] = random.choice(initial_selfies)
        mutation_stats['NEW'] += 1

    # 5. Geçerlilik kontrolü
    if not is_valid_polymer(individual[0]):
        individual[0] = random.choice(initial_selfies)
    return individual

# =========================
# 6. run_single_objective_flow Güncellemesi
# =========================
def run_single_objective_flow(models, generations, targets, active_props, initial_pop, ranges_dict):
    # --- DEAP Kurulumu ---
    toolbox = base.Toolbox()
    toolbox.register("attr_selfies", random.choice, initial_pop)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_selfies, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Optimize edilmiş evaluate fonksiyonu
    toolbox.register("evaluate", evaluate_individual_optimized, models=models, targets=targets, active_props=active_props, ranges=ranges_dict)
    
    toolbox.register("mate", cxSelfies)
    toolbox.register("select", tools.selTournament, tournsize=7)
    
    pop_size = 100 
    pop = toolbox.population(n=pop_size)
    
    # --- PERFORMANS TAKİP VERİ YAPISI ---
    history = {
        "gen": [],
        "best_fitness": [],
        "avg_fitness": [],
        "diversity": [] # Standart sapma
    }

    # İlk değerlendirme
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # --- UI Elementleri (Canlı Dashboard) ---
    st.markdown("### 🧬 Evrimsel Süreç İzleme Paneli")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Grafikler için yan yana iki kolon (Canlı güncellenecek)
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.caption("📉 Yakınsama (Convergence)")
        chart_fitness_placeholder = st.empty()
    with col_chart2:
        st.caption("🌊 Popülasyon Çeşitliliği (Diversity)")
        chart_diversity_placeholder = st.empty()

    log_expander = st.expander("📝 GA Logları (Detay)", expanded=False)
    with log_expander:
        log_placeholder = st.empty()
        mutation_placeholder = st.empty()

    log_data = [] 

    # --- ANA DÖNGÜ ---
    for gen in range(generations):
        # Adaptif oranlar
        scale = gen / generations
        cxpb = 0.7 - (0.2 * scale)
        mutpb = 0.05 - (0.2 * scale)
        extendpb = 0.05 - (0.15 * scale)
        newpb = 0.01 - (0.05 * scale)
        chempb = 0.05 - (0.15 * scale)

        # Seçilim ve Klonlama
        offspring = toolbox.select(pop, pop_size)
        offspring = list(map(toolbox.clone, offspring))

        # Çaprazlama
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        # Mutasyon
        for i in range(len(offspring)):
            if not offspring[i].fitness.valid: 
                 pass
            offspring[i] = generate_offspring(offspring[i], initial_pop, mutpb=mutpb, extendpb=extendpb, newpb=newpb, chempb=chempb)
            del offspring[i].fitness.values

        # Değerlendirme
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = offspring
        
        # --- İstatistik Toplama (Kritik Bölüm) ---
        # --- ESKİ KOD (SİLİN) ---
        # fits = [ind.fitness.values[0] for ind in pop]
        # best_val = min(fits)
        # mean_val = sum(fits) / len(pop)
        # std_val = np.std(fits)

        # --- YENİ KOD (BUNU YAPIŞTIRIN) ---
        fits = [ind.fitness.values[0] for ind in pop]
        
        # 1. Sadece "Canlı" (Geçerli) bireyleri filtrele (Ceza puanı 999'dan küçük olanlar)
        valid_fits = [f for f in fits if f < 999.0]
        
        # 2. İstatistikleri sadece canlılar üzerinden hesapla
        if valid_fits:
            best_val = min(valid_fits) # Zaten min değişmez ama garanti olsun
            mean_val = sum(valid_fits) / len(valid_fits) # GERÇEK ORTALAMA
            std_val = np.std(valid_fits) # GERÇEK ÇEŞİTLİLİK
        else:
            # Herkes öldüyse (Çok nadir olur)
            best_val = 1000.0
            mean_val = 1000.0
            std_val = 0.0

        # 3. Hayatta Kalma Oranını Hesapla (Survival Rate)
        survival_rate = (len(valid_fits) / len(pop)) * 100
        
        # Geçmişe kaydet (survival_rate'i de ekleyebilirsin istersen)
        history["gen"].append(gen)
        history["best_fitness"].append(best_val)
        history["avg_fitness"].append(mean_val)
        history["diversity"].append(std_val)
        
        log_data.append({
            "Nesil": gen + 1,
            "En İyi Hata": round(best_val, 4),
            "Ortalama (Valid)": round(mean_val, 4),
            "Canlılık Oranı %": round(survival_rate, 1) # Log tablosunda bunu görmek çok işinize yarar
        })

        # --- UI Güncelleme (Her adımda veya 2 adımda bir) ---
        if gen % 2 == 0 or gen == generations - 1:
            progress_bar.progress((gen + 1) / generations)
            status_text.markdown(f"**Nesil {gen+1}/{generations}** | En İyi Hata: `{best_val:.4f}` | Çeşitlilik: `{std_val:.4f}`")
            
            # 1. Fitness Grafiği Verisi
            df_fit = pd.DataFrame({
                "En İyi (Best)": history["best_fitness"],
                "Ortalama (Avg)": history["avg_fitness"]
            })
            chart_fitness_placeholder.line_chart(df_fit, height=250)
            
            # 2. Diversity Grafiği Verisi
            df_div = pd.DataFrame({
                "Çeşitlilik (Std Dev)": history["diversity"]
            })
            # Çeşitlilik grafiğini kırmızı tonla göstermek için (Streamlit varsayılanı kullanır ama veri tek kolon)
            chart_diversity_placeholder.line_chart(df_div, height=250)
            
            # Log Tablosu
            df_log = pd.DataFrame(log_data)
            log_placeholder.dataframe(df_log.sort_values(by="Nesil", ascending=False).head(5), use_container_width=True)
            mutation_placeholder.json(mutation_stats)

    # Sonuç
    best_ind = tools.selBest(pop, 5)[0]
    best_smiles = selfies_to_smiles_safe(best_ind[0])
    
    # ... (Yukarıdaki GA döngüsü bittikten sonra) ...

    # Sonuç - En iyi bireyi seç
    best_ind = tools.selBest(pop, 5)[0]
    best_smiles = selfies_to_smiles_safe(best_ind[0])
    
    if best_smiles:
        # 1. Standart Fingerprint (Eski modeller için)
        fp = get_morgan_fp(best_smiles)
        
        # 2. Gaz Modeli için Gelişmiş Özellikler (Yeni model için)
        gas_features = get_gas_features_combined(best_smiles)

        preds = {}
        
        # Tüm modeller için tahmin yaparken ayrım yapmalıyız
        for prop in models.keys():
            # ÖZEL DURUM: GasPerma
            if prop == 'GasPerma':
                if gas_features is not None:
                    # Model 2054 özellik bekler
                    log_pred = models[prop].predict(gas_features)[0]
                    # Log10'u geri çevir (10^x)
                    preds[prop] = 10 ** log_pred
                else:
                    preds[prop] = 0.0
            
            # STANDART DURUM: Diğer modeller (Tg, Td, vs.)
            else:
                # Modeller sadece 2048 özellik (fp) bekler
                preds[prop] = models[prop].predict(fp)[0]
        
        # History sözlüğünü döndürüyoruz
        return {'smiles': best_smiles, 'preds': preds, 'total_error': best_ind.fitness.values[0]}, history
    else:
        return None, history

import requests

@st.cache_data
def check_pubchem_availability(smiles: str):
    """
    Verilen SMILES için PubChem'de kayıtlı mı kontrol eder.
    Yıldızları (*) temizleyerek arama yapar.
    """
    # DÜZELTME: Yıldızları temizle veya Hidrojene çevir
    clean_smi = smiles.replace('*', '') 
    
    # URL encoded hale getirmek gerekebilir ama requests bunu genelde yapar.
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{clean_smi}/cids/JSON"
    
    try:
        response = requests.get(url, timeout=5)
        
        # 404 (Bulunamadı) normaldir, hata fırlatmasın
        if response.status_code == 404:
            return False, None, None
            
        response.raise_for_status()
        data = response.json()
        
        if "IdentifierList" in data and "CID" in data["IdentifierList"]:
            cid = data["IdentifierList"]["CID"][0]
            
            # İsim sorgusu
            name_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON"
            name_resp = requests.get(name_url, timeout=5)
            if name_resp.status_code == 200:
                name_data = name_resp.json()
                name = name_data["PropertyTable"]["Properties"][0].get("IUPACName", "Bilinmiyor")
            else:
                name = "Bilinmiyor"
                
            return True, cid, name
        else:
            return False, None, None
            
    except Exception:
        # Hata olsa bile uygulamayı durdurma, sessizce geç
        return False, None, None
# --- TİCARİ KONTROL FONKSİYONU ---
def check_commercial_availability(query):
    """
    Verilen ismi veya SMILES'ı PubChem'de arar.
    Ticari olarak satılıp satılmadığını (Vendor sayısı) kontrol eder.
    """
    try:
        # İsim veya SMILES ile arama yap
        compounds = pcp.get_compounds(query, 'name')
        if not compounds:
            compounds = pcp.get_compounds(query, 'smiles')
            
        if compounds:
            cid = compounds[0].cid
            # PubChem'den "Vendor" (Satıcı) bilgisini çekmek biraz daha karmaşıktır,
            # bu yüzden basitçe "Kayıt var mı?" kontrolü yapıyoruz.
            # Kayıt varsa %99 ticaridir veya sentezlenebilir.
            synonyms = compounds[0].synonyms
            common_name = synonyms[0] if synonyms else query
            return True, cid, common_name
        else:
            return False, None, None
    except:
        return False, None, None
def make_3d_view_with_reason(smiles):
    try:
        clean_smi = str(smiles).replace('*', '[H]')
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is None:
            return None, "SMILES geçersiz veya RDKit ile molekül oluşturulamadı."
        
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) != 0:
            return None, "3D koordinatlar hesaplanamadı (Embed başarısız)."
        
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            return None, "3D yapı enerji optimizasyonunda başarısız."
        
        mblock = Chem.MolToMolBlock(mol)
        view = py3Dmol.view(width=400, height=400)
        view.addModel(mblock, 'mol')
        view.setStyle({'stick':{'colorscheme':'Jmol'}})
        view.zoomTo()
        view.spin(True)
        return view, None
    except Exception as e:
        return None, f"Beklenmeyen bir hata: {e}"

def get_ai_interpretation(api_key, smiles, preds, targets, active_props):
    """Gemini API kullanarak polimer analizi yapar."""
    if not api_key:
        return "⚠️ Analiz için lütfen sol menüden geçerli bir Google Gemini API Anahtarı giriniz."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') # Hızlı ve ekonomik model
        
        # Dinamik Prompt Hazırlama
        prompt = f"""
        Sen uzman bir Polimer Kimyagerisin ve Malzeme Bilimci'sin. 
        Aşağıda genetik algoritma ile üretilmiş yeni bir polimer adayı var.
        
        Molekül (SMILES): {smiles}
        
        Tahmin Edilen Özellikler:
        """
        
        for prop in active_props:
            target_val = targets.get(prop, "Belirtilmedi")
            pred_val = preds.get(prop, 0.0)
            prompt += f"- {prop}: Tahmin={pred_val:.2f} (Hedef={target_val})\n"
            
        prompt += """
        
        Lütfen bu polimeri şu başlıklar altında Türkçe olarak detaylıca analiz et:
        1. **Yapı-Özellik İlişkisi:** Bu yapısal özellikler (halkalar, fonksiyonel gruplar, zincir uzunluğu vb.) neden bu tahmin değerlerini (özellikle Tg ve Td) ortaya çıkarmış olabilir? Kimyasal mantığı nedir?
        2. **Potansiyel Uygulama Alanları:** Bu özelliklere sahip bir polimer endüstride nerede kullanılabilir? (Örn: Havacılık, paketleme, elektronik, membran vb.)
        3. **Sentezlenebilirlik Yorumu:** Yapıya bakarak sentez zorluğu veya stabilite hakkında kısa bir yorum yap.
        
        Yanıtın profesyonel, bilimsel ama anlaşılır olsun. Markdown formatı kullan.
        """
        
        with st.spinner('Yapay Zeka polimeri inceliyor...'):
            response = model.generate_content(prompt)
            return response.text
            
    except Exception as e:
        return f"❌ AI Bağlantı Hatası: {str(e)}"
# --- SA Score Fonksiyonu ---

def get_sa_score_local(p_smiles):
    """
    Yerel SA Score Hesaplayıcı.
    Eğer klasörde 'sascorer.py' varsa onu kullanır, yoksa basit hesaplama yapar.
    """
    try:
        import sascorer
        smi_clean = str(p_smiles).replace('*', '[H]').replace('(*)', '[H]').replace('[*]', '[H]')
        mol = Chem.MolFromSmiles(smi_clean)
        if mol is None: 
            raise ValueError("Mol oluşturulamadı")
        return sascorer.calculateScore(mol)
    except:
        # Basit yedek hesaplama: uzunluk ve halka sayısına göre
        length = len(str(p_smiles))
        score = 2.0 + (length * 0.05)
        if "c1" in str(p_smiles): 
            score += 0.5
        return min(score, 10.0)
# =========================================================================
# VII. YEŞİL KİMYA / SÜRDÜRÜLEBİLİRLİK MOTORU
# =========================================================================

def calculate_green_score(smiles):
    """
    Polimerin potansiyel biyo-bozunurluğunu ve çevresel etkisini puanlar.
    Puan: 1 (Çok Kötü/Kalıcı) - 10 (Mükemmel/Bozunabilir)
    """
    mol = Chem.MolFromSmiles(smiles.replace('*', '[H]'))
    if not mol: return 0, "Hesaplanamadı", "#7f8c8d"
    
    score = 5.0 # Nötr başlangıç
    notes = []
    
    # --- 1. BOZUNABİLİR BAĞLAR (Pozitif) ---
    # Ester Bağı: Hidroliz olur, doğada parçalanır (Örn: PLA)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C;!R](=[O])[O;!R]")):
        score += 3.0
        notes.append("Ester bağı (Hidroliz olabilir)")
        
    # Amid Bağı: Enzimlerle parçalanabilir (Örn: Proteinler, Naylon)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C;!R](=[O])[N;!R]")):
        score += 2.0
        notes.append("Amid bağı (Biyo-bozunurluk potansiyeli)")
        
    # Eter Bağı (PEG gibi): Suda çözünürlük sağlar, biyolojik atılımı kolaylaştırır
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[C][O][C]")):
        score += 1.0
        notes.append("Eter grubu (Hidrofilik özellik)")

    # --- 2. KALICILIK ve TOKSİSİTE (Negatif) ---
    # Halojenler (F, Cl, Br): Doğada birikim yapar, toksiktir (Örn: PVC, Teflon)
    halogens = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() in ['F', 'Cl', 'Br']]
    if halogens:
        count = len(halogens)
        penalty = min(4.0, count * 1.0) # En fazla 4 puan kır
        score -= penalty
        notes.append(f"{count} adet Halojen atomu (Kalıcılık/Toksisite riski)")
        
    # Aromatik Halkalar: Bakterilerin parçalaması zordur
    aromatic_atoms = [atom for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    if len(aromatic_atoms) > 4: # Çok fazla halka varsa
        score -= 2.0
        notes.append("Yüksek Aromatiklik (Zor parçalanma)")

    # --- 3. SONUÇ SINIRLANDIRMA ---
    score = max(1.0, min(10.0, score)) # 1-10 arasına sabitle
    
    # Renk Kodu Belirle
    if score >= 7.0: color = "#2ecc71" # Yeşil (İyi)
    elif score >= 4.0: color = "#f1c40f" # Sarı (Orta)
    else: color = "#e74c3c" # Kırmızı (Kötü)
    
    return score, ", ".join(notes), color
import plotly.graph_objects as go

def create_radar_chart(preds, targets, active_props, ranges):
    """
    Hedeflenen özellikler ile tahmin edilen özellikleri karşılaştıran
    havalı bir Radar (Spider) Grafiği çizer.
    """
    categories = []
    target_values = []
    pred_values = []
    
    for prop in active_props:
        if prop in preds and prop in targets:
            # Başlıkları güzelleştir
            label = prop
            if prop == 'ThermalCond': label = 'Iletkenlik'
            if prop == 'Solubility': label = 'Cozunurluk'
            
            categories.append(label)
            
            # Değerleri al
            t_val = targets[prop]
            p_val = preds[prop]
            
            # Normalizasyon (Grafikte düzgün durması için 0-1 arasına çekiyoruz)
            # Min-Max normalizasyonu
            min_v = ranges[prop]['min']
            max_v = ranges[prop]['max']
            
            # Sıfıra bölme hatası önlemi
            if max_v - min_v == 0: denom = 1
            else: denom = max_v - min_v
            
            norm_t = (t_val - min_v) / denom
            norm_p = (p_val - min_v) / denom
            
            # Sınırlandırma (Grafik dışına taşmasın)
            norm_t = max(0.0, min(1.0, norm_t))
            norm_p = max(0.0, min(1.0, norm_p))
            
            target_values.append(norm_t)
            pred_values.append(norm_p)
            
    # Grafiği kapatmak için ilk değeri sona ekle
    categories = categories + [categories[0]]
    target_values = target_values + [target_values[0]]
    pred_values = pred_values + [pred_values[0]]
    
    fig = go.Figure()

    # Hedef Alanı (Mavi Çizgi)
    fig.add_trace(go.Scatterpolar(
        r=target_values,
        theta=categories,
        fill='toself',
        name='Hedeflenen',
        line=dict(color='#3498db', dash='dash')
    ))
    
    # Tahmin Alanı (Kırmızı Dolgu)
    fig.add_trace(go.Scatterpolar(
        r=pred_values,
        theta=categories,
        fill='toself',
        name='Uretilen Polimer',
        line=dict(color='#e74c3c'),
        opacity=0.7
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] # Normalize ettiğimiz için 0-1 arası
            )),
        showlegend=True,
        margin=dict(l=40, r=40, t=20, b=20),
        height=300 # Kompakt boyut
    )
    
    return fig
# =========================================================================
# VI. RETROSENTEZ MOTORU (Yeni Eklenen Kısım)
# =========================================================================

# =========================================================================
# GÜNCELLENMİŞ RETROSENTEZ MOTORU (Imide Desteği Eklendi)
# =========================================================================

# =========================================================================
# GÜNCELLENMİŞ RETROSENTEZ MOTORU v2.0 (Akıllı Varsayılan Eklendi)
# =========================================================================

# =========================================================================
# GÜNCELLENMİŞ RETROSENTEZ MOTORU v3.0 (Poliüre & Poliüretan Eklendi)
# =========================================================================

def decompose_polymer(smiles):
    """
    Polimeri parçalar. v3.0: Üre ve Üretan bağlarını da tanır.
    """
    clean_smi = smiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(clean_smi)
    if not mol: return None, "Geçersiz Molekül"
    
    breakdown_results = []
    
    # --- KURAL 1: İMİD (Poliimid) ---
    imide_pattern = Chem.MolFromSmarts("[CX3](=[OX1])[#7][CX3](=[OX1])")
    if mol.HasSubstructMatch(imide_pattern):
        return [{
            "type": "Poliimid Sentezi",
            "reaction": "Siklo-dehidrasyon",
            "monomers": ["Dianhidrit", "Diamin"],
            "mechanism": "Dianhidrit + Diamin -> Poliimid"
        }]

    # --- KURAL 2: ÜRE (Polyurea) --- [YENİ]
    # R-NH-C(=O)-NH-R' -> R-N=C=O (İzosiyanat) + H2N-R' (Amin)
    urea_pattern = Chem.MolFromSmarts("[N;!R][C;!R](=[O])[N;!R]")
    if mol.HasSubstructMatch(urea_pattern):
        breakdown_results.append({
            "type": "Poliüre (Polyurea) Sentezi",
            "reaction": "Basamaklı Polimerizasyon (Hızlı)",
            "monomers": ["Diizosiyanat (Diisocyanate)", "Diamin (Diamine)"],
            "mechanism": "İzosiyanat + Amin -> Üre Bağı (Yan ürün yok)"
        })

    # --- KURAL 3: ÜRETAN (Polyurethane) --- [YENİ]
    # R-NH-C(=O)-O-R' -> R-N=C=O (İzosiyanat) + HO-R' (Alkol/Polyol)
    urethane_pattern = Chem.MolFromSmarts("[N;!R][C;!R](=[O])[O;!R]")
    if mol.HasSubstructMatch(urethane_pattern):
        breakdown_results.append({
            "type": "Poliüretan (PU) Sentezi",
            "reaction": "Poliladisyon",
            "monomers": ["Diizosiyanat (Örn: TDI, MDI)", "Diol / Polyol"],
            "mechanism": "İzosiyanat + Alkol -> Üretan Bağı"
        })

    # --- KURAL 4: ESTER (Polyester) ---
    ester_pattern = Chem.MolFromSmarts("[C;!R](=[O])[O;!R]") 
    if mol.HasSubstructMatch(ester_pattern) and not breakdown_results: # Üretan değilse bak
        breakdown_results.append({
            "type": "Polyester Sentezi",
            "reaction": "Kademeli Polimerizasyon",
            "monomers": ["Dikarboksilik Asit", "Diol"],
            "mechanism": "Asit + Alkol -> Ester + Su"
        })

    # --- KURAL 5: AMİD (Nylon) ---
    amide_pattern = Chem.MolFromSmarts("[C;!R](=[O])[N;!R]")
    if mol.HasSubstructMatch(amide_pattern) and not breakdown_results: # Üre değilse bak
         breakdown_results.append({
            "type": "Poliamid (Nylon) Sentezi",
            "reaction": "Polikondenzasyon",
            "monomers": ["Dikarboksilik Asit", "Diamin"],
            "mechanism": "Asit + Amin -> Amid + Su"
        })

    # --- VARSAYILAN ---
    if not breakdown_results:
        # Akıllı kontrol: Azot/Oksijen var mı?
        has_hetero = any(atom.GetSymbol() in ['N', 'O', 'S'] for atom in mol.GetAtoms())
        if has_hetero and "C=C" not in smiles:
             breakdown_results.append({
                "type": "Kompleks Kondenzasyon Polimeri",
                "reaction": "Özel Sentez (AI Analizi Önerilir)",
                "monomers": ["Fonksiyonel Grup A", "Fonksiyonel Grup B"],
                "mechanism": "Uç grupların reaksiyonu"
            })
        else:
            breakdown_results.append({
                "type": "Vinil Polimerizasyonu (Katılma)",
                "reaction": "Radikalik",
                "monomers": [smiles.replace('*', '')],
                "mechanism": "Çift bağ açılması"
            })
            
    return breakdown_results
def draw_retrosynthesis_grid(monomer_smiles_list):
    """Monomerlerin listesini alır ve yan yana çizer."""
    mols = [Chem.MolFromSmiles(s) for s in monomer_smiles_list]
    mols = [m for m in mols if m is not None] # Hatalıları temizle
    if not mols: return None
    
    img = Draw.MolsToGridImage(
        mols, 
        molsPerRow=min(len(mols), 3), 
        subImgSize=(200, 200),
        legends=[f"Monomer {i+1}" for i in range(len(mols))]
    )
    return img

def get_ai_retrosynthesis_guide(api_key, polymer_smiles, monomer_info):
    """Gemini'den detaylı sentez rotası ister."""
    if not api_key: return "⚠️ Detaylı sentez planı için API Key gerekli."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Sen uzman bir Sentetik Polimer Kimyagerisin.
        Aşağıdaki polimer için endüstriyel veya laboratuvar ölçekli bir RETROSENTEZ (geriye dönük sentez) planı hazırla.
        
        Hedef Polimer (SMILES): {polymer_smiles}
        Algarlanan Olası Yöntem: {monomer_info}
        
        Lütfen şu formatta yanıtla:
        1. **Önerilen Monomerler:** Bu yapıyı oluşturmak için hangi ticari kimyasallar (IUPAC isimleri) gerekir?
        2. **Sentez Yöntemi:** Hangi reaksiyon türü uygundur? (Örn: Radikalik, Kondenzasyon, ROMP?)
        3. **Kritik Koşullar:** Sıcaklık, basınç veya spesifik katalizör (AIBN, Ziegler-Natta, H2SO4 vb.) önerisi.
        4. **Zorluk Analizi:** Bu sentezin pratik zorlukları nelerdir?
        
        Kısa, net ve bilimsel olsun.
        """
        with st.spinner('AI Sentez Rotasını Hesaplıyor...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"Hata: {str(e)}"
from fpdf import FPDF
import tempfile

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'PolimerX - Ar-Ge Proje Raporu', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

def clean_text(text):
    """FPDF için Türkçe karakterleri ASCII'ye çevirir (Hızlı çözüm)"""
    replacements = {
        'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U', 'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I', 'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf_report(poly_data, targets, active_props, ai_analysis_text, retro_info):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # 1. Başlık Bilgileri
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("1. Polimer Özellik Tablosu"), 0, 1)
    pdf.set_font("Arial", size=10)
    
    # Tablo Başlığı
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(60, 8, "Ozellik", 1, 0, 'C', 1)
    pdf.cell(60, 8, "Hedef", 1, 0, 'C', 1)
    pdf.cell(60, 8, "Tahmin Degeri", 1, 1, 'C', 1)
    
    # --- GÜNCELLEME: Tüm tahminleri döngüye al ---
    all_preds = poly_data['preds']
    
    for prop, val in all_preds.items():
        # Hedeflenen değer var mı kontrol et
        if prop in active_props:
            target_val = str(targets.get(prop, '-'))
            # Hedeflenenleri kalın (bold) veya işaretli gösterebiliriz ama
            # şimdilik standart formatta yazıyoruz.
        else:
            target_val = "-" # Hedef belirtilmedi
            
        pred_val = f"{val:.2f}"
        
        # Satırı yaz
        pdf.cell(60, 8, clean_text(prop), 1)
        pdf.cell(60, 8, target_val, 1)
        pdf.cell(60, 8, pred_val, 1, 1)
    
    pdf.ln(10)
    
    # 2. Molekül Görseli
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("2. Molekuler Yapi"), 0, 1)
    
    # SMILES
    pdf.set_font("Courier", size=8)
    pdf.multi_cell(0, 5, poly_data['smiles'])
    pdf.ln(5)
    
    # Görseli ekle
    mol_img = draw_2d_molecule(poly_data['smiles'])
    if mol_img:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            mol_img.save(tmp_file.name)
            pdf.image(tmp_file.name, x=60, w=90)
    
    pdf.ln(10)
    
    # 3. Retrosentez Bilgisi
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("3. Uretim Plani (Retrosentez)"), 0, 1)
    pdf.set_font("Arial", size=10)
    
    if not retro_info or len(retro_info) < 5:
        pdf.multi_cell(0, 6, clean_text("Retrosentez analizi yapilmadi veya veri yok."))
    else:
        clean_retro = clean_text(str(retro_info))
        pdf.multi_cell(0, 6, clean_retro)
    
    pdf.ln(10)
    
    # 4. AI Yorumu
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("4. Yapay Zeka Uzman Görüşü"), 0, 1)
    pdf.set_font("Arial", size=10)
    
    if not ai_analysis_text or len(ai_analysis_text) < 5:
        pdf.multi_cell(0, 6, clean_text("AI analizi talep edilmedi."))
    else:
        clean_ai = clean_text(ai_analysis_text).replace('**', '').replace('#', '')
        pdf.multi_cell(0, 6, clean_ai)
    
    return pdf.output(dest='S').encode('latin-1')
from rdkit import DataStructs

@st.cache_data
def get_reference_fingerprints(smiles_list):
    """
    Referans veri setindeki tüm SMILES'ların parmak izlerini önceden hesaplar ve önbelleğe alır.
    Bu işlem sadece bir kez yapılır, böylece uygulama hızlanır.
    """
    fps = []
    names = [] # Varsa isimleri, yoksa SMILES'ın kendisi
    
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(str(smi).replace('*', '[H]'))
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
                fps.append(fp)
                names.append(f"Veri Seti Kaydı #{i+1}") # Veya smi
        except:
            continue
    return fps, names

def calculate_novelty_optimized(generated_smiles, ref_smiles_list):
    """
    Toplu Tanimoto benzerliği hesaplar (Çok hızlıdır).
    """
    # 1. Üretilen molekülün parmak izi
    gen_mol = Chem.MolFromSmiles(generated_smiles.replace('*', '[H]'))
    if not gen_mol: return 0.0, "Hesaplanamadı"
    gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 3, 2048)
    
    # 2. Referans parmak izlerini önbellekten çek
    ref_fps, ref_names = get_reference_fingerprints(ref_smiles_list)
    
    if not ref_fps: return 0.0, "Veri Seti Boş"
    
    # 3. RDKit'in Toplu (Bulk) Karşılaştırma Fonksiyonu
    # Bu döngüden 100 kat daha hızlıdır.
    sims = DataStructs.BulkTanimotoSimilarity(gen_fp, ref_fps)
    
    # 4. En yüksek benzerliği bul
    max_sim = max(sims)
    max_idx = sims.index(max_sim)
    most_similar_name = ref_names[max_idx]
    
    # Eşleşen SMILES'ı döndürmek daha bilgilendirici olabilir
    most_similar_smiles = ref_smiles_list[max_idx] if max_idx < len(ref_smiles_list) else "Bilinmiyor"
    
    return max_sim, most_similar_smiles
# =========================================================================
# IV. STREAMLIT ANA KISIM
# =========================================================================

# st.title("...") yerine:

st.markdown('<h1 class="main-title">🧬 PolimerX <br><span style="font-size:1.5rem; color:#666; font-weight:400;">Yapay Zeka Destekli Materyal Keşfi</span></h1>', unsafe_allow_html=True)

models = load_critic_models()
ALL_PROPS = list(models.keys()) # Yüklenen modellerin anahtarları: ['Tg', 'Td', 'EPS']

# --- Yardımcı: Senkron Slider + Number input ---
def add_synced_input(prop_key, label, min_val, max_val, default, step, is_int=False):
    """Sidebar üzerinde bir slider ve number_input oluşturur; ikisini session_state üzerinden senkronlar.
    Döndürülen değer her zaman current value (float/int) olur.
    """
    s_key = f"{prop_key}_val"
    slider_key = f"{prop_key}_slider"
    num_key = f"{prop_key}_num"

    # Başlangıç değeri session_state'e konur
    if s_key not in st.session_state:
        st.session_state[s_key] = default
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state[s_key]
    if num_key not in st.session_state:
        st.session_state[num_key] = st.session_state[s_key]

    def _on_slider_change():
        # slider değiştiğinde number_input değerini güncelle
        try:
            st.session_state[num_key] = st.session_state[slider_key]
            st.session_state[s_key] = st.session_state[slider_key]
        except Exception:
            pass

    def _on_num_change():
        # number_input değiştiğinde slider'ı güncelle
        try:
            st.session_state[slider_key] = st.session_state[num_key]
            st.session_state[s_key] = st.session_state[num_key]
        except Exception:
            pass

    # Slider (min/max tipi int/float ile uyumlu olmalı)
    if is_int:
        st.sidebar.slider(label + " (slider)", min_value=int(min_val), max_value=int(max_val), step=int(step), key=slider_key, on_change=_on_slider_change)
        st.sidebar.number_input(label + " (value)", min_value=int(min_val), max_value=int(max_val), step=int(step), key=num_key, on_change=_on_num_change)
    else:
        st.sidebar.slider(label + " (slider)", min_value=float(min_val), max_value=float(max_val), step=float(step), key=slider_key, on_change=_on_slider_change)
        st.sidebar.number_input(label + " (value)", min_value=float(min_val), max_value=float(max_val), step=float(step), format="%.4f", key=num_key, on_change=_on_num_change)

    return st.session_state[s_key]

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
    
    if st.sidebar.checkbox("LOI (Yanıcılık İndeksi)"): 
        active_props.append('LOI')
        
    if st.sidebar.checkbox("Çözünürlük (Hildebrand)"): 
        active_props.append('Solubility')
        
    if st.sidebar.checkbox("Isıl İletkenlik (Thermal Cond.)"): 
        active_props.append('ThermalCond')
        
    if st.sidebar.checkbox("Isıl Genleşme (CTE)"): 
        active_props.append('CTE')
     # En az bir hedef seçilmemişse uyarı ver
        
    if not active_props:
        st.sidebar.warning("Lütfen optimize edilecek en az bir hedef seçin.")
        st.stop()

    # 2. Hedef Değerler (Sadece seçilenler için giriş alanı göster)
    st.sidebar.markdown("### Hedef Değerler")
    targets = {}

    # Önerilen aralıklar (kullanıcının onayladığı değerler)
    # Sıcaklıklar (°C)
    ranges = {
        'Tg': {'min': -150.0, 'max': 300.0, 'default': 200.0, 'step': 1.0, 'is_int': False},
        'Td': {'min': 150.0, 'max': 600.0, 'default': 350.0, 'step': 1.0, 'is_int': False},
        'Tm': {'min': 50.0, 'max': 450.0, 'default': 250.0, 'step': 1.0, 'is_int': False},
        # Diğer özellikler
        'EPS': {'min': 1.5, 'max': 12.0, 'default': 2.5, 'step': 0.1, 'is_int': False},
        'BandgapBulk': {'min': 0.5, 'max': 6.0, 'default': 2.5, 'step': 0.01, 'is_int': False},
        'BandgapChain': {'min': 0.5, 'max': 6.0, 'default': 2.5, 'step': 0.01, 'is_int': False},
        'BandgapCrystal': {'min': 0.5, 'max': 7.0, 'default': 2.5, 'step': 0.01, 'is_int': False},
        'GasPerma': {'min': 0.0, 'max': 1000.0, 'default': 2.5, 'step': 0.1, 'is_int': False},
        'Refractive': {'min': 1.2, 'max': 2.0, 'default': 1.5, 'step': 0.01, 'is_int': False},

        'LOI': {'min': 15.0, 'max': 100.0, 'default': 28.0, 'step': 0.5, 'is_int': False},
        
        # Çözünürlük (Hildebrand): 7-10 arası apolar çözücüler, 12+ polar
        'Solubility': {'min': 5.0, 'max': 20.0, 'default': 9.5, 'step': 0.1, 'is_int': False},
        
        # Isıl İletkenlik: Polimerler genelde 0.1-0.5 arasıdır (yalıtkan)
        'ThermalCond': {'min': 0.0, 'max': 1.0, 'default': 0.2, 'step': 0.01, 'is_int': False},
        
        # CTE (Genleşme): Düşük olması (boyutsal kararlılık) istenir.
        'CTE': {'min': 0.0, 'max': 300.0, 'default': 60.0, 'step': 5.0, 'is_int': False}
    }

    # Her seçili özellik için senkron slider + number_input ekle
    for prop in active_props:
        if prop in ranges:
            r = ranges[prop]
            label = prop
            # Kullanıcıya daha dostça etiket gösterimi
            if prop == 'Tg': label = 'Hedef Tg (°C)'
            elif prop == 'Td': label = 'Hedef Td (°C)'
            elif prop == 'Tm': label = 'Hedef Tm (°C)'
            elif prop == 'EPS': label = 'Hedef EPS'
            elif prop == 'BandgapBulk': label = 'Hedef BandgapBulk (eV)'
            elif prop == 'BandgapChain': label = 'Hedef BandgapChain (eV)'
            elif prop == 'BandgapCrystal': label = 'Hedef BandgapCrystal (eV)'
            elif prop == 'GasPerma': label = 'Hedef GasPerma'
            elif prop == 'Refractive': label = 'Hedef Refractive Index'

            val = add_synced_input(prop, label, r['min'], r['max'], r['default'], r['step'], is_int=r['is_int'])
            targets[prop] = val
        else:
            # Eğer ranges sözlüğünde yoksa varsayılan number_input (güncelleme kolaylığı için)
            targets[prop] = st.sidebar.number_input(f"Hedef {prop}:", value=0.0)

    # 3. GA Parametreleri
    generations = st.sidebar.slider("Evrim Nesli Sayısı", 10, 300, 10)

    # Başlangıç popülasyonu (Gerçek verinizi buraya koyun)
    initial_selfies, reference_smiles = get_initial_population()
    # Sidebar'ın en altına ekleyebilirsiniz
    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 AI Asistan Ayarları")
    api_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="AI yorumu almak için https://aistudio.google.com/app/apikey adresinden ücretsiz anahtar alabilirsiniz.")
    # --- BUTON VE HESAPLAMA KISMI ---
    if st.sidebar.button("🚀 Hedefi Ara", type="primary"):
        
        if not initial_selfies:
            st.error("Başlangıç popülasyonu boş veya geçersiz.")
            st.stop()
            
        with st.spinner(f'Genetik Algoritma Çalışıyor... Hedefler: {", ".join(active_props)}'):
            # Hesaplama yapılıyor
            best_poly_data, history = run_single_objective_flow(models, generations, targets, active_props, initial_selfies, ranges)

        # SONUÇLARI HAFIZAYA (SESSION STATE) KAYDET
        if best_poly_data:
            st.session_state['ga_results'] = best_poly_data
            st.session_state['ga_history'] = history
            st.session_state['ga_targets'] = targets # O anki hedefleri de sakla
            st.session_state['ga_active_props'] = active_props # O anki aktif özellikleri de sakla
            
    # --- SONUÇLARI GÖSTERME KISMI (BUTON BLOĞUNUN DIŞINDA) ---
    
    # Hafızada sonuç varsa ekrana bas (Sayfa yenilense de burası çalışır)
    if 'ga_results' in st.session_state:
        
        # Verileri hafızadan geri çağır
        best_poly_data = st.session_state['ga_results']
        history = st.session_state['ga_history']
        saved_targets = st.session_state['ga_targets']
        saved_active_props = st.session_state['ga_active_props']
        
        preds = best_poly_data['preds']
        
        st.success("✅ Optimizasyon Başarıyla Tamamlandı! (Sonuçlar Hafızada)")
        
        # 4 SEKME YAPISI
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Genel Bakış", "🧬 Yapısal Analiz", "📈 Evrim Geçmişi", "💾 Raporlama", "🤖 AI Analizi", "🧪 Retrosentez"])
        # --- TAB 1: ÖZET ---
        with tab1:
            col_main, col_score, col_green = st.columns([2, 1, 1])
            
            with col_main:
                st.markdown(f"### 🏆 Toplam Hata: **{best_poly_data['total_error']:.4f}**")
                
            with col_score:
                sa = get_sa_score_local(best_poly_data['smiles'])
                st.metric("Sentez Zorluğu (SA)", f"{sa:.2f}", help="1 (Kolay) - 10 (Zor)")
                
            with col_green:
                # Yeni Yeşil Kimya Skorunu Hesapla
                g_score, g_note, g_color = calculate_green_score(best_poly_data['smiles'])
                
                # Özel renkli metrik gösterimi (HTML ile)
                st.markdown(f"""
                <div style="background-color:{g_color}20; border: 1px solid {g_color}; border-radius: 5px; padding: 5px; text-align: center;">
                    <strong style="color:{g_color}; font-size: 0.8rem;">🌱 Yeşil Skor</strong><br>
                    <span style="font-size: 1.5rem; font-weight: bold; color:{g_color};">{g_score:.1f}/10</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Notları altına küçük yazıyla ekle
            if g_note:
                st.caption(f"**Çevresel Analiz:** {g_note}")

            st.divider()
            if 'Solubility' in preds:
                sol_val = preds['Solubility']
                solvents, partials = get_soluble_solvents(sol_val)
                
                st.markdown("### 🧪 Tahmini Çözünürlük Analizi")
                c1, c2 = st.columns(2)
                
                with c1:
                    st.info(f"**Çözünmesi Beklenenler:**")
                    if solvents:
                        # Yeşil etiketlerle göster
                        for s in solvents:
                            st.markdown(f"- ✅ {s}")
                    else:
                        st.warning("Bu polimer çok dirençli görünüyor (veya çok özel bir çözücü gerektiriyor).")
                
                with c2:
                    st.warning(f"**Şişme / Zor Çözünme Beklenenler:**")
                    if partials:
                        for s in partials:
                            st.markdown(f"- ⚠️ {s}")
                    else:
                        st.write("-")
                
                st.caption(f"*Analiz, 'Benzer Benzeri Çözer' ilkesine göre Polimer (δ={sol_val:.1f}) ve Çözücü arasındaki Hildebrand farkına dayanır.*")
            cols = st.columns(3)
            for idx, prop in enumerate(ALL_PROPS):
                with cols[idx % 3]:
                    is_active = prop in saved_active_props
                    target_val = saved_targets.get(prop, '-')
                    target_text = f"Hedef: {target_val}" if is_active else "Takip Dışı"
                    border_color = "#2ecc71" if is_active else "#95a5a6"
                    pred_value = preds[prop]
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid {border_color};">
                        <small>{prop}</small><br>
                        <h3 style="margin:0; padding:0;">{pred_value:.2f}</h3>
                        <small style="opacity:0.7">{target_text}</small>
                    </div>
                    """, unsafe_allow_html=True)
            st.divider()
            st.subheader("🎯 Hedef Uyumluluk Analizi")
            if len(saved_active_props) >= 3:
                    fig = create_radar_chart(preds, saved_targets, saved_active_props, ranges)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                    st.info("Radar grafiği için en az 3 özellik (Örn: Tg, LOI, CTE) seçmelisiniz.")
                    # Az özellik varsa bar chart gösterelim
                    st.progress(100) # Görsel dolgu
        # --- TAB 2: GÖRSELLİK ---
        with tab2:
            col_2d, col_3d = st.columns(2)
            with col_2d:
                st.subheader("2D Yapı (Teknik Çizim)")
                img = draw_2d_molecule(best_poly_data['smiles'])
                if img:
                    st.image(img, width=400)
                st.caption("SMILES Kodu:")
                st.code(best_poly_data['smiles'], language="text")

            with col_3d:
                st.subheader("3D Konformasyon")
                view, reason = make_3d_view_with_reason(best_poly_data["smiles"])
                if view:
                    showmol(view, height=400, width=400)
                else:
                    st.warning(f"3D Model oluşturulamadı: {reason}")
            
            is_avail, cid, name = check_pubchem_availability(best_poly_data['smiles'])
            if is_avail:
                 st.info(f"💡 Bu molekül PubChem'de kayıtlı: **{name}** (CID: {cid})")
            st.divider()
            # --- YENİ: ÖZGÜNLÜK / NOVELTY ANALİZİ ---
            st.subheader("🔍 Özgünlük Analizi (Novelty Search)")
            
            # reference_smiles değişkenini get_initial_population'dan almıştık
            similarity_score, similar_smi = calculate_novelty_optimized(best_poly_data['smiles'], reference_smiles)
            
            c1, c2 = st.columns([1, 3])
            
            with c1:
                # Benzerlik Skoru
                st.metric("Eğitim Setine Benzerlik", f"%{similarity_score*100:.1f}")
                
            with c2:
                # Yorum
                if similarity_score > 0.99:
                    st.error(f"⚠️ **Kopya:** Yapay zeka eğitim setindeki bir veriyi ezberlemiş.")
                    st.code(f"Benzer Kayıt: {similar_smi}")
                elif similarity_score > 0.85:
                    st.warning(f"ℹ️ **Türev:** Eğitim setindeki bir yapıya çok benziyor.")
                    with st.expander("Benzer Yapıyı Gör"):
                        st.code(similar_smi)
                else:
                    st.success(f"🌟 **KEŞİF:** Bu yapı eğitim setinde YOK! Tamamen özgün bir tasarım.")
                    st.caption(f"En yakın benzerlik sadece %{similarity_score*100:.1f} oranında.")
            
            st.progress(similarity_score)
            st.caption("*Benzerlik, Tanimoto İndeksi (Morgan Fingerprints) kullanılarak hesaplanmıştır.*")

        # --- TAB 3: GRAFİK ---
        # --- TAB 3: PERFORMANS ANALİZİ ---
        with tab3:
            st.subheader("📈 Genetik Algoritma Performans Raporu")
            
            if 'best_fitness' in history and len(history['best_fitness']) > 0:
                # Matplotlib ile Profesyonel Çizim
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # X Ekseni
                gens = range(len(history['best_fitness']))
                
                # Grafik 1: Yakınsama (Convergence)
                ax1.plot(gens, history['best_fitness'], label='En İyi Birey (Best)', color='green', linewidth=2)
                ax1.plot(gens, history['avg_fitness'], label='Popülasyon Ortalaması (Avg)', color='blue', linestyle='--', alpha=0.7)
                ax1.set_ylabel('Hata Skoru')
                ax1.set_title('Yakınsama Analizi (Convergence)', fontweight='bold')
                ax1.legend()
                ax1.grid(True, which='both', linestyle='--', alpha=0.5)
                
                # Grafik 2: Çeşitlilik (Diversity)
                ax2.plot(gens, history['diversity'], label='Standart Sapma (Diversity)', color='red', linewidth=2)
                ax2.fill_between(gens, history['diversity'], color='red', alpha=0.1)
                ax2.set_ylabel('Çeşitlilik (Std Dev)')
                ax2.set_xlabel('Jenerasyon')
                ax2.set_title('Popülasyon Çeşitliliği (Diversity)', fontweight='bold')
                ax2.legend()
                ax2.grid(True, which='both', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Yorumlama Kılavuzu
                st.info("""
                **Bu Grafikler Nasıl Okunur?**
                * **Yakınsama (Üst):** Yeşil çizgi sürekli düşmeli ve bir noktada yataylaşmalıdır (Plateau). Mavi çizgi yeşile çok yaklaşırsa popülasyon "öğrenmiş" demektir.
                * **Çeşitlilik (Alt):** Kırmızı çizginin sıfıra çok hızlı düşmemesi gerekir. Sıfıra düşerse model "Erken Yakınsama (Premature Convergence)" tuzağına düşmüştür; yani arama uzayını yeterince taramadan bir sonuca saplanmıştır.
                """)
            else:
                st.warning("Henüz grafik çizilecek veri yok.")
            # --- TAB 3: BENCHMARK VE PERFORMANS ---
        with tab3:
            st.header("🏆 Performans Kıyaslama (Benchmark)")
            st.markdown("Modelin başarısını kanıtlamak için onu 'Rastgele Arama' ile yarıştırın.")
            
            # Eğer GA sonuçları varsa
            if 'ga_history' in st.session_state and 'best_fitness' in st.session_state['ga_history']:
                history = st.session_state['ga_history']
                ga_best_curve = history['best_fitness']
                
                # Benchmark Butonu
                if st.button("🏁 Rastgele Arama ile Kıyasla (Benchmark Başlat)"):
                    with st.spinner("Rastgele Arama yapılıyor... Bu işlem GA kadar sürebilir."):
                        # GA'nın toplam bütçesini hesapla (Jenerasyon x 100 birey)
                        generations_run = len(ga_best_curve)
                        pop_size = 100 # Kodunuzda sabit 100'dü
                        total_evals = generations_run * pop_size
                        
                        # Benchmark'ı çalıştır
                        random_curve = run_random_benchmark(
                            models, saved_targets, saved_active_props, 
                            initial_selfies, ranges, 
                            total_budget=total_evals, 
                            batch_size=pop_size
                        )
                        
                        # Sonucu Session State'e kaydet (Sayfa yenilenince gitmesin)
                        st.session_state['random_curve'] = random_curve
                        st.success("Benchmark Tamamlandı!")

                # --- GRAFİK ÇİZİMİ ---
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 1. GA Çizgisi (Yeşil)
                ax.plot(ga_best_curve, label='Genetik Algoritma (Sizin Modeliniz)', color='green', linewidth=2.5)
                
                # 2. Random Search Çizgisi (Gri/Siyah) - Varsa çiz
                if 'random_curve' in st.session_state:
                    # Uzunlukları eşitle (Bazen 1 eksik/fazla olabilir)
                    min_len = min(len(ga_best_curve), len(st.session_state['random_curve']))
                    r_curve = st.session_state['random_curve'][:min_len]
                    g_curve = ga_best_curve[:min_len]
                    
                    ax.plot(r_curve, label='Rastgele Arama (Random Search)', color='gray', linestyle='--', linewidth=2)
                    
                    # Farkı hesapla (Son jenerasyon)
                    diff = r_curve[-1] - g_curve[-1]
                    st.caption(f"**Sonuç:** GA modeliniz, rastgele aramadan **{diff:.2f} puan** daha iyi performans gösterdi.")
                
                ax.set_title("Zeka Testi: GA vs Şans", fontweight='bold')
                ax.set_xlabel("Jenerasyon (Her adımda 100 yeni deneme)")
                ax.set_ylabel("Hata Skoru (Düşük İyidir)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                
                st.pyplot(fig)
                
                st.info("""
                **Grafik Nasıl Yorumlanır?**
                * **Yeşil Çizgi:** Hızlıca aşağı iniyorsa, modeliniz 'öğreniyor' demektir.
                * **Gri Çizgi:** Genelde daha yukarıda ve düz kalır.
                * **Fark:** İki çizgi arasındaki boşluk, Yapay Zekanızın kattığı değerdir.
                """)
                
            else:
                st.warning("Önce 'Hedefi Ara' butonuna basarak GA'yı çalıştırın, sonra kıyaslama yapabilirsiniz.")
            st.divider()
            st.header("🎲 Büyük Stres Testi (Mass Random Testing)")
            st.markdown("""
            Modelin **genelleştirme yeteneğini** ölçmek için rastgele hedeflerle çoklu deneme yapın.
            * Her denemede farklı özellikler ve farklı hedef değerler seçilir.
            * Modelin "kolay" ve "zor" hedeflere tepkisi ölçülür.
            """)
            
            col_mass_input, col_mass_btn = st.columns([1, 2])
            with col_mass_input:
                mass_trials = st.number_input("Test Sayısı", min_value=10, max_value=500, value=100, step=10)
            
            if col_mass_btn.button("🔥 100+ Rastgele Testi Başlat"):
                with st.spinner("Model zorlu bir sınava giriyor... Kahvenizi alın, bu biraz sürebilir."):
                    df_results = run_mass_random_test(models, generations, initial_selfies, ranges, num_trials=mass_trials)
                    
                    # --- SONUÇ ANALİZİ ---
                    st.subheader("📊 Test Sonuçları")
                    
                    # 1. Özet Metrikler
                    avg_error = df_results["Final Hata Skoru"].mean()
                    success_count = df_results[df_results["Final Hata Skoru"] < 5.0].shape[0]
                    success_rate = (success_count / mass_trials) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Ortalama Hata", f"{avg_error:.2f}")
                    m2.metric("Başarı Oranı (Hata < 5.0)", f"%{success_rate:.1f}")
                    m3.metric("En Zorlu Senaryo Hatası", f"{df_results['Final Hata Skoru'].max():.2f}")
                    
                    # 2. Histogram (Hata Dağılımı)
                    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
                    ax_hist.hist(df_results["Final Hata Skoru"], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
                    ax_hist.set_title("Hata Skorlarının Dağılımı (Histogram)")
                    ax_hist.set_xlabel("Hata Skoru (Sola yığılma iyidir)")
                    ax_hist.set_ylabel("Deneme Sayısı")
                    ax_hist.axvline(avg_error, color='red', linestyle='dashed', linewidth=1, label=f'Ortalama: {avg_error:.2f}')
                    ax_hist.legend()
                    st.pyplot(fig_hist)
                    
                    # 3. Scatter Plot (Zorluk vs Hata)
                    # Hedef sayısı arttıkça hata artıyor mu?
                    fig_sc, ax_sc = plt.subplots(figsize=(10, 5))
                    ax_sc.scatter(df_results["Hedef Sayısı"], df_results["Final Hata Skoru"], alpha=0.6, c=df_results["Final Hata Skoru"], cmap='viridis')
                    ax_sc.set_title("Hedef Sayısı vs. Başarı")
                    ax_sc.set_xlabel("Aktif Hedef Sayısı (Zorluk)")
                    ax_sc.set_ylabel("Hata Skoru")
                    ax_sc.grid(True, alpha=0.3)
                    st.pyplot(fig_sc)
                    
                    # 4. Detaylı Tablo
                    with st.expander("📄 Tüm Test Verilerini Gör"):
                        st.dataframe(df_results)

        # --- TAB 4: İNDİRME ---
# --- TAB 4: RAPORLAMA ve İNDİRME ---
        with tab4:
            st.header("💾 Raporlama Merkezi")
            st.markdown("Proje verilerini CSV veya detaylı PDF raporu olarak dışa aktarabilirsiniz.")
            
            c1, c2 = st.columns(2)
            
            # --- CSV İNDİRME KISMI (Mevcut) ---
            # ... (Buradaki CSV kodlarınız aynen kalabilir) ...
            export_dict = {
                "SMILES": best_poly_data['smiles'],
                "Toplam Hata": best_poly_data['total_error'],
                "SA Score": get_sa_score_local(best_poly_data['smiles'])
            }
            export_dict.update(preds)
            df_best = pd.DataFrame([export_dict])
            csv_best = df_best.to_csv(index=False).encode('utf-8')

            with c1:
                st.download_button(
                    label="📊 Veri Setini İndir (.csv)",
                    data=csv_best,
                    file_name="polimer_data.csv",
                    mime="text/csv"
                )
            
            st.divider()
            
            # --- PDF RAPOR OLUŞTURMA KISMI (Yeni Yeri) ---
            st.subheader("📄 Kapsamlı PDF Raporu")
            st.info("Bu rapor; tüm tahminleri, molekül yapısını, AI yorumlarını ve varsa sentez planını içerir.")

            # Verileri Session State'ten Topla
            
            # 1. AI Yorumu (Tab 5'ten)
            gen_ai_analysis = st.session_state.get('ai_analysis', "Genel AI analizi yapilmadi.")
            
            # 2. Retrosentez Bilgisi (Tab 6'dan)
            # Eğer kullanıcı Tab 6'ya hiç gitmediyse, bu veriler eksik olabilir.
            manual_retro = st.session_state.get('retro_manual_text', "Otomatik ayristirma verisi yok (Retrosentez sekmesini ziyaret edin).")
            ai_retro = st.session_state.get('ai_retro_text', "AI sentez recetesi olusturulmadi.")
            
            full_retro_info = manual_retro + "\n\n--- AI Sentez Notlari ---\n" + ai_retro

            if st.button("🚀 PDF Raporu Oluştur", type="primary", use_container_width=True):
                with st.spinner("Rapor derleniyor..."):
                    pdf_data = create_pdf_report(
                        best_poly_data, 
                        saved_targets, 
                        saved_active_props, 
                        gen_ai_analysis, 
                        full_retro_info
                    )
                    
                    st.success("Rapor hazır!")
                    st.download_button(
                        label="📥 PDF Dosyasını İndir",
                        data=pdf_data,
                        file_name="PolimerX_Final_Raporu.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        with tab5:
            st.subheader("🧠 Yapay Zeka Uzman Görüşü")
            
            if not api_key:
                st.info("💡 Bu polimer hakkında detaylı kimyasal yorum almak için sol menüden **Google Gemini API Key** girmelisiniz.")
                st.markdown("[👉 Ücretsiz API Key Almak İçin Tıkla](https://aistudio.google.com/app/apikey)")
            else:
                # Butonla tetikleyelim ki her sayfa yenilemede kredi harcamasın
                if st.button("✨ Polimeri Analiz Et", type="primary"):
                    analysis_result = get_ai_interpretation(
                        api_key, 
                        best_poly_data['smiles'], 
                        best_poly_data['preds'], 
                        saved_targets, 
                        saved_active_props
                    )
                    st.markdown(analysis_result)
                    
                    # Analizi de kaydetmek isterseniz session state'e atabilirsiniz
                    st.session_state['ai_analysis'] = analysis_result
                
                # Eğer daha önce analiz yapıldıysa hafızadan göster
                elif 'ai_analysis' in st.session_state:
                    st.markdown(st.session_state['ai_analysis'])
        # --- TAB 6: RETROSENTEZ ve RAPORLAMA ---
        # --- TAB 6: RETROSENTEZ (Sadece Analiz) ---
        with tab6:
            st.header("🧪 Retrosentez Analizi")
            
            target_smiles = best_poly_data['smiles']
            
            # --- 1. OTOMATİK AYRIŞTIRMA ---
            st.subheader("1. Yapısal Ayrıştırma")
            retro_results = decompose_polymer(target_smiles)
            
            monomer_info_text = "Otomatik analiz yapilmadi." # Varsayılan
            
            if retro_results:
                best_route = retro_results[0]
                
                # Metni oluştur
                monomer_info_text = f"Yontem: {best_route['type']}\nMekanizma: {best_route['mechanism']}\n"
                
                st.info(f"**Algılanan Sentez Türü:** {best_route['type']}")
                st.write(f"**Mekanizma:** {best_route['mechanism']}")
                
                st.markdown("**Olası Başlangıç Monomerleri:**")
                img_retro = draw_retrosynthesis_grid(best_route['monomers'])
                if img_retro: st.image(img_retro)
                
                # Ticari Kontrol
                st.markdown("#### 🛒 Ticari Bulunabilirlik")
                found_monomers = []
                for i, m in enumerate(best_route['monomers']):
                    col_code, col_check = st.columns([3, 1])
                    with col_code:
                        st.code(f"Monomer {i+1}: {m}")
                    with col_check:
                        if st.button(f"🔍 Kontrol #{i+1}", key=f"chk_{i}"):
                            is_avail, cid, name = check_commercial_availability(m)
                            if is_avail:
                                st.success(f"Var: {name}")
                                found_monomers.append(name)
                            else:
                                st.error("Ticari kayit yok")
                
                if found_monomers:
                    monomer_info_text += f"\nTicari Kaydi Olanlar: {', '.join(found_monomers)}"
                else:
                    monomer_info_text += "\nTicari kayit kontrolu yapilmadi veya bulunamadi."
            
            else:
                st.warning("Yapısal ayrıştırma başarısız.")
                monomer_info_text = "Yapısal ayrıştırma başarısız."

            # --- VERİYİ KAYDET (Tab 4 için) ---
            st.session_state['retro_manual_text'] = monomer_info_text

            st.divider()

            # --- 2. AI SENTEZ PLANI ---
            st.subheader("2. AI Sentez Reçetesi")
            
            if api_key and st.button("⚗️ Sentez Rotasını Oluştur (AI)", type="primary"):
                ai_retro_text = get_ai_retrosynthesis_guide(api_key, target_smiles, str(retro_results))
                st.markdown(ai_retro_text)
                st.session_state['ai_retro_text'] = ai_retro_text
            
            elif 'ai_retro_text' in st.session_state:
                st.markdown(st.session_state['ai_retro_text'])
            # --- MEVCUT TAB 6 KODUNUN DEVAMI ---
            
            st.divider()

            # --- 2. YEREL MODEL TAHMİNİ (GEMINI YERİNE) ---
            st.subheader("2. T5-Model Tahmini (Machine Learning)")
            st.caption("Eğittiğimiz model, moleküler yapıyı analiz ederek monomerleri tahmin ediyor.")

            if st.button("🧠 Monomerleri Tahmin Et", type="primary"):
                with st.spinner("Yapay zeka düşünüyor..."):
                    # Tahmin Fonksiyonunu Çağır
                    prediction = predict_monomers_local(best_poly_data['smiles'])
                    
                    # Sonucu Göster
                    st.success("Tahmin Başarılı!")
                    
                    st.markdown(f"""
                    <div style="background-color:#e8f5e9; padding:15px; border-radius:10px; border:1px solid #4CAF50;">
                        <h4 style="color:#2e7d32; margin:0;">🧪 Önerilen Monomerler:</h4>
                        <code style="font-size:1.1em; color:#1b5e20; background-color:#e8f5e9;">{prediction}</code>
                    </div>
                    """, unsafe_allow_html=True)
                    print("Predicted monomers:", prediction)
                    # Görselleştirme
                    monomers_list = prediction.split(' . ') # Veri setinde " . " ile ayırmıştık
                    img_retro = draw_retrosynthesis_grid(monomers_list)
                    if img_retro:
                        st.image(img_retro, caption="Modelin Önerdiği Yapı Taşları")
                    
                    # Session State'e kaydet (PDF raporu için)

                    st.session_state['retro_manual_text'] = f"AI Tahmini: {prediction}"




