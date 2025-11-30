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

# Kimya kütüphaneleri
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies as sf
from datasets import load_dataset
# import stmol as showmol # 3D görselleştirme kütüphanesi (varsa)

RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Draw

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
    return valid_selfies
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

# --- YARDIMCI KİMYA FONKSİYONLARI (Değişmedi) --

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
    
    # 1. Önbellek Kontrolü (Hızlandırıcı)
    if s_selfies in FITNESS_CACHE:
        return FITNESS_CACHE[s_selfies]

    # --- Standart Hesaplama Başlar ---
    s_smiles = selfies_to_smiles_safe(s_selfies)
    if s_smiles is None:
        return (1000.0,)

    fp = get_morgan_fp(s_smiles)
    if fp is None:
        return (1000.0,)

    preds = {}
    for prop in active_props:
        if prop in models:
            preds[prop] = models[prop].predict(fp)[0]
    
    total_error = 0.0
    if not active_props:
        return (1000.0,)

    for prop in active_props:
        if prop in preds:
            # Ranges sözlüğünü dışarıdan parametre olarak alıyoruz
            norm_error = abs(preds[prop] - targets[prop]) / (ranges[prop]['max'] - ranges[prop]['min'])
            total_error += np.exp(norm_error * 10) - 1
    
    if total_error == 0.0 and len(active_props) > 0:
         return (1000.0,)
    
    # SA Score da pahalıdır, hesaplamaya dahil edelim
    sa_score = get_sa_score_local(s_smiles)
    total_error += sa_score / 10.0
    
    result = (total_error,)
    
    # 2. Sonucu Önbelleğe Yaz
    FITNESS_CACHE[s_selfies] = result
    
    return result

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

def generate_offspring(individual, initial_selfies, mutpb=0.3, extendpb=0.2, newpb=0.1, chempb=0.3):
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
    
    # Optimize edilmiş evaluate fonksiyonunu kullanıyoruz
    toolbox.register("evaluate", evaluate_individual_optimized, models=models, targets=targets, active_props=active_props, ranges=ranges_dict)
    
    toolbox.register("mate", cxSelfies)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop_size = 100 
    pop = toolbox.population(n=pop_size)
    best_history = []

    # İlk değerlendirme
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # --- UI Elementleri ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Logları göstermek için bir alan açıyoruz
    log_expander = st.expander("📝 GA Logları (Canlı Akış)", expanded=True)
    with log_expander:
        log_placeholder = st.empty() # Tabloyu buraya basacağız
        mutation_placeholder = st.empty() # Mutasyon istatistikleri buraya

    log_data = [] # Verileri burada biriktireceğiz

    # --- ANA DÖNGÜ ---
    for gen in range(generations):
        # Adaptif oranlar
        scale = gen / generations
        cxpb = 0.7 - (0.2 * scale)
        mutpb = 0.3 - (0.2 * scale)
        extendpb = 0.2 - (0.15 * scale)
        newpb = 0.1 - (0.05 * scale)
        chempb = 0.3 - (0.15 * scale)

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
            if not offspring[i].fitness.valid: # Sadece değişenlere uygula (Hız için önemli)
                 pass
            # İstatistik için global mutation_stats sözlüğünü kullanıyor fonksiyonunuz
            offspring[i] = generate_offspring(offspring[i], initial_pop, mutpb=mutpb, extendpb=extendpb, newpb=newpb, chempb=chempb)
            del offspring[i].fitness.values

        # Değerlendirme (Sadece fitness'ı geçersiz olanlar)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = offspring
        
        # --- İstatistik Toplama ---
        fits = [ind.fitness.values[0] for ind in pop]
        best_val = min(fits)
        mean_val = sum(fits) / len(pop)
        std_val = np.std(fits)
        
        best_history.append(best_val)
        
        # Log verisini kaydet
        log_data.append({
            "Nesil": gen + 1,
            "En İyi Hata": round(best_val, 4),
            "Ortalama Hata": round(mean_val, 4),
            "Std Sapma": round(std_val, 4)
        })

        # --- UI Güncelleme (Her 5 nesilde bir veya son nesilde) ---
        # Her nesilde güncellemek yavaşlatır, 5'te bir yapmak en iyisidir.
        # --- UI Güncelleme (Her 5 nesilde bir veya son nesilde) ---
        if gen % 5 == 0 or gen == generations - 1:
            progress_bar.progress((gen + 1) / generations)
            status_text.text(f"Nesil {gen+1}/{generations} | En İyi Hata: {best_val:.2f}")
            # DataFrame olarak tabloyu güncelle
            df_log = pd.DataFrame(log_data)
            
            # DÜZELTME BURADA: use_container_width yerine width="stretch"
            log_placeholder.dataframe(
                df_log.sort_values(by="Nesil", ascending=False).head(10), 
                width="stretch"
            )
            
            # Mutasyon istatistiklerini yazdır
            mutation_placeholder.json(mutation_stats)

    # Sonuç
    best_ind = tools.selBest(pop, 1)[0]
    best_smiles = selfies_to_smiles_safe(best_ind[0])
    
    if best_smiles:
        fp = get_morgan_fp(best_smiles)
        preds = {prop: models[prop].predict(fp)[0] for prop in models.keys()}
        return {'smiles': best_smiles, 'preds': preds, 'total_error': best_ind.fitness.values[0]}, best_history
    else:
        return None, best_history

import requests

@st.cache_data # Sorguları önbelleğe al ki hızlanasın
def check_pubchem_availability(smiles: str):
    """
    Verilen SMILES için PubChem'de kayıtlı mı kontrol eder.
    Kayıtlıysa (True, CID, isim) döner, kayıtlı değilse (False, None, None).
    """
    # PubChem API URL
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if "IdentifierList" in data and "CID" in data["IdentifierList"]:
            cid = data["IdentifierList"]["CID"][0]  # İlk CID'i alıyoruz
            # CID üzerinden isim sorgulama
            name_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON"
            name_resp = requests.get(name_url, timeout=5)
            name_resp.raise_for_status()
            name_data = name_resp.json()
            name = name_data["PropertyTable"]["Properties"][0]["IUPACName"]
            return True, cid, name
        else:
            return False, None, None
    except Exception as e:
        print("PubChem sorgusunda hata:", e)
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
        'Refractive': {'min': 1.2, 'max': 2.0, 'default': 1.5, 'step': 0.01, 'is_int': False}
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
    generations = st.sidebar.slider("Evrim Nesli Sayısı", 10, 100, 10)

    # Başlangıç popülasyonu (Gerçek verinizi buraya koyun)
    initial_selfies = get_initial_population()
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Genel Bakış", "🧬 Yapısal Analiz", "📈 Evrim Geçmişi", "💾 Raporlama", "🤖 AI Analizi"])

        # --- TAB 1: ÖZET ---
        with tab1:
            col_main, col_score = st.columns([3, 1])
            with col_main:
                st.markdown(f"### 🏆 En İyi Adayın Toplam Hatası: **{best_poly_data['total_error']:.4f}**")
            with col_score:
                sa = get_sa_score_local(best_poly_data['smiles'])
                st.metric("Sentezlenebilirlik (SA)", f"{sa:.2f}", help="1 (Kolay) - 10 (Zor)")

            st.divider()
            
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

        # --- TAB 3: GRAFİK ---
        with tab3:
            st.subheader("Genetik Algoritma Performansı")
            st.line_chart(history, width="stretch") # Düzeltilmiş width

        # --- TAB 4: İNDİRME ---
        with tab4:
            st.subheader("📥 Sonuçları Dışa Aktar")
            st.markdown("Analiz sonuçlarını Excel veya Python'da kullanmak için CSV formatında indirebilirsiniz.")
            
            c1, c2 = st.columns(2)
            
            # CSV Hazırlama
            export_dict = {
                "SMILES": best_poly_data['smiles'],
                "Toplam Hata": best_poly_data['total_error'],
                "SA Score": get_sa_score_local(best_poly_data['smiles'])
            }
            export_dict.update(preds)
            for k, v in saved_targets.items():
                if k in saved_active_props:
                    export_dict[f"Hedef_{k}"] = v
            
            df_best = pd.DataFrame([export_dict])
            csv_best = df_best.to_csv(index=False).encode('utf-8')

            with c1:
                st.download_button(
                    label="🧪 En İyi Adayı İndir (.csv)",
                    data=csv_best,
                    file_name="polimer_sonuc.csv",
                    mime="text/csv",
                    type="primary"
                )

            # Geçmiş Verisi
            df_hist = pd.DataFrame(history, columns=["En_Iyi_Hata"])
            df_hist["Nesil"] = df_hist.index + 1
            csv_hist = df_hist.to_csv(index=False).encode('utf-8')

            with c2:
                st.download_button(
                    label="📈 Evrim Geçmişini İndir (.csv)",
                    data=csv_hist,
                    file_name="optimizasyon_gecmisi.csv",
                    mime="text/csv"
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