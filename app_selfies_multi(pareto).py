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

# NSGA-II için DEAP kütüphanesi
import deap.base as base
import deap.creator as creator
import deap.tools as tools
from deap import algorithms

# Kimya kütüphaneleri
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies as sf

RDLogger.DisableLog('rdApp.*')

# --- SABİTLER ---
CRITIC_MODEL_NAMES = ['xgb_tg.joblib', 'xgb_td.joblib', 'rf_eps.joblib']
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
        # Bu noktada diğer 5 model de yüklenebilir
        return models
    except Exception as e:
        st.error(f"⚠️ Model Yükleme Hatası! ({e}). Lütfen dosya adlarını kontrol edin.")
        return None

# --- SMILES / SELFIES DÖNÜŞÜMÜ ---
# (Önceki en sağlam versiyonlar)

def smiles_to_selfies_safe(smiles):
    if not smiles: return None
    clean_smi = smiles.replace('*', '[H]').replace('(*)', '[H]').replace('[*]', '[H]')
    try:
        selfies_string = sf.encoder(clean_smi)
        return selfies_string.replace('[H]', '[*]')
    except:
        return None

def selfies_to_smiles_safe(selfies_string):
    if not selfies_string: return None
    try:
        temp_selfies = selfies_string.replace('[*]', '[H]')
        smiles = sf.decoder(temp_selfies)
        return smiles.replace('[H]', '*')
    except:
        return None

def get_morgan_fp(p_smiles):
    """Tg, Td ve EPS tahminleri için parmak izini hazırlar."""
    smi_clean = str(p_smiles).replace('*', '[H]').replace('(*)', '[H]').replace('[*]', '[H]')
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, N_BITS)
    return np.array([fp]) # XGBoost 2D array bekler

# --- NSGA-II GENETİK OPERATÖRLERİ ---

def cxSelfies(ind1, ind2):
    """SELFIES çaprazlama (Tek nokta kesme)."""
    tokens1 = list(sf.split_selfies(ind1[0]))
    tokens2 = list(sf.split_selfies(ind2[0]))
    
    min_len = min(len(tokens1), len(tokens2))
    if min_len <= 1: return ind1, ind2
    
    k = random.randint(1, min_len - 1)
    
    ind1[0] = "".join(tokens1[:k] + tokens2[k:])
    ind2[0] = "".join(tokens2[:k] + tokens1[k:])
    
    return ind1, ind2

def mutSelfies(individual):
    """SELFIES mutasyonu (Rastgele token silme/ekleme)."""
    tokens = list(sf.split_selfies(individual[0]))
    if not tokens: return individual,
    
    # Silme
    if random.random() < 0.6 and len(tokens) > 1:
        idx = random.randint(0, len(tokens) - 1)
        del tokens[idx]
        
    # Ekleme
    if random.random() < 0.4:
        idx = random.randint(0, len(tokens))
        new_token = random.choice(['[C]', '[N]', '[O]', '[F]'])
        tokens.insert(idx, new_token)
    
    individual[0] = "".join(tokens)
    return individual,

# =========================================================================
# II. NSGA-II DEĞERLENDİRME ÇEKİRDEĞİ (EVALUATE FUNCTION)
# =========================================================================

def evaluate_individual_moo(individual, models, target_tg, sa_score_func):
    """
    Bireyi 3 hedefe göre değerlendirir: (Td, EPS, Tg_Error)
    """
    s_selfies = individual[0]
    s_smiles = selfies_to_smiles_safe(s_selfies)
    
    # Geçersiz yapı veya dönüştürme hatası varsa ağır ceza ver
    if s_smiles is None:
        # Td'yi minimize et, EPS'i maksimize et, Tg hatasını maksimize et
        return (-1000.0, 1000.0, 1000.0) 

    fp = get_morgan_fp(s_smiles)
    if fp is None:
        return (-1000.0, 1000.0, 1000.0)

    # 1. Tahminleri Al
    preds = {
        'Tg': models['Tg'].predict(fp)[0],
        'Td': models['Td'].predict(fp)[0],
        'EPS': models['EPS'].predict(fp)[0],
    }
    
    # 2. MOO Hedeflerini Hesapla
    
    # Hedef 1 (Td): Maksimize Etmek İstenen (Isı Dayanımı)
    td_score = preds['Td'] 
    
    # Hedef 2 (EPS): Minimize Etmek İstenen (Yalıtkanlık)
    eps_score = preds['EPS']
    
    # Hedef 3 (Tg): Minimize Etmek İstenen (Hata)
    tg_error = abs(preds['Tg'] - target_tg)
    
    # NOT: SA Score, burada ek bir dördüncü kısıt olarak ele alınabilir.
    # sa_score_val = sa_score_func(s_smiles)
    
    # Fitness tuple'ını döndür. NSGA-II bu tuple'ı optimize etmeye çalışır.
    # Sıra: (Td, EPS, Tg_Error)
    return (td_score, eps_score, tg_error)

# =========================================================================
# III. STREAMLIT ARAYÜZ VE NSGA-II AKIŞI
# =========================================================================

# DEAP Yapısını Tanımlama (Streamlit'in tekrar çalıştırma hatasını önlemek için)
if "FitnessMulti" not in creator.__dict__:
    # HEDEFLER: (Maximize Td, Minimize EPS, Minimize Tg_Error)
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def run_nsga2_flow(models, generations, target_tg, initial_pop):
    
    # 1. Toolbox'ı Tanımlama
    toolbox = base.Toolbox()
    toolbox.register("attr_selfies", random.choice, initial_pop)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_selfies, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operatörleri Kaydetme
    toolbox.register("evaluate", evaluate_individual_moo, models=models, target_tg=target_tg, sa_score_func=lambda x: 1.0) # SA Score basit tutuldu
    toolbox.register("mate", cxSelfies)
    toolbox.register("mutate", mutSelfies)
    toolbox.register("select", tools.selNSGA2)
    
    # 2. NSGA-II Parametreleri
    pop_size = 100
    mu = 100
    lambda_ = 100
    cxpb = 0.7  # Çaprazlama olasılığı
    mutpb = 0.3 # Mutasyon olasılığı
    
    pop = toolbox.population(n=pop_size)
    
    # 3. Akışı Çalıştırma
    # NSGA-II algoritması
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu, lambda_, cxpb, mutpb, 
                                          generations, stats=None, halloffame=None, verbose=True)
    
    # 4. Sonuçları Toplama (Pareto Cephesini döndür)
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    return pareto_front

# --- STREAMLIT ARAYÜZ (BASİTLEŞTİRİLMİŞ VERSİYON) ---

st.title("🧪 PolimerX MOO (Çoklu-Hedefli Optimizasyon)")

models = load_critic_models()

if models:
    st.sidebar.header("⚙️ Hedefler")
    target_tg = st.sidebar.slider("Hedef Tg (°C)", 0.0, 400.0, 200.0)
    generations = st.sidebar.slider("Evrim Nesli Sayısı", 10, 50, 20)
    
    # Kontrol için rastgele bir başlangıç popülasyonu (Gerçek app.py'de bu sizin 8.4k verinizdir)
    initial_pop_simulated = ["C[N+](C)(C)CC(=O)[O-]", "CCCCCC(=O)NCCCCCCNCC(=O)O", "c1ccc(C)c(C)c1"] * 50
    initial_selfies_simulated = [smiles_to_selfies_safe(s) for s in initial_pop_simulated if smiles_to_selfies_safe(s)]

    if st.sidebar.button("🚀 MOO Keşfini Başlat", type="primary"):
        
        with st.spinner('NSGA-II Algoritması Çalışıyor...'):
            pareto_front = run_nsga2_flow(models, generations, target_tg, initial_selfies_simulated)

        st.success(f"Optimizasyon Tamamlandı! **{len(pareto_front)}** Adet Pareto-Optimal Polimer Bulundu.")
        
        # Sonuçları DataFrame olarak göstermek
        final_results = []
        for ind in pareto_front:
            s_smiles = selfies_to_smiles_safe(ind[0])
            if s_smiles:
                final_results.append({
                    'SMILES': s_smiles,
                    'Tg_Hedef_Hata': f"{ind.fitness.values[2]:.2f}°C",
                    'Td': f"{ind.fitness.values[0]:.2f}°C (Max)",
                    'EPS': f"{ind.fitness.values[1]:.2f} (Min)",
                })

        df_pareto = pd.DataFrame(final_results)
        st.dataframe(df_pareto)
        
        st.markdown("### 📊 Yorum")
        st.info("Bu tabloda **Td** değeri yüksek, **EPS** ve **Tg Hatası** değerleri düşük olan polimerler Pareto Cephesi üzerindeki en iyi ödünleşmeleri (trade-off) temsil eder.")