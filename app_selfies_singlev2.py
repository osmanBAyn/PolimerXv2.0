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

# import numpy as np
# from rdkit.Chem import QED
# sascorer genellikle harici bir scripttir, projenizde yoksa QED ile ilerleyebilirsiniz
# from rdkit.Chem import RDConfig
# import os, sys
# sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# import sascorer

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
def run_single_objective_flow(models, generations, targets, active_props, initial_pop):
    toolbox = base.Toolbox()
    toolbox.register("attr_selfies", random.choice, initial_pop)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_selfies, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual_single_obj, models=models, targets=targets, active_props=active_props)
    toolbox.register("mate", cxSelfies)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop_size = 100
    # Başlangıç parametreleri
    cxpb_start, cxpb_end = 0.7, 0.5
    mutpb_start, mutpb_end = 0.3, 0.1
    extendpb_start, extendpb_end = 0.2, 0.05
    newpb_start, newpb_end = 0.1, 0.05
    chempb_start, chempb_end = 0.3, 0.15

    pop = toolbox.population(n=pop_size)
    best_history = []

    # Başlangıç popülasyonunu değerlendir
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    progress_bar = st.progress(0)
    status_text = st.empty()

    log_panel = st.expander("📝 GA Logları", expanded=False)

    def adaptive_prob(initial, final, gen, max_gen):
        return initial + (final - initial) * (gen / max_gen)

    for gen in range(generations):
        # Adaptif parametreler
        cxpb = adaptive_prob(cxpb_start, cxpb_end, gen, generations)
        mutpb = adaptive_prob(mutpb_start, mutpb_end, gen, generations)
        extendpb = adaptive_prob(extendpb_start, extendpb_end, gen, generations)
        newpb = adaptive_prob(newpb_start, newpb_end, gen, generations)
        chempb = adaptive_prob(chempb_start, chempb_end, gen, generations)

        offspring = toolbox.select(pop, pop_size)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Çaprazlama
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(ind1, ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Mutasyon + zincir uzatma + yeni birey (adaptif)
        for i, ind in enumerate(offspring):
            offspring[i] = generate_offspring(ind, initial_pop, mutpb=mutpb, extendpb=extendpb, newpb=newpb, chempb=chempb)
            del offspring[i].fitness.values

        # Değerlendir
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        
        pop = offspring
        best_of_gen = tools.selBest(pop, 1)[0]
        best_history.append(best_of_gen.fitness.values[0])

        progress_bar.progress((gen + 1) / generations)
        status_text.text(f"Nesil {gen+1}/{generations}: En İyi Toplam Hata: {best_of_gen.fitness.values[0]:.2f}")

        # Log paneline yazdır
        with log_panel:
            fitness_vals = [ind.fitness.values[0] for ind in pop]
            mean_error = np.mean(fitness_vals)
            std_error = np.std(fitness_vals)
            st.write(f"Nesil {gen+1}: Ortalama Hata={mean_error:.2f}, Std={std_error:.2f}")
            st.write(f"Mutasyon dağılımı: {mutation_stats}")
        
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

st.title("🎯 PolimerX - Dinamik Optimizasyon")

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

            # Grafik
            chart_data = pd.DataFrame(history, columns=["Toplam Sapma"])
            st.line_chart(chart_data)
            
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


            st.divider()

            col_model, col_sa = st.columns([1, 1])

            # ==== SOL KUTU: 3D MODEL veya NEDEN OLUŞMADI ====
            with col_model:
                st.markdown("### 🧬 Molekül 3D Görünümü")

                view, reason = make_3d_view_with_reason(best_poly_data["smiles"])

                box_w, box_h = 380, 380  # aynı boyut için

                if view:
                    showmol(view, height=box_h, width=box_w)

                else:
                    # kare kutu
                    st.markdown(
                        f"""
                        <div style="
                            width:{box_w}px;
                            height:{box_h}px;
                            border-radius:12px;
                            background-color:#1b1b1b;
                            display:flex;
                            justify-content:center;
                            align-items:center;
                            color:#e0e0e0;
                            font-size:17px;
                            text-align:center;
                            border:1px solid #444;
                            padding:10px;
                        ">
                            3D model üretilemedi<br>
                            ({reason})
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ==== SAĞ KUTU: SA SCORE + PUBCHEM ====
            with col_sa:
                st.markdown("### 🔬 Özellik Bilgileri")

                sa = get_sa_score_local(best_poly_data["smiles"])

                # SA Score'u merkeze alan düzen
                center_col_l, center_col_mid, center_col_r = st.columns([1, 2, 1])
                with center_col_mid:
                    st.metric("🧪 Sentez Zorluğu (SA)", f"{sa:.2f} / 10")

                    is_avail, cid, name = check_pubchem_availability(best_poly_data["smiles"])
                    if is_avail:
                        st.success(f"✅ PubChem'de Bulundu!\n\n**Adı:** {name}\n**CID:** {cid}")
                        st.markdown(f"[Satın Alma Linki (Simülasyon)](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
                    else:
                        st.error("❌ Ticari kaydı bulunamadı.")
        
        else:
            st.error("Belirlenen hedeflere yakın geçerli bir polimer bulunamadı. Lütfen nesil sayısını veya hedef değerleri değiştirin.")
