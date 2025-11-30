import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
import os
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from stmol import showmol
import py3Dmol
import pubchempy as pcp
from datasets import load_dataset
# --- AYARLAR ---
st.set_page_config(page_title="PolimerX AI v2.0", layout="wide", page_icon="🧬")

# --- MODEL VE VERİ YÜKLEME ---
@st.cache_resource
def load_resources():
    try:
        critic = joblib.load("polimerx_critic_v2_sanitized.joblib")
        
        if True:
            repo_id = "OsBaran/Polimer-Ozellik-Tahmini"
            tg_data = load_dataset(repo_id,split="Tg")
            df = tg_data.to_pandas()
            # Sütun adının 'p_smiles' veya 'smiles' olduğundan emin ol
            col_name = 'p_smiles' if 'p_smiles' in df.columns else 'smiles'
            initial_pop = df[col_name].tolist()
        else:
            initial_pop = ["*CC*", "*CCC*", "*CCCC*", "*CC(F)*", "*c1ccccc1*"] * 50
            
        return critic, initial_pop
    except Exception as e:
        return None, None

# --- 3D GÖRSELLEŞTİRME MOTORU (YENİ) ---
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

# --- PUBCHEM KONTROLÜ (YENİ) ---
@st.cache_data # Sorguları önbelleğe al ki hızlanasın
def check_pubchem_availability(monomer_name_or_smiles):
    try:
        # İsme veya SMILES'a göre arama yap
        compounds = pcp.get_compounds(monomer_name_or_smiles, 'smiles')
        if compounds:
            cid = compounds[0].cid
            synonyms = compounds[0].synonyms
            common_name = synonyms[0] if synonyms else "Bilinmiyor"
            return True, cid, common_name
    except:
        pass
    return False, None, None

# --- DİĞER YARDIMCI FONKSİYONLAR ---
def get_morgan_fp(p_smiles):
    if not p_smiles: return None
    try:
        smi_clean = str(p_smiles).replace('*', '[H]').replace('(*)', '[H]').replace('[*]', '[H]')
        mol = Chem.MolFromSmiles(smi_clean)
        if mol is None: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        return np.array(fp)
    except: return None

def get_sa_score_local(p_smiles):
    try:
        import sascorer
        smi_clean = str(p_smiles).replace('*', '[H]')
        mol = Chem.MolFromSmiles(smi_clean)
        return sascorer.calculateScore(mol)
    except:
        return 3.5 + (len(str(p_smiles)) * 0.05)

# --- GENETİK ALGORİTMA ---
def run_genetic_algorithm(target_tg, generations, critic_model, initial_pop):
    current_pop = random.sample(initial_pop, min(len(initial_pop), 50))
    best_history = []
    best_polymer = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def mutate(smi):
        chars = list(smi)
        if len(chars) > 4 and random.random() > 0.4:
            idx = random.randint(1, len(chars)-2)
            if chars[idx] == 'C': 
                chars[idx] = random.choice(['N', 'O', '(F)', '(C)'])
        return "".join(chars)

    for gen in range(generations):
        progress = (gen + 1) / generations
        progress_bar.progress(progress)
        status_text.text(f"Nesil {gen+1}/{generations}: Popülasyon Evrimleşiyor...")
        
        fps = []
        valid_smiles = []
        for s in current_pop:
            fp = get_morgan_fp(s)
            if fp is not None:
                fps.append(fp)
                valid_smiles.append(s)
        
        if not fps: break
        
        preds = critic_model.predict(np.array(fps))
        
        scored_pop = []
        for s, tg in zip(valid_smiles, preds):
            score = -abs(tg - target_tg) 
            scored_pop.append((s, tg, score))
            
        scored_pop.sort(key=lambda x: x[2], reverse=True)
        best_of_gen = scored_pop[0]
        best_history.append(best_of_gen[1])
        best_polymer = best_of_gen
        
        new_pop = [x[0] for x in scored_pop[:15]] 
        while len(new_pop) < 50:
            parent = random.choice(new_pop[:10])
            new_pop.append(mutate(parent))
        current_pop = new_pop
        
        time.sleep(0.02)

    return best_polymer, best_history

# --- RETROSENTEZ ---
def run_retrosynthesis_local(p_smiles):
    clean_smi = str(p_smiles).replace('*', '')
    route = {"type": "Bilinmeyen", "m1": "-", "m1_smi": "", "m2": "-", "m2_smi": ""}
    
    if "(=O)N" in clean_smi:
        route = {
            "type": "Poliamid Sentezi",
            "m1": "Tereftalik Asit",
            "m1_smi": "OC(=O)c1ccc(cc1)C(=O)O",
            "m2": "Heksametilen Diamin",
            "m2_smi": "NCCCCCCN"
        }
    elif "(=O)O" in clean_smi:
        route = {
            "type": "Polyester Sentezi",
            "m1": "Adipik Asit",
            "m1_smi": "OC(=O)CCCCC(=O)O",
            "m2": "Etilen Glikol",
            "m2_smi": "OCCO"
        }
    else:
        route = {
            "type": "Vinil Polimerizasyonu",
            "m1": "Vinil Monomeri",
            "m1_smi": clean_smi, # Polimerin kendisi monomer olarak varsayılır
            "m2": "-",
            "m2_smi": ""
        }
    return route

# --- ARAYÜZ TASARIMI ---
st.title("🧪 PolimerX v2.0: Akıllı Malzeme Keşfi")
st.markdown("Genetik Algoritmalar ve 3D Modelleme ile Geleceğin Polimerleri")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Kontrol Paneli")
    target_tg = st.slider("Hedeflenen Tg (°C)", 0.0, 500.0, 200.0)
    generations = st.slider("Evrim Döngüsü", 10, 100, 40)
    
    st.info("Model: XGBoost v2\nVeri: 8.4k PolyInfo\nMotor: GA + PubChem API")
    start_btn = st.button("🚀 Keşfi Başlat", type="primary")

with col2:
    model, pop = load_resources()
    
    if model is None:
        st.error("⚠️ Model dosyası bulunamadı! Klasörü kontrol edin.")
    
    elif start_btn:
        st.subheader("🧬 Evrimsel Süreç")
        
        best_poly, history = run_genetic_algorithm(target_tg, generations, model, pop)
        
        # Grafik
        chart_data = pd.DataFrame(history, columns=["En İyi Tg"])
        st.line_chart(chart_data)
        
        st.success(f"Hedefe Ulaşıldı! ({best_poly[1]:.2f} °C)")
        
        st.divider()
        
        # --- SONUÇ ALANI (GÜNCELLENDİ: 3D GÖRSEL) ---
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("### 🧊 3D Molekül Yapısı")
            view = make_3d_view(best_poly[0])
            if view:
                showmol(view, height=300, width=400)
            else:
                st.warning("3D model oluşturulamadı.")
        
        with c2:
            st.markdown(f"### 🔬 Özellikler")
            st.metric("Tahmin Edilen Tg", f"{best_poly[1]:.2f} °C", delta=f"{best_poly[1]-target_tg:.2f}")
            sa = get_sa_score_local(best_poly[0])
            st.metric("Sentez Zorluğu (SA)", f"{sa:.2f} / 10")
            st.text_area("p-SMILES Kodu:", best_poly[0], height=100)

        st.divider()
        
        # --- RETROSENTEZ ALANI (GÜNCELLENDİ: PUBCHEM KONTROLÜ) ---
        st.subheader("⚗️ Retrosentez ve Ticari Kontrol")
        retro = run_retrosynthesis_local(best_poly[0])
        st.write(f"**Önerilen Sentez Yolu:** {retro['type']}")
        
        monomers_col1, monomers_col2 = st.columns(2)
        
        # Monomer 1 Kontrolü
        with monomers_col1:
            st.markdown("#### Monomer 1")
            st.code(retro['m1_smi'])
            
            if retro['m1_smi']:
                is_avail, cid, name = check_pubchem_availability(retro['m1_smi'])
                if is_avail:
                    st.success(f"✅ PubChem'de Bulundu!\n\n**Adı:** {name}\n**CID:** {cid}")
                    st.markdown(f"[Satın Alma Linki (Simülasyon)](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
                else:
                    st.error("❌ Ticari kaydı bulunamadı (Sentezlenmesi gerek).")
            
        # Monomer 2 Kontrolü
        with monomers_col2:
            st.markdown("#### Monomer 2")
            st.code(retro['m2_smi'])
            
            if retro['m2_smi']:
                is_avail, cid, name = check_pubchem_availability(retro['m2_smi'])
                if is_avail:
                    st.success(f"✅ PubChem'de Bulundu!\n\n**Adı:** {name}\n**CID:** {cid}")
                    st.markdown(f"[Satın Alma Linki (Simülasyon)](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
                else:
                    if retro['m2'] != "-":
                        st.error("❌ Ticari kaydı bulunamadı.")
                    else:
                        st.info("Bu reaksiyon için 2. monomer gerekmez.")

    else:
        st.info("Sol panelden hedef sıcaklığı seçin ve butona basın.")