import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="ScholarMind",
    layout="centered"
)

# --- 2. CSS TASARIMI (MODERN VE TEMIZ) ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    
    /* Kart Tasarımı (Ana Sayfa) */
    .paper-card {
        background-color: white; 
        padding: 20px; 
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); 
        margin-bottom: 20px;
        border-left: 5px solid #3498db;
        transition: transform 0.2s;
    }
    .paper-card:hover { transform: translateX(5px); border-color: #2ecc71; }
    
    /* Detay Sayfası Tasarımı */
    .detail-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
        font-weight: 800;
        font-size: 2rem;
        margin-bottom: 10px;
    }
    .meta-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .meta-label { font-weight: bold; color: #555; margin-right: 10px; min-width: 80px; }
    .meta-value { color: #2c3e50; font-weight: 600; }
    
    /* Özet Kutusu */
    .abstract-box {
        background-color: #fff;
        padding: 25px;
        border-radius: 10px;
        border-left: 4px solid #f1c40f;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        font-size: 1rem;
        line-height: 1.6;
        color: #444;
        margin-top: 20px;
    }
    
    /* Etiketler */
    .badge {
        background-color: #e8f4fd; color: #3498db; 
        padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;
    }
    
    /* Sanal Kapak */
    .cover-image {
        width: 100%;
        height: 250px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. VERI YUKLEME ---
@st.cache_resource
def load_data_safe():
    files = ["balanced_df.pkl", "embeddings.npy", "users.pkl"]
    if any(not os.path.exists(f) for f in files):
        st.error("Veri dosyalari eksik! Lutfen pkl/npy dosyalarini klasore atin.")
        st.stop()
    
    try:
        df = pd.read_pickle("balanced_df.pkl")
        embeddings = np.load("embeddings.npy")
        with open("users.pkl", "rb") as f: users = pickle.load(f)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return df, embeddings, users, model
    except Exception as e:
        st.error(f"Hata: {e}")
        st.stop()

df, embeddings, users, model = load_data_safe()

# --- 4. SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'selected_paper' not in st.session_state: st.session_state.selected_paper = None
if 'saved_papers' not in st.session_state: st.session_state.saved_papers = set()

# --- 5. ALGORITMALAR ---
def search_papers(query, top_n=20):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = scores.argsort()[::-1][:top_n]
    return df.iloc[top_indices]

def hybrid_recommendation(paper_idx, top_n=3):
    target_emb = embeddings[paper_idx].reshape(1, -1)
    sim_scores = cosine_similarity(target_emb, embeddings)[0]
    candidate_indices = sim_scores.argsort()[::-1][1:101]
    
    recs = []
    for idx in candidate_indices:
        score_content = sim_scores[idx]
        score_pop = df.iloc[idx]["normalized_popularity"]
        final = (score_content * 0.7) + (score_pop * 0.3)
        recs.append((idx, final))
        
    recs = sorted(recs, key=lambda x: x[1], reverse=True)[:top_n]
    return df.iloc[[x[0] for x in recs]]

# --- 6. NAVIGASYON ---
def go_to_detail(row):
    st.session_state.selected_paper = row
    st.session_state.page = 'detail'

def go_home():
    st.session_state.page = 'home'

def toggle_save(pid):
    if pid in st.session_state.saved_papers:
        st.session_state.saved_papers.remove(pid)
        st.toast("Kutuphaneden cikarildi")
    else:
        st.session_state.saved_papers.add(pid)
        st.toast("Kutuphaneye eklendi")

# --- 7. ARAYUZ ---

# === SAYFA 1: ANA ARAMA EKRANI ===
if st.session_state.page == 'home':
    st.markdown("<h1 style='text-align: center;'>ScholarMind</h1>", unsafe_allow_html=True)
    
    # Kutuphane Bilgisi
    if st.session_state.saved_papers:
        st.info(f"Kutuphanenizde {len(st.session_state.saved_papers)} makale var.")
    
    # Arama
    query = st.text_input("", placeholder="Ne araştırmak istersiniz? (Ornek: Neural Networks)", key="search")
    
    if query:
        st.subheader(f"Sonuclar: '{query}'")
        results = search_papers(query, top_n=15)
    else:
        st.subheader("Haftanin Populer Makaleleri")
        results = df.sort_values("normalized_popularity", ascending=False).head(50).sample(15)

    # LISTE (Alt Alta Kartlar)
    for idx, (index, row) in enumerate(results.iterrows()):
        st.markdown(f"""
        <div class="paper-card">
            <h3 style="margin:0; color:#2c3e50;">{row['title']}</h3>
            <div style="margin:5px 0;">
                <span class="badge">{row['main_category']}</span>
                <span style="color:#888; font-size:0.9rem;">Yil: {row['year']}</span>
            </div>
            <div style="color:#555; font-size:0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {row['abstract']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tam Genislik Buton
        if st.button("Detayli Incele", key=f"btn_{row['id']}"):
            go_to_detail(row)
            st.rerun()

# === SAYFA 2: DETAY EKRANI (DOLU DOLU GORUNUM) ===
elif st.session_state.page == 'detail':
    p = st.session_state.selected_paper
    
    # Geri Tusu
    if st.button("Listeye Don"):
        go_home()
        st.rerun()
        
    # --- 1. UST KISIM (KAPAK + BASLIK) ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_img, col_info = st.columns([1, 2])
    
    with col_img:
        # Sanal Kapak
        st.markdown(f"""
        <div class="cover-image">
            <span style="font-size:1rem; opacity:0.8;">{p['id']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Kaydet Butonu (Gorselin Altina)
        is_saved = p['id'] in st.session_state.saved_papers
        btn_txt = "Kutuphanede" if is_saved else "Kutuphaneye Ekle"
        btn_type = "primary" if not is_saved else "secondary"
        
        if st.button(btn_txt, type=btn_type, use_container_width=True):
            toggle_save(p['id'])
            st.rerun()

    with col_info:
        # Baslik ve Ozellikler Listesi (Alt Alta)
        st.markdown(f"<div class='detail-header'>{p['title']}</div>", unsafe_allow_html=True)
        
        # Liste Halinde Ozellikler
        properties = [
            ("Yazarlar", p['authors']),
            ("Yil", str(p['year'])),
            ("Kategori", p['main_category']),
            ("Popularite", f"{p['normalized_popularity']:.2f} / 1.0"),
            ("Kaynak", f"<a href='{p['link']}' target='_blank'>ArXiv Linki</a>")
        ]
        
        for label, value in properties:
            st.markdown(f"""
            <div class="meta-box">
                <span class="meta-label">{label}:</span>
                <span class="meta-value">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    # --- 2. ALT KISIM (OZET KUTUSU) ---
    st.markdown(f"""
    <div class="abstract-box">
        <h3 style="margin-top:0; color:#f39c12;">Ozet (Abstract)</h3>
        {p['abstract']}
    </div>
    """, unsafe_allow_html=True)
    
    # --- 3. BENZERLER ---
    st.markdown("---")
    st.subheader("Buna Bakanlar Sunlari da Inceledi")
    
    paper_idx = df[df['id'] == p['id']].index[0]
    recs = hybrid_recommendation(paper_idx, top_n=3)
    
    cols = st.columns(3)
    for i, (idx, row) in enumerate(recs.iterrows()):
        with cols[i]:
            st.info(f"**{row['title'][:40]}...**")
            if st.button("Goz At", key=f"rec_{row['id']}", use_container_width=True):
                go_to_detail(row)
                st.rerun()