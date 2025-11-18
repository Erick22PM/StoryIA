import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os

st.set_page_config(page_title="Navegador de videos", layout="wide")
st.title("🎬 Navegador de Videos")

# =======================
# Estilos CSS para cartas
# =======================
st.markdown(
    """
    <style>
    .video-card {
        border-radius: 10px;
        padding: 8px;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
        height: 260px; /* altura fija */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .video-card-title {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 6px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .video-card-stats {
        font-size: 0.7rem;
        color: #6b7280;
        margin-top: 2px;
    }
    .video-card-btn {
        padding: 6px 12px;
        background: #1f6feb;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
    }
    .video-card-btn:hover {
        background: #1d4ed8;
    }
    /* Todas las imágenes con la MISMA altura */
    .stImage img {
        border-radius: 8px;
        height: 130px;
        width: 100%;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# Cargar DataFrame
# ======================================================
@st.cache_data
def load_data(path):
    df = pd.read_parquet(path)

    # Si los embeddings están como strings, convertirlos
    if isinstance(df["embedding_guion"].iloc[0], str):
        df["embedding_guion"] = df["embedding_guion"].apply(ast.literal_eval)
    if isinstance(df["embedding_img"].iloc[0], str):
        df["embedding_img"] = df["embedding_img"].apply(ast.literal_eval)

    # Tamaño de guion (si no existe)
    if "tamano_guion" not in df.columns:
        df["tamano_guion"] = df["transcripcion"].apply(lambda x: len(str(x).split()))

    return df

DF_PATH = "./DATA/dataframes/final_datafame.parquet"
df = load_data(DF_PATH)

# ======================================================
# SIDEBAR: búsqueda y filtros
# ======================================================
with st.sidebar:
    st.header("🔎 Búsqueda y filtros")

    # BÚSQUEDA (para usar con embeddings)
    search_query = st.text_input("Buscar por guion (texto)", placeholder="Escribe algo del guion...")

    # Filtro por canal
    canales = ["Todos"] + sorted(df["canal"].dropna().astype(str).unique().tolist())
    canal_sel = st.selectbox("Canal", canales)

    # Rangos para likes, duración y tamaño de guion
    likes_min, likes_max = int(df["likes"].min()), int(df["likes"].max())
    likes_range = st.slider("Rango de likes", likes_min, likes_max, (likes_min, likes_max))

    dur_min, dur_max = float(df["duracion_seg"].min()), float(df["duracion_seg"].max())
    dur_range = st.slider("Duración (segundos)", dur_min, dur_max, (dur_min, dur_max))

    tg_min, tg_max = int(df["tamano_guion"].min()), int(df["tamano_guion"].max())
    tg_range = st.slider("Tamaño del guion (palabras)", tg_min, tg_max, (tg_min, tg_max))

    # Orden
    orden_campo = st.selectbox(
        "Ordenar por",
        ["id", "likes", "comments", "shares", "duracion_seg", "tamano_guion", "canal"],
    )
    orden_inverso = st.checkbox("Orden descendente", True)

# ======================================================
# Aplicar filtros básicos (canal, likes, duración, tamaño guion)
# ======================================================
df_filtrado = df.copy()

if canal_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["canal"].astype(str) == canal_sel]

df_filtrado = df_filtrado[
    (df_filtrado["likes"].between(likes_range[0], likes_range[1])) &
    (df_filtrado["duracion_seg"].between(dur_range[0], dur_range[1])) &
    (df_filtrado["tamano_guion"].between(tg_range[0], tg_range[1]))
]

# ======================================================
# Búsqueda semántica por embedding de guion (placeholder)
# ======================================================

def generar_embedding_query(query, df, columna_texto="transcripcion", columna_embed="embedding_guion"):
    """
    Genera un embedding aproximado del query usando los textos existentes.
    - Busca textos que contengan palabras del query.
    - Promedia los embeddings.
    - Si no encuentra coincidencias → usa TF-IDF.
    """
    palabras = query.lower().split()

    # Filtrar textos que contengan cualquier palabra del query
    mask = df[columna_texto].str.lower().apply(
        lambda x: any(p in x for p in palabras)
    )

    df_match = df[mask]

    if len(df_match) > 0:
        # Promedio de embeddings width='stretch'
        return np.mean(np.stack(df_match[columna_embed].values), axis=0)

    # Si no encuentra nada → TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = df[columna_texto].astype(str).tolist() + [query]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # Último vector = query
    q_vec = X[-1].toarray()[0]

    # Convertimos TF-IDF a dimensión embed con proyección
    # (Mejor que un vector aleatorio, sin costo)
    embeds = np.stack(df[columna_embed].values)
    proy = embeds.T @ (q_vec @ X[:-1].toarray())

    return proy / np.linalg.norm(proy)


usar_busqueda = search_query.strip() != ""


if usar_busqueda:
    st.subheader(f"🔍 Resultados para: **{search_query}**")

    q_embedding = generar_embedding_query(search_query, df_filtrado)

    matrix = np.stack(df_filtrado["embedding_guion"].values)
    sim = cosine_similarity([q_embedding], matrix)[0]

    df_filtrado = df_filtrado.assign(similaridad=sim)
    df_filtrado = df_filtrado.sort_values("similaridad", ascending=False)

else:
    st.subheader("📁 Todos los videos")
    df_filtrado = df_filtrado.sort_values(orden_campo, ascending=orden_inverso)

# ======================================================
# Paginación
# ======================================================
ITEMS_POR_PAG = 20

if "page" not in st.session_state:
    st.session_state.page = 0

total_videos = len(df_filtrado)
total_paginas = max(1, (total_videos - 1) // ITEMS_POR_PAG + 1)

# Asegurar que la página actual esté dentro de rango
if st.session_state.page > total_paginas - 1:
    st.session_state.page = total_paginas - 1

inicio = st.session_state.page * ITEMS_POR_PAG
fin = inicio + ITEMS_POR_PAG
df_page = df_filtrado.iloc[inicio:fin]

st.caption(f"Mostrando {len(df_page)} de {total_videos} videos | Página {st.session_state.page + 1} de {total_paginas}")

col_prev, col_next = st.columns([1, 1])
with col_prev:
    if st.button("⬅️ Anterior") and st.session_state.page > 0:
        st.session_state.page -= 1
        st.rerun()
with col_next:
    if st.button("Siguiente ➡️") and st.session_state.page < total_paginas - 1:
        st.session_state.page += 1
        st.rerun()

st.markdown("---")

# ======================================================
# Mostrar galería de videos en tarjetas alineadas
# ======================================================
# 5 columnas por fila
num_cols = 5
cols = st.columns(num_cols)

for idx, (_, row) in enumerate(df_page.iterrows()):
    col = cols[idx % num_cols]
    with col:

        img = row.get("img_path", None)
        if img and os.path.exists(img):
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                st.image(img, width=180)  # Ancho controlado
        else:
            st.image("https://via.placeholder.com/400x300?text=Sin+imagen", width='stretch')


        # Título: canal + ID
        titulo = f"{row.get('canal', 'Sin canal')} · {row['id']}"
        st.markdown(f'<div class="video-card-title">{titulo}</div>', unsafe_allow_html=True)

        # Stats resumidas
        stats = f"👍 {row['likes']} · 💬 {row['comments']} · 🔁 {row['shares']}"
        st.markdown(f'<div class="video-card-stats">{stats}</div>', unsafe_allow_html=True)

        # Botón "Ver guion"
        st.markdown(
            f"""
            <div style="text-align:center; margin-top:8px;">
                <a href="/Todos_los_guiones?id={row['id']}" target="_self">
                    <button class="video-card-btn">
                        Ver guión
                    </button>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)
