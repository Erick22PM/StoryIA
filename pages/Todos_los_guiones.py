import streamlit as st
import pandas as pd
import os
import utils.dashboard_guion
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
from utils.data_loader import load_main_dataset
with st.sidebar:
    st.image("assets/logo.png", width=150)
st.title("📄 Vista de guión por ID")


# ======================================================
# Cargar dataframe (con cache)
# ======================================================

PARQUET_PATH = "./DATA/dataframes/PROD_DATASET.parquet"

try:
    df = load_main_dataset()
except Exception as e:
    st.error(f"❌ Error cargando archivo parquet: {e}")
    st.stop()


# ======================================================
# OBTENER ID (PRIMERO DE session_state, LUEGO DE URL)
# ======================================================
video_id = None

# Prioridad 1 → session_state
if "selected_id" in st.session_state and st.session_state.selected_id is not None:
    video_id = str(st.session_state.selected_id)

# Prioridad 2 → parámetros URL (compatibilidad con enlaces antiguos)
query_params = st.query_params
if video_id is None:
    video_id = query_params.get("id", None)

# Validación final
if video_id is None:
    st.error("❌ No se encontró un ID válido. Usa un botón 'Ver guión' o la URL con parámetro ?id=XXX.")
    st.stop()


# ======================================================
# Buscar el ID en el dataframe
# ======================================================
if str(video_id) not in df["id"].astype(str).values:
    st.error(f"❌ El ID '{video_id}' no existe en la base de datos.")
    st.stop()

fila = df[df["id"].astype(str) == str(video_id)].iloc[0]


# ======================================================
# Extraer datos relevantes
# ======================================================
transcripcion = fila["transcripcion"]
descripcion = fila["descripcion_archivo"]
likes = format(int(fila["likes"]), ",")
comments = format(int(fila["comments"]), ",")
shares = format(int(fila["shares"]), ",")
duracion = round(fila["duracion_seg"], 1)
url = fila["url"]
img_path = fila["img_path"]
canal = fila.get("canal", "No especificado")
ruta_audio = fila.get("ruta", None)


# ======================================================
# Render visual
# ======================================================
st.markdown(f"## 🎬 Video ID: `{video_id}`")
st.write(f"**Canal:** {canal}")
st.write(f"**Duración:** {duracion} segundos")
st.write(f"**Likes:** {likes} | **Comments:** {comments} | **Shares:** {shares}")
st.write(f"[🔗 Ver video original]({url})")
st.markdown("---")


# Mostrar guion e imagen
col_img, col_text = st.columns([1, 2])

with col_img:
    if img_path and os.path.exists(img_path):
        st.subheader("🖼 Miniatura")
        st.image(img_path, width='stretch')
    else:
        st.warning("⚠️ Imagen no encontrada.")

with col_text:
    st.subheader("📄 Guión")
    st.write(transcripcion)

st.markdown("---")

# Mostrar descripción
st.subheader("📝 Descripción (incluye hashtags)")
st.write(descripcion)

st.markdown("---")

# ======================================================
# 🧠 ANÁLISIS NARRATIVO
# ======================================================
st.subheader("🧠 Análisis narrativo del guion")

estilo = fila.get("estilo_narrativo", "No disponible")
densidad = fila.get("densidad_informativa", "No disponible")
complejidad = fila.get("complejidad_gramatical", "No disponible")

elementos_raw = fila.get("elementos_retencion", [])
if isinstance(elementos_raw, str):
    try:
        elementos = ast.literal_eval(elementos_raw)
    except:
        elementos = [elementos_raw]
else:
    elementos = elementos_raw

emocion = fila.get("emocion_principal", "No disponible")
tokens = fila.get("longitud_tokens", "No disponible")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎭 Estilo narrativo")
    st.write(estilo)

    st.markdown("### 📚 Complejidad gramatical")
    st.write(complejidad)

    st.markdown("### 🔥 Emoción principal")
    st.write(emocion)

with col2:
    st.markdown("### 🧩 Densidad informativa")
    st.write(densidad)

    st.markdown("### 🔢 Longitud en tokens")
    st.write(tokens)

st.markdown("### 🎯 Elementos de retención")
if isinstance(elementos, list):
    for e in elementos:
        st.markdown(f"- {e}")
else:
    st.write(elementos)

utils.dashboard_guion.get_dashboard_guion(transcripcion)


# ======================================================
# 🔍 RECOMENDACIONES
# ======================================================
st.markdown("## 🔍 Recomendaciones similares")

embedding_g = np.array(fila["embedding_guion"])
embedding_i = np.array(fila["embedding_img"])

matrix_g = np.stack(df["embedding_guion"].values)
matrix_i = np.stack(df["embedding_img"].values)

sim_g = cosine_similarity([embedding_g], matrix_g)[0]
sim_i = cosine_similarity([embedding_i], matrix_i)[0]

df["sim_guion"] = sim_g
df["sim_img"] = sim_i

df_recom_g = df[df["id"] != fila["id"]].nlargest(5, "sim_guion")
df_recom_i = df[df["id"] != fila["id"]].nlargest(5, "sim_img")


# =============================
# Recomendaciones por GUION
# =============================
st.subheader("✍️ Recomendaciones por similitud de guion")

cols = st.columns(5)
for col, (_, r) in zip(cols, df_recom_g.iterrows()):
    with col:
        img = r.get("img_path", None)
        if img and os.path.exists(img):
            st.image(img, width='stretch')

        if st.button("Ver guión", key=f"btn_g_{r['id']}"):
            st.session_state.selected_id = r["id"]
            st.switch_page("pages/Todos_los_guiones.py")

st.markdown("---")


# =============================
# Recomendaciones por IMAGEN
# =============================
st.subheader("🖼 Recomendaciones por similitud de imagen")

cols = st.columns(5)
for col, (_, r) in zip(cols, df_recom_i.iterrows()):
    with col:
        img = r.get("img_path", None)
        if img and os.path.exists(img):
            st.image(img, width='stretch')
        else:
            st.image("https://via.placeholder.com/400x300?text=Sin+imagen", width='stretch')

        if st.button("Ver guión", key=f"btn_i_{r['id']}"):
            st.session_state.selected_id = r["id"]
            st.switch_page("pages/Todos_los_guiones.py")

