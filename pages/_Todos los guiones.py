import streamlit as st
import pandas as pd
import os
import utils.dashboard_guion
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


st.title("📄 Vista de guión por ID")


# ======================================================
# Cargar dataframe (con cache)
# ======================================================
@st.cache_data
def load_data(parquet_path: str):
    return pd.read_parquet(parquet_path)


# Ruta a tu parquet (ajústala)
PARQUET_PATH = "./DATA/dataframes/final_datafame.parquet"

try:
    df = load_data(PARQUET_PATH)
except Exception as e:
    st.error(f"❌ Error cargando archivo parquet: {e}")
    st.stop()



# ======================================================
# Leer parámetro desde la URL
# ======================================================
query_params = st.query_params

video_id = query_params.get("id")

if video_id is None:
    st.warning("⚠️ Debes proporcionar un ID en la URL. Ejemplo:")
    st.code("localhost:8501/Todos_los_guiones?id=1234")
    st.stop()



# ======================================================
# Buscar el ID en el dataframe
# ======================================================
if video_id not in df["id"].astype(str).values:
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

# Audio original del video (si existe)
if ruta_audio and os.path.exists(ruta_audio):
    st.subheader("🎧 Audio del video")
    with open(ruta_audio, "rb") as f:
        st.audio(f.read(), format="audio/mp3")
else:
    st.info("No hay audio almacenado para este video.")



utils.dashboard_guion.get_dashboard_guion(transcripcion)



# ======================================================
# Recomendaciones
# ======================================================

st.markdown("## 🔍 Recomendaciones similares")

# Obtener el embedding del video actual
embedding_g = np.array(fila["embedding_guion"])
embedding_i = np.array(fila["embedding_img"])

# Crear matrices completas
matrix_g = np.stack(df["embedding_guion"].values)
matrix_i = np.stack(df["embedding_img"].values)

# Similitud coseno
sim_g = cosine_similarity([embedding_g], matrix_g)[0]
sim_i = cosine_similarity([embedding_i], matrix_i)[0]

# Añadir similitudes al DF
df["sim_guion"] = sim_g
df["sim_img"] = sim_i

# Filtrar top 5 sin incluir el propio video
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
        else:
            st.warning("No img")

        st.markdown(
            f"""
            <div style="text-align:center; margin-top:5px;">
                <a href="/Todos_los_guiones?id={r['id']}" target="_self">
                    <button style="
                        padding:6px 12px;
                        background:#1f6feb;
                        color:white;
                        border:none;
                        border-radius:6px;
                        cursor:pointer;
                        font-size:14px;
                    ">
                        Ver guión
                    </button>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

# Separator
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
            st.warning("No img")

        st.markdown(
            f"""
            <div style="text-align:center; margin-top:5px;">
                <a href="/Todos_los_guiones?id={r['id']}" target="_self">
                    <button style="
                        padding:6px 12px;
                        background:#1f6feb;
                        color:white;
                        border:none;
                        border-radius:6px;
                        cursor:pointer;
                        font-size:14px;
                    ">
                        Ver guión
                    </button>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )