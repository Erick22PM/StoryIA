import streamlit as st
import utils.audio_generator
from utils.mostrar_datos_ingresados import mostrar_datos_ingresados
import os

from utils.chatbot_narracoach import NarraCoach
from utils.chat_ui import bubble_user, bubble_assistant, thinking_spinner

with st.sidebar:
    st.image("assets/logo.png", width=150)

st.title("📄 Vista del Guión Cargado")

# =====================================================
# 📌 VALIDAR SI EL USUARIO YA CARGÓ UN GUIÓN
# =====================================================
if "guion_text" not in st.session_state or st.session_state.guion_text is None:
    st.warning("⚠️ No hay guión cargado. Ve primero a la página 'Carga de archivos'.")
    if st.button("Ir a Cargar Guión"):
        st.switch_page("pages/1_Carga_de_archivos.py")
    st.stop()


# =====================================================
# 📄 MOSTRAR GUION, IMAGEN Y OTROS DATOS BÁSICOS
# =====================================================
mostrar_datos_ingresados(st.session_state)

st.markdown("---")


# =====================================================
# 📊 MOSTRAR PUNTAJE DEL MODELO
# =====================================================
st.subheader("📊 Score")

if "puntaje_modelo" in st.session_state:
    puntaje = st.session_state.puntaje_modelo
    st.success(f"✨ **Score del guión: {puntaje:.2f} / 100**")

    # Barra de progreso
    st.progress(min(max(puntaje / 100, 0), 1))  # normaliza entre 0 y 1
else:
    st.error("❌ No se encontró el puntaje del modelo.")
    st.stop()

if st.button("Ir a feedback del guión"):
    st.switch_page("pages/3_Chatbot.py")


st.markdown("---")

# =====================================================
# 🖼️ IMÁGENES SIMILARES (por embedding de imagen)
# =====================================================
st.subheader("🖼️ Imágenes similares a la miniatura cargada")

if "guion_image" not in st.session_state or st.session_state.guion_image is None:
    st.warning("⚠️ No se encontró la miniatura cargada.")
else:
    try:
        from utils.embedding_img import get_img_similares

        with st.spinner("🔍 Buscando imágenes similares..."):
            df_similares = get_img_similares(st.session_state.guion_image)

        if df_similares.empty:
            st.info("No se encontraron imágenes similares.")
        else:
            st.success("✨ Imágenes encontradas")

            canales = df_similares["canal"].unique()

            for canal in canales:
                st.write(f"### Canal: **{canal}**")

                df_canal = df_similares[df_similares["canal"] == canal]

                cols = st.columns(3)

                for idx, (_, row) in enumerate(df_canal.iterrows()):
                    col = cols[idx % 3]

                    with col:

                        # --- Mostrar imagen centrada ---
                        img = row.get("img_path", None)
                        if img and os.path.exists(img):
                            c1, c2, c3 = st.columns([1, 5, 1])
                            with c2:
                                st.image(img, width=180)
                        else:
                            st.image("https://via.placeholder.com/400x300?text=Sin+imagen", width=180)

                        # --- Título ---
                        titulo = f"{row.get('canal', 'Sin canal')} · {row['id']}"
                        st.markdown(
                            f'<div style="text-align:center; font-size:16px; font-weight:600; margin-top:4px;">{titulo}</div>',
                            unsafe_allow_html=True
                        )

                        # --- Distancia (score visual) ---
                        st.markdown(
                            f'<div style="text-align:center; font-size:14px; opacity:0.7;">Distancia: {row["distancia"]:.4f}</div>',
                            unsafe_allow_html=True
                        )

                        # --- Botón "Ver guión" ---
                        if st.button("Ver guión", key=f"btn_{row['id']}"):
                            st.session_state.selected_id = row["id"]
                            st.switch_page("pages/Todos_los_guiones.py")


                        # Separación vertical
                        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

            st.markdown("---")

    except Exception as e:
        st.error(f"❌ Error al buscar imágenes similares: {e}")

