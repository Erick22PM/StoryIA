import streamlit as st
import numpy as np
from utils.text_embeddings import embed_text_robusto

if "procesado" not in st.session_state:
    st.session_state.procesado = False

with st.sidebar:
    st.image("assets/logo.png", width=150)
    
st.title("📝 Cargar de archivos")

# --- Inicializar variables de sesión ---
if "guion_text" not in st.session_state:
    st.session_state.guion_text = None

if "guion_image" not in st.session_state:
    st.session_state.guion_image = None

if "guion_embedding" not in st.session_state:
    st.session_state.guion_embedding = None  # Embedding del guion

if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = True   # Comenzamos en modo edición


# ====================================================
# 🔧 MODO: EDITAR / CARGAR
# ====================================================
if st.session_state.edit_mode:

    st.subheader("Ingresar guión")
    guion = st.text_area(
        "Escribe o pega tu guión aquí:",
        height=250,
        value=st.session_state.guion_text if st.session_state.guion_text else ""
    )

    st.subheader("Subir imagen")
    imagen = st.file_uploader(
        "Sube una imagen",
        type=["jpg", "png", "jpeg"]
    )

    # Botón de guardar
    # Botón de guardar
    if st.button("Guardar datos"):
        if not guion.strip():
            st.warning("⚠️ Debes ingresar un guión antes de continuar.")
        else:
            guion_cambiado = (guion != st.session_state.guion_text)

            st.session_state.guion_text = guion
            if imagen:
                st.session_state.guion_image = imagen

            # ===================================================
            # 🔥 PROCESAMIENTO COMPLETO DEL GUION
            # ===================================================
            if guion_cambiado:
                with st.spinner("🔄 Procesando guión, generando embedding y score..."):

                    # 1. Generar embedding
                    embedding = embed_text_robusto(guion)
                    if embedding is None:
                        st.error("❌ Error generando embedding.")
                        st.stop()

                    st.session_state.guion_embedding = embedding

                    # 2. Importar tu función de modelos
                    from utils.procesar_guion import procesar_guion_completo

                    # 3. Procesar guion con tus modelos
                    with st.spinner("🏗 Generando score..."):
                        resultados = procesar_guion_completo(
                            texto=guion
                        )

                    # Guardar salida del modelo
                    st.session_state.guion_resultados = resultados
                    st.session_state.puntaje_modelo = float(resultados)
                    st.session_state.procesado = True

            st.session_state.edit_mode = False
            st.success("Datos procesados correctamente.")

            # ===================================================
            # 🚀 REDIRECCIÓN AUTOMÁTICA A 2_Guion.py
            # ===================================================
            st.switch_page("pages/2_Guion.py")


# ====================================================
# 📄 MODO: VISTA PREVIA
# ====================================================
else:

    st.markdown("## Contenido cargado")

    # Mostrar datos cargados
    if st.session_state.guion_image:

        col_img, col_text = st.columns([1, 2])

        with col_img:
            st.subheader("Imagen cargada")
            st.image(st.session_state.guion_image, width='stretch')

        with col_text:
            st.subheader("Guión cargado")
            st.write(st.session_state.guion_text)

    else:
        st.subheader("Guión cargado")
        st.write(st.session_state.guion_text)

    st.markdown("---")

    # Mostrar estado del embedding
    if st.session_state.guion_embedding is not None:
        st.success("🧠 Embedding listo y almacenado.")
        st.write(f"Dimensión del embedding: {len(st.session_state.guion_embedding)}")
    else:
        st.error("❌ No se ha generado embedding todavía.")

    st.markdown("---")

    # Botón para cambiar a modo edición
    if st.button("✏️ Editar archivos subidos o cargar nuevos"):
        st.session_state.edit_mode = True
        st.rerun()
