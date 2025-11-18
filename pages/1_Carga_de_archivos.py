import streamlit as st
import numpy as np
from utils.text_embeddings import embed_text_robusto

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
    if st.button("Guardar datos"):
        if not guion.strip():
            st.warning("⚠️ Debes ingresar un guión antes de continuar.")
        else:
            # Detectar si el guion cambió
            guion_cambiado = (guion != st.session_state.guion_text)

            st.session_state.guion_text = guion

            if imagen:
                st.session_state.guion_image = imagen

            # =========================================
            # 🔥 RE-CALCULAR EMBEDDING SI CAMBIÓ EL GUIÓN
            # =========================================
            if guion_cambiado:
                with st.spinner("🔄 Generando embedding del guión..."):
                    embedding = embed_text_robusto(guion)

                if embedding is None:
                    st.error("❌ Hubo un error generando el embedding.")
                else:
                    st.session_state.guion_embedding = embedding
                    st.success("✨ Embedding generado y almacenado.")

            st.session_state.edit_mode = False
            st.success("Datos guardados correctamente.")
            st.rerun()


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
