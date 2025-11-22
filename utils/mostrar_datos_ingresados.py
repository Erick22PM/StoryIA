import streamlit as st
import streamlit as st

def validar_imagen(session_state, nombre_variable="guion_image", pagina_destino="pages/1_Carga_de_archivos.py"):
    """
    Valida que session_state[nombre_variable] exista y tenga una imagen cargada válida.
    Si no existe, muestra una advertencia y un botón para ir a la página de carga.
    Devuelve True si existe, False si no.
    """

    imagen = session_state.get(nombre_variable)

    if imagen is None:
        st.warning("⚠️ No se ha cargado ninguna imagen.")

        if st.button("Ir a Cargar Imagen"):
            st.switch_page(pagina_destino)

        return False

    return True

def mostrar_datos_ingresados(session_state):
    st.markdown("## Guión cargado")

    # ============================
    # 1) Validar que exista guion
    # ============================
    guion = session_state.get("guion_text", None)

    if not guion or not str(guion).strip():
        st.warning("⚠️ No hay guión cargado. Ve primero a la página 'Carga de archivos'.")

        if st.button("Ir a Cargar Guión"):
            st.switch_page("pages/1_Carga_de_archivos.py")

        return  # No seguimos, porque sin guion no tiene sentido

    # ============================
    # 2) Validar imagen (opcional)
    # ============================
    imagen = session_state.get("guion_image", None)

    if not imagen:
        # No hay imagen -> avisamos y mostramos solo el texto
        st.warning("⚠️ No se ha cargado ninguna imagen. Ve a 'Carga de archivos' para subir una.")

        if st.button("Ir a Cargar Imagen"):
            st.switch_page("pages/1_Carga_de_archivos.py")

        st.subheader("Texto del guión")
        st.write(guion)
        return

    # ============================
    # 3) Si hay guion e imagen → layout completo
    # ============================
    col_img, col_text = st.columns([1, 2])

    with col_img:
        st.subheader("Imagen")
        st.image(imagen, width='stretch')

    with col_text:
        st.subheader("Texto del guión")
        st.write(guion)
