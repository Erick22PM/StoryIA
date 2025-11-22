import streamlit as st
import utils.audio_generator
import utils.mostrar_datos_ingresados
from requests.exceptions import RequestException
from elevenlabs.core.api_error import ApiError

with st.sidebar:
    st.image("assets/logo.png", width=150)
st.title("🎙️ Generar audio del guión")

# =====================================================
# VALIDACIÓN DE GUION CARGADO
# =====================================================
if "guion_text" not in st.session_state or st.session_state.guion_text is None:
    st.warning("⚠️ No hay guión cargado. Ve primero a la página 'Carga de archivos'.")

    if st.button("Ir a Cargar Guión"):
        st.switch_page("pages/1_Carga_de_archivos.py")

    st.stop()


st.markdown("## 📝 Guión cargado")

# =====================================================
# MOSTRAR GUION + IMAGEN
# =====================================================

utils.mostrar_datos_ingresados.mostrar_datos_ingresados(st.session_state)

# =====================================================
# INICIALIZAR VARIABLES DE AUDIO EN SESSION_STATE
# =====================================================
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

if "audio_id" not in st.session_state:
    st.session_state.audio_id = None


# =====================================================
# SECCIÓN DE GENERACIÓN DE AUDIO
# =====================================================
st.markdown("---")
st.markdown("## 🔊 Generar audio del guión")
st.write("Para obtener IDs de voces ingrese a: https://elevenlabs.io/app/voice-library (requiere registro)")


# ID DE VOZ
default_voice_id = "spPXlKT5a4JMfbhPRAzA"

audio_id = st.text_input(
    "ID de la voz:",
    value=st.session_state.audio_id if st.session_state.audio_id else default_voice_id,
    help="Puedes dejar el ID por defecto si no sabes cuál usar."
)


# =====================================================
# VALIDADOR DE ID
# =====================================================
def validar_voice_id(v):
    if not v or len(v.strip()) < 10:
        return False
    if " " in v:
        return False
    return True


# =====================================================
# FUNCION INTERNA PARA GENERAR / REGENERAR AUDIO
# =====================================================
def generar_audio_handler():
    """Genera o regenera el audio con validadores y manejo de errores."""
    
    # Validar ID
    if not validar_voice_id(audio_id):
        st.error("❌ El ID de voz no es válido. Debe ser una cadena sin espacios y de al menos 10 caracteres.")
        st.stop()

    # Validar guion
    if len(st.session_state.guion_text.strip()) == 0:
        st.error("❌ El guion está vacío. Cárgalo nuevamente.")
        st.stop()

    # Advertencia por largo
    if len(st.session_state.guion_text) > 5000:
        st.warning("⚠️ Tu guion es muy largo. La generación podría tardar o fallar.")

    st.info("⏳ Generando audio... espera un momento.")

    try:
        audio_bytes = utils.audio_generator.generar_audio(st.session_state.guion_text, audio_id)

        if not audio_bytes or len(audio_bytes) < 50:
            st.error("⚠️ La API no devolvió audio. Revisa el ID de voz.")
            st.stop()

        # Guardar en session_state
        st.session_state.audio_bytes = audio_bytes
        st.session_state.audio_id = audio_id

        st.success("✅ Audio generado exitosamente")
        st.rerun()

    except ApiError as e:
        mensaje = e.body.get("detail", {}).get("message", "Error desconocido.")
        st.error(f"❌ Error de ElevenLabs: {mensaje}")

    except RequestException:
        st.error("❌ No se pudo conectar con ElevenLabs. Revisa tu conexión a internet.")
    
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")


# =====================================================
# SI YA EXISTE AUDIO → MOSTRAR Y PERMITIR REGENERAR
# =====================================================
if st.session_state.audio_bytes:
    st.audio(st.session_state.audio_bytes, format="audio/mp3")

    st.download_button(
        label="⬇️ Descargar audio",
        data=st.session_state.audio_bytes,
        file_name=f"{st.session_state.audio_id}.mp3",
        mime="audio/mpeg"
    )

    st.markdown("---")

    # 🔁 BOTÓN PARA REGENERAR
    if st.button("🔁 Volver a generar audio"):
        generar_audio_handler()


# =====================================================
# SI NO EXISTE AUDIO → MOSTRAR BOTÓN DE GENERAR
# =====================================================
else:
    if st.button("🎙️ Generar audio"):
        generar_audio_handler()
