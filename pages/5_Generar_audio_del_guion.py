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

# Mostrar guion + imagen
utils.mostrar_datos_ingresados.mostrar_datos_ingresados(st.session_state)

# =====================================================
# Session state inicial
# =====================================================
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

if "audio_id" not in st.session_state:
    st.session_state.audio_id = None

if "eleven_api_key" not in st.session_state:
    st.session_state.eleven_api_key = ""


# =====================================================
# CAMPO PARA API KEY
# =====================================================
st.markdown("## 🔐 API Key de ElevenLabs")

st.session_state.eleven_api_key = st.text_input(
    "Ingresa tu API Key de ElevenLabs:",
    type="password",
    value=st.session_state.eleven_api_key,
    help="Puedes obtener tu API desde https://elevenlabs.io/app/settings/api-key"
)

# Validar si existe API antes de permitir generar audio
def validar_api(key):
    return key and len(key.strip()) > 10 and " " not in key


# =====================================================
# SECCIÓN DE GENERACIÓN DE AUDIO
# =====================================================
st.markdown("---")
st.markdown("## 🔊 Generar audio del guión")
st.write("Para obtener IDs de voces ingrese a: https://elevenlabs.io/app/voice-library (requiere registro)")

default_voice_id = "spPXlKT5a4JMfbhPRAzA"

audio_id = st.text_input(
    "ID de la voz:",
    value=st.session_state.audio_id if st.session_state.audio_id else default_voice_id,
    help="Puedes dejar el ID por defecto si no sabes cuál usar."
)


# =====================================================
# Función que genera el audio
# =====================================================
def generar_audio_handler():

    # Validar API
    if not validar_api(st.session_state.eleven_api_key):
        st.error("❌ Debes ingresar una API Key válida de ElevenLabs.")
        st.stop()

    # Validar ID de voz
    if not audio_id or len(audio_id.strip()) < 10:
        st.error("❌ El ID de voz no es válido.")
        st.stop()

    # Validar guion
    if len(st.session_state.guion_text.strip()) == 0:
        st.error("❌ El guion está vacío.")
        st.stop()

    st.info("⏳ Generando audio...")

    try:
        audio_bytes = utils.audio_generator.generar_audio(
            texto=st.session_state.guion_text,
            voice_id=audio_id,
            api_key=st.session_state.eleven_api_key
        )

        st.session_state.audio_bytes = audio_bytes
        st.session_state.audio_id = audio_id

        st.success("✅ Audio generado exitosamente")
        st.rerun()

    except ApiError as e:
        mensaje = e.body.get("detail", {}).get("message", "Error desconocido.")
        st.error(f"❌ Error de ElevenLabs: {mensaje}")

    except RequestException:
        st.error("❌ No se pudo conectar con ElevenLabs. Revisa tu conexión.")

    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")


# =====================================================
# Render de UI (mostrar audio o botón de generar)
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

    if st.button("🔁 Volver a generar audio"):
        generar_audio_handler()

else:
    if st.button("🎙️ Generar audio"):
        generar_audio_handler()
