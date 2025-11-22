import streamlit as st
from utils.chatbot_narracoach import NarraCoach
from openai import OpenAI
import os

# --- Bloqueo: si no está procesado, no puedes entrar ---
if "procesado" not in st.session_state or st.session_state.procesado is False:
    st.error("⚠️ Debes primero cargar y procesar un guion.")
    st.stop()

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)

st.title("🤖 Chat con NarraCoach")
st.caption("💬 Tu coach narrativo personal para mejorar tu guion.")

with st.sidebar:
    st.image("assets/logo.png", width=150)

# ======================================================
# VALIDAR DATOS NECESARIOS
# ======================================================
if "guion_text" not in st.session_state or st.session_state.guion_text is None:
    st.error("⚠️ Debes cargar un guion primero...")
    st.stop()

if "guion_embedding" not in st.session_state:
    st.error("⚠️ No se encontró embedding del guion.")
    st.stop()

if "puntaje_modelo" not in st.session_state:
    st.error("⚠️ Falta el score calculado del guion.")
    st.stop()


# ======================================================
# SESSION STATE
# ======================================================
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "narra_session" not in st.session_state:
    st.session_state.narra_session = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False


# ======================================================
# PRIMER MENSAJE AUTOMÁTICO
# ======================================================
if not st.session_state.initialized:

    # Crear sesión del coach
    coach = NarraCoach()
    st.session_state.narra_session = coach.crear_sesion(
        guion_usuario=st.session_state.guion_text,
        embedding_usuario=st.session_state.guion_embedding,
        score=st.session_state.puntaje_modelo
    )

    # Crear burbuja del bot para la respuesta inicial
    with st.chat_message("assistant") as container:

        # Mostrar spinner mientras genera la respuesta
        with st.spinner("✏️ NarraCoach está analizando tu guion..."):
            first_response = st.session_state.narra_session.send(
                "Genera un feedback narrativo inicial del guion del usuario."
            )

        # Mostrar respuesta final
        st.write(first_response)

    # Guardar historial
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": first_response}
    )

    st.session_state.initialized = True
    st.rerun()


# ======================================================
# MOSTRAR HISTORIAL COMPLETO
# ======================================================
for msg in st.session_state.chat_messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ======================================================
# ENTRADA DEL USUARIO
# ======================================================
if user_message := st.chat_input("Escribe tu mensaje aquí..."):

    # Mostrar y guardar mensaje del usuario
    st.chat_message("user").write(user_message)
    st.session_state.chat_messages.append({"role": "user", "content": user_message})

    # Crear burbuja del bot con streaming
    with st.chat_message("assistant"):

        # spinner para indicar "pensando..."
        with st.spinner("✏️ NarraCoach está pensando..."):
            response = st.session_state.narra_session.send(user_message)

        # Mostrar respuesta del bot
        st.write(response)

    # Guardar respuesta
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": response}
    )
