import streamlit as st

# ============================================
# 🔵 BURBUJA DEL USUARIO
# ============================================
def bubble_user(text):
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: flex-end;
            margin: 8px 0;
        ">
            <div style="
                background-color: #DCFCE7;
                color: #065F46;
                padding: 10px 14px;
                border-radius: 12px;
                max-width: 70%;
                font-size: 15px;
                border: 1px solid #A7F3D0;
            ">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================
# 🟢 BURBUJA DEL ASISTENTE
# ============================================
def bubble_assistant(text):
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: flex-start;
            margin: 8px 0;
        ">
            <div style="
                background-color: #F3F4F6;
                color: #111827;
                padding: 10px 14px;
                border-radius: 12px;
                max-width: 70%;
                font-size: 15px;
                border: 1px solid #D1D5DB;
            ">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================
# ⏳ LOADER DEL BOT
# ============================================
def thinking_spinner(container):
    with container:
        st.markdown(
            """
            <div style="
                display: flex;
                justify-content: flex-start;
                margin: 8px 0;
            ">
                <div style="
                    background-color: #F3F4F6;
                    color: #6B7280;
                    padding: 10px 14px;
                    border-radius: 12px;
                    font-style: italic;
                    border: 1px solid #E5E7EB;
                ">
                    ✏️ NarraCoach está pensando...
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
