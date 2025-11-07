import os
import streamlit as st

st.title("StoryIA 📖")
st.caption("Este es un proyecto de IA para generar historias")

prompt = st.chat_input("Escribe tu historia")

if prompt:
    st.write(f"Historia: {prompt}")
