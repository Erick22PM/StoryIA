# utils/hashtag_openai.py
import json
from openai import OpenAI
import os

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)

def generar_hashtags_desde_guion(guion_text: str) -> dict:
    """
    Devuelve un diccionario:
    {
        "relacionados": [ "#hashtag1", ... 5 en total ],
        "viralidad": [ "#hashtag6", ... 5 en total ]
    }
    """
    mensaje_sistema = (
        "Eres un experto en marketing digital y redes sociales. "
        "Tu tarea es proponer hashtags para videos tipo TikTok/shorts/YouTube."
    )

    mensaje_usuario = f"""
Tengo el siguiente guion de video:

\"\"\"{guion_text}\"\"\"

1) Genera 5 hashtags específicos y directamente relacionados con el contenido del guion.
2) Genera 5 hashtags más generales sobre redes sociales, crecimiento orgánico, viralidad y alcance (hashtags populares en Tiktok).

**Formato de salida obligatorio (JSON puro, sin texto extra):**
{{
  "relacionados": ["#hashtag1", "#hashtag2", "#hashtag3", "#hashtag4", "#hashtag5"],
  "viralidad": ["#hashtag6", "#hashtag7", "#hashtag8", "#hashtag9", "#hashtag10"]
}}

No escribas nada que no sea JSON válido.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": mensaje_sistema},
            {"role": "user", "content": mensaje_usuario},
        ],
        temperature=0.7,
    )

    contenido = resp.choices[0].message.content

    # Parsear JSON devuelto por el modelo
    data = json.loads(contenido)
    return data
