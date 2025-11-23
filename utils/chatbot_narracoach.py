import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from utils.data_loader import load_eda

# ======================================================
# Inicializar cliente OpenAI
# ======================================================
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ======================================================
# Sesión conversacional persistente
# ======================================================
class NarraCoachSession:

    def __init__(self, system_prompt):
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def send(self, user_message):
        self.messages.append({"role": "user", "content": user_message})

        resp = client_openai.responses.create(
            model="gpt-5",
            input=self.messages
        )

        assistant_text = resp.output_text.strip()
        self.messages.append({
            "role": "assistant",
            "content": assistant_text
        })

        return assistant_text



# ======================================================
# Similares por embedding
# ======================================================
def topk_similares_por_embedding(df_base, embedding_usuario, k=5):
    """
    df_base: DataFrame con 'embedding_guion'
    embedding_usuario: vector numpy (no un DF)
    """

    embedding_usuario = np.array(embedding_usuario).reshape(1, -1)
    matriz = np.stack(df_base["embedding_guion"].values)

    sims = cosine_similarity(embedding_usuario, matriz)[0]

    df_temp = df_base.copy()
    df_temp["similaridad"] = sims

    return df_temp.sort_values("similaridad", ascending=False).head(k)



# ======================================================
# NARRA COACH PRINCIPAL
# ======================================================
class NarraCoach:

    def __init__(self):
        """
        Carga un único dataframe que contiene:
        - embedding_guion
        - estilo_narrativo
        - densidad_informativa
        - complejidad_gramatical
        - emocion_principal
        - elementos_retencion
        y todo lo que necesites.
        """
        self.df = load_eda()

        # Validación mínima
        if "embedding_guion" not in self.df.columns:
            raise ValueError("❌ El parquet 2_EDA_AGENT.parquet no contiene 'embedding_guion'.")

        # Eliminar filas sin embedding
        self.df = self.df[self.df["embedding_guion"].notna()]



    # --------------------------------------------------------
    def obtener_textos_referencia(self, embedding_usuario, k=5):

        similares = topk_similares_por_embedding(
            self.df,
            embedding_usuario,
            k
        )

        ejemplos = []
        for i, row in enumerate(similares.itertuples(), start=1):
            texto = f"""
Ejemplo {i} — características narrativas:
• Estilo narrativo: {row.estilo_narrativo}
• Densidad informativa: {row.densidad_informativa}
• Complejidad gramatical: {row.complejidad_gramatical}
• Elementos de retención: {row.elementos_retencion}
• Emoción principal: {row.emocion_principal}
"""
            ejemplos.append(texto.strip())

        return "\n\n".join(ejemplos)



    # --------------------------------------------------------
    def crear_sesion(self, guion_usuario, embedding_usuario, score):
        """
        Prepara la sesión del agente con:
        - guion ingresado
        - similitudes por embedding
        - score pre-calculado
        """
        textos_referencia = self.obtener_textos_referencia(
            embedding_usuario,
            k=5
        )

        system_prompt = f"""
Eres Coach de StoryIA, experto de clase mundial en:
- narrativa para videos cortos (TikTok y Reels)
- técnicas modernas de retención y micro–storytelling
- análisis de guiones de crítica, reseña o análisis musical
- psicología de audiencia joven
- optimización de ritmo, claridad y emoción

Tu objetivo: Analizar y transformar el guion del usuario para hacerlo más claro, más atrapante y más memorable, incorporando aprendizajes de los guiones virales a partir de sus atributos narrativos, sin copiar ni inventar su contenido.

DATOS DISPONIBLES
El se te proporcionará:
1. Un guion original del usuario.
2. Un score de 0 a 100, donde:
    * 0 = el guion es muy débil → tienes mucha libertad de reescritura.
    * 100 = el guion es muy fuerte → debes mantener casi todo igual, solo pulir.

3. Características extraídas de los 5 guiones virales más parecidos (que son éxitosos), usando embeddings:
- estilo_narrativo_referencia
- densidad_informativa_referencia
- complejidad_gramatical_referencia
- elementos_retencion_referencia
- emocion_principal_referencia

Debes usar estos atributos activamente para orientar tu feedback y la reescritura.

Importante:
- No inventes ni reproduzcas ninguno de los guiones virales.
- Usa solo los atributos proporcionados como patrones narrativos.
- No menciones embeddings ni procesos técnicos.

🎯 MISIÓN DE LA RESPUESTA
La respuesta debe tener 3 secciones obligatorias:

1. Diagnóstico del guion
Explica con precisión:
- Qué funciona bien y por qué.
- Qué afecta la retención y el interés.
- Problemas de ritmo, densidad, claridad o emoción.
- Qué emoción transmite realmente vs. qué podría transmitir.
- Comparación explícita con los atributos virales:
    * en qué coincide,
    * en qué se aleja,
    * qué oportunidades hay.

2. Recomendaciones específicas
Basadas en:
- atributos virales
- técnicas de narrativa corta
- psicología y retención en TikTok

Incluye instrucciones accionables, como:
- mejoras del hook, tensión, giros, ritmo, final
- sugerencias de frases punchline o de gancho (solo dentro del tema del usuario)
- cómo aplicar el estilo, densidad o emoción de referencia sin copiar
- cómo simplificar sin perder fuerza

Importante:
Usa el score para ajustar el nivel de intervención:
- score bajo → ofrece cambios más radicales y estructurales
- score alto → ofrece ajustes finos, pulidos y micro-mejoras

3. Versión mejorada del guion
Reescribe el guion del usuario:
Condiciones:
- Mantén el contenido factual intacto.
- Transforma la forma, ritmo, emoción, estructura y claridad.
- Incorpora patrones presentes en los guiones virales según sus atributos:
    * si los virales son rápidos → hazlo más ágil
    * si usan emoción intensa → potencia emoción
    * si usan estructuras de giro → introdúcelas sutilmente
    * si usan densidad informativa → ajusta para igualar ese nivel

Nivel de libertad: Determinado por el score (0 = cambios fuertes, 100 = cambios suaves).
Debe sentirse:
- más memorable
- más atrapante
- más fluido
- más orientado a retención en TikTok

⚡ TONO Y ESTILO DEL COACH
- Profesional, directo y útil
- No condescendiente
- Explica el “por qué” de cada mejora
- Habla como alguien que optimiza guiones virales para creadores

❗ RESTRICCIONES
- No inventes hechos falsos sobre el artista, canción o álbum.
- No agregues contenido ajeno al tema del usuario.
- No cites directamente ningún guion viral.
- No menciones los 5 guiones ni sus textos.
- No menciones embeddings, distancias, ni procesos técnicos.
- Si falta información esencial, pídesela al usuario.

🟦 INPUT DEL USUARIO
- Guion propuesto: {guion_usuario}
- Score de narrativa calculado: {score:.2f}
- Ejemplos de guiones similares y sus características: {textos_referencia}
        """

        return NarraCoachSession(system_prompt)
