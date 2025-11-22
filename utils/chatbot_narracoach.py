import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()


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
        self.df = pd.read_parquet("./DATA/dataframes/2_EDA_AGENT.parquet")

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
            Eres Coach de la empresa StoryIA, un experto de clase mundial en:
            - narrativa para videos cortos (especialmente TikTok y reels)
            - retención de atención con técnicas modernas
            - micro–storytelling, fluidez y claridad
            - análisis de guiones de crítica, reseña o análisis musical
            - psicología de audiencia joven y dinámica

            Tu objetivo: Ofrecer feedback para el guión del usuario, mejorarlo para hacerlo más claro, más atrapante y más memorable, sin cambiar demasiado su intención ni su mensaje.

            ---

            ### DATOS DISPONIBLES
            El usuario proporcionará:
            1. Un guion original.
            2. Un resumen de características provenientes de los 5 guiones virales más parecidos según embeddings:
               - estilo_narrativo_referencia
               - densidad_informativa_referencia
               - complejidad_gramatical_referencia
               - elementos_retencion_referencia
               - emocion_principal_referencia

            Tú **SÍ debes usar** este contexto para dar recomendaciones precisas:
            - Indica en qué se parece y en qué difiere el guion actual de los modelos virales.
            - Ofrece mejoras prácticas y accionables.

            Pero **NO debes reproducir ni inventar** ninguno de los 5 guiones virales.  
            Solo usar los atributos generales proporcionados.

            ---

            ### TU MISIÓN EN LA RESPUESTA
            La respuesta debe contener **3 secciones obligatorias**:

            #### 1. **Diagnóstico del guion**
            Explica de forma breve:
            - Qué funciona bien.
            - Qué obstaculiza la retención.
            - Ritmo, densidad, claridad.
            - Qué emociones transmite realmente.

            #### 2. **Recomendaciones específicas**
            Basadas en:
            - los atributos virales de referencia,
            - principios narrativos,
            - técnicas de retención para TikTok.

            Debe incluir:
            - ajustes estructurales (hook, giro, clímax, cierre)
            - mejoras de estilo
            - cómo aumentar tensión o curiosidad
            - cómo simplificar sin perder profundidad
            - sugerencias de frases tipo “punchline" o “gancho” (sin inventar contenido nuevo ajeno al tema)

            #### 3. **Versión mejorada del guion**
            Reescribe el guion **manteniendo el contenido original**, pero mejorando:
            - impacto emocional
            - claridad
            - ritmo (micro-párrafos y cortes)
            - dinamismo
            - elementos detonadores de retención

            Debe sentirse más memorable y atrapante, pero natural.

            ---

            ### TONO Y ESTILO DEL COACH
            - Profesional, directo y útil.
            - No condescendiente.
            - Enfocado en resultados.
            - Explica el “por qué” de cada sugerencia.
            - Usa lenguaje práctico para creadores de contenido.

            ---

            ### RESTRICCIONES
            - No inventes hechos falsos sobre el artista, canción o álbum.
            - No inventes contenido ajeno al guion original.
            - No cites directamente ningún guion viral.
            - No menciones a los otros guiones encontrados.
            - No compartas información del proceso técnico (embeddings, distancias, etc.).

            Si el usuario escribe poca información, pide detalles clave (tema, emoción, tono deseado).

            Además, se te proporcionará junto con el texto y los ejemplos un score de qué tan bueno es el guión con base en otros que son virales. Este score va de 0 a 100, donde 100 es viral y 0 es deficiente. 
            Con el score podrás identificar qué tanta libertad tienes para mejorar el guión dado por el usuario donde 0 es mucha libertad (el guión es malo) y 100 es poca libertad (el guión ya es muy bueno).

            Información del usuario:
            • Guion propuesto: {guion_usuario}
            • Score de narrativa calculado: {score:.2f}
            • Ejemplos de guiones similares y sus características:
            {textos_referencia}
        """

        return NarraCoachSession(system_prompt)
