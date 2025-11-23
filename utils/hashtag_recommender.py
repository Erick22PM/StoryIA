# utils/hashtag_recommender.py

import numpy as np
import re
from collections import Counter

STOPWORDS = {
    "el","la","los","las","un","una","uno","unos","unas",
    "de","del","al","a","y","o","en","es","que","por","para",
    "con","sin","se","su","sus","tu","te","lo","le","les",
    "mi","mis","me"
}

def extraer_palabras_clave(texto):
    """
    Extrae palabras clave sin usar modelos:
    - Mayúsculas completas
    - Nombres propios (Mayúscula Inicial)
    - Palabras repetidas
    """
    # Tokenizar básico
    palabras = re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", texto)

    claves = set()

    # 1. Palabras en MAYÚSCULAS
    for p in palabras:
        if len(p) > 3 and p.isupper():
            claves.add(p.lower())

    # 2. Nombres propios (Inicial mayúscula, no después de punto)
    for p in palabras:
        if p[0].isupper() and p.lower() not in STOPWORDS:
            claves.add(p.lower())

    # 3. Frecuencia contextual
    contador = Counter([p.lower() for p in palabras if p.lower() not in STOPWORDS])
    for palabra, freq in contador.items():
        if freq >= 3 and len(palabra) > 3:
            claves.add(palabra)

    return list(claves)


# ================================================================
#  RECOMENDAR HASHTAGS EXISTENTES (mejorada con palabras clave)
# ================================================================
def recomendar_hashtags_existentes(
    embedding_texto,
    df_hash,
    texto_original,
    top_k=10,
    alpha_sim=0.6,
    alpha_eng=0.2,
    alpha_kw=0.2
):

    # ---------------------------------
    # 1. Embedding matching
    # ---------------------------------
    q_emb = embedding_texto / np.linalg.norm(embedding_texto)

    hash_emb_matrix = np.stack(df_hash["embedding"].values)
    hash_emb_matrix = hash_emb_matrix / np.linalg.norm(hash_emb_matrix, axis=1, keepdims=True)

    similitudes = np.dot(hash_emb_matrix, q_emb)

    df_hash = df_hash.copy()
    df_hash["similitud"] = similitudes

    # ---------------------------------
    # 2. Engagement
    # ---------------------------------
    df_hash["eng_norm"] = (df_hash["engagement"] - df_hash["engagement"].min()) / \
                          (df_hash["engagement"].max() - df_hash["engagement"].min() + 1e-9)

    # ---------------------------------
    # 3. Palabras Clave
    # ---------------------------------
    keywords = extraer_palabras_clave(texto_original)
    keywords_set = set(keywords)

    def score_kw(hashtag):
        # match directo: si el hashtag contiene la keyword
        h = hashtag.lower().replace("#", "")
        return sum(1 for kw in keywords_set if kw in h)

    df_hash["kw_match"] = df_hash["hashtag"].apply(score_kw)

    # Normalizar kw_match (0 a 1)
    if df_hash["kw_match"].max() > 0:
        df_hash["kw_norm"] = df_hash["kw_match"] / df_hash["kw_match"].max()
    else:
        df_hash["kw_norm"] = 0

    # ---------------------------------
    # 4. Score Final (ajustable)
    # ---------------------------------
    df_hash["score"] = (
        alpha_sim * df_hash["similitud"] +
        alpha_eng * df_hash["eng_norm"] +
        alpha_kw * df_hash["kw_norm"]
    )

    return df_hash.sort_values("score", ascending=False).head(top_k)
