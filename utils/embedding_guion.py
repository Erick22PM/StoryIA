from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from openai import OpenAI
import time
from load_nltk import load_tokenizer
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
import os

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)

tokenizer = load_tokenizer()

# --- EMBEDDING INDIVIDUAL CON REINTENTOS ---
def embed_text_robusto(text, max_retries=5):
    for intento in range(max_retries):
        try:
            emb = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return np.array(emb.data[0].embedding)

        except Exception as e:
            # Espera exponencial
            time.sleep(1.5 ** intento)

            if intento == max_retries - 1:
                print(f"❌ Error definitivo al procesar texto: {e}")
                return None


# --- PROCESAR DATAFRAME EN PARALELO ---
def embed_dataframe_parallel(df, col="transcripcion", workers=8):
    df = df.copy()

    textos = df[col].tolist()
    embeddings = [None] * len(df)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(embed_text_robusto, textos[i]): i
            for i in range(len(textos))
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generando embeddings"):
            idx = futures[future]
            embeddings[idx] = future.result()

    df["embedding_guion"] = embeddings
    return df

def chunk_text(text, max_tokens=2000):
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)

    return chunks

def contar_tokens(text):
    return len(tokenizer.encode(text))

def embed_query_auto(text, token_threshold=8000):
    num_tokens = contar_tokens(text)

    if num_tokens < token_threshold:
        return embed_text_robusto(text)
    else:
        return embed_text_largo(text)


def embed_text_largo(text, max_retries=5):
    """Embedding robusto con chunking automático."""
    chunks = chunk_text(text, max_tokens=2000)
    embeddings = []

    for chunk in chunks:
        emb = embed_text_robusto(chunk, max_retries=max_retries)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return None

    # Promedio normalizado (recomendado para documentos largos)
    emb = np.mean(np.stack(embeddings), axis=0)
    emb = emb / np.linalg.norm(emb)

    return emb

def reprocesar_embeddings_nan(df, col_texto="transcripcion", col_emb="embedding_guion", workers=8):
    df = df.copy()

    # 1. Identificar NaNs
    mask_nan = df[col_emb].isna()
    idx_nan = df[mask_nan].index.tolist()

    print(f"🔍 Se encontraron {len(idx_nan)} registros sin embedding. Reprocesando...")

    # 2. Procesarlos en paralelo
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(embed_text_largo, df.loc[i, col_texto]): i
            for i in idx_nan
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Reprocesando NaNs"):
            i = futures[future]
            df.at[i, col_emb] = future.result()

    return df

def buscar_guiones_similares(df_embeddings, texto_query, top_k=5):
    # 1. Obtener embedding de la query según su longitud
    q_vec = embed_query_auto(texto_query)

    if q_vec is None:
        raise ValueError("No se pudo generar embedding para la query.")

    q_vec = q_vec.reshape(1, -1)

    # 2. Convertir embeddings existentes a matriz
    matriz = np.stack(df_embeddings["embedding_guion"].values)

    # 3. Calcular similitud
    sims = cosine_similarity(q_vec, matriz)[0]

    df_embeddings = df_embeddings.copy()
    df_embeddings["similaridad"] = sims

    # 4. Devolver resultados ordenados
    return df_embeddings.sort_values("similaridad", ascending=False).head(top_k)
