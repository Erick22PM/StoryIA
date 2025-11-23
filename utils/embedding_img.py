import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.data_loader import load_main_dataset

# Cargar CLIP (ViT-B/32)
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Función para obtener embedding CLIP
def get_clip_embedding(path):
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        img_preprocessed = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img_preprocessed)
        # Normalizar el vector (opcional pero útil para similitud coseno)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error procesando {path}: {e}")
        return None

def get_clip_embedding_streamlit(input_image):
    """
    input_image puede ser:
      - UploadedFile de Streamlit
      - ruta local (string)
    """

    if hasattr(input_image, "read"):  # UploadedFile
        image = Image.open(input_image)  
    else:  # ruta string
        image = Image.open(str(input_image))

    image = image.convert("RGB")
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_image(image_input).cpu().numpy()[0]

    return embedding / np.linalg.norm(embedding)


def buscar_similares_por_canal(
    df_embeddings,
    img_path,
    top_k=5
):
    """
    df_embeddings: DataFrame con columnas ["id", "canal", "embedding"]
    img_path: ruta de la imagen que quieres analizar
    top_k: cantidad a retornar por canal (3)
    """

    # 1. Embedding de la imagen consultada
    query_vec = get_clip_embedding_streamlit(img_path)
    if query_vec is None:
        raise ValueError("No se pudo generar el embedding de la imagen de entrada.")

    query_vec = np.array(query_vec).reshape(1, -1)

    resultados = []

    # 2. Procesar por canal
    for canal, df_canal in df_embeddings.groupby("canal"):
        
        # Matriz de embeddings del canal
        X = np.vstack(df_canal["embedding_img"].values)

        # Distancia coseno
        sim = cosine_similarity(query_vec, X)[0]
        dist = 1 - sim

        # Top k
        idx_top = np.argsort(dist)[:top_k]

        for idx in idx_top:
            resultados.append({
                "id": df_canal.iloc[idx]["id"],
                "canal": canal,
                "distancia": float(dist[idx]),
                "embedding": df_canal.iloc[idx]["embedding_img"],
                "img_path": df_canal.iloc[idx].get("img_path", None)
            })

    # 3. Orden global final
    df_result = pd.DataFrame(resultados).sort_values("distancia").reset_index(drop=True)

    return df_result

def get_img_similares(image_from_user):
    df = load_main_dataset()

    image_to_compare = image_from_user

    df_result = buscar_similares_por_canal(
        df_embeddings=df,
        img_path=image_to_compare,
        top_k=3
    )

    return df_result



























