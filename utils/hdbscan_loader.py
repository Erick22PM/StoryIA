import os
import requests
import joblib
import gc

HDBSCAN_URL = "https://huggingface.co/ErickPM22/clustering/resolve/main/hdbscan_model.pkl"
LOCAL_HDBSCAN_PATH = "/mount/src/modelo_hdbscan.pkl"


def ensure_hdbscan_local():
    """
    Se asegura de que el archivo exista localmente.
    SOLO se descarga una vez por sesión del servidor.
    """
    if os.path.exists(LOCAL_HDBSCAN_PATH):
        return LOCAL_HDBSCAN_PATH

    print("📥 Descargando modelo HDBSCAN (una sola vez)...")
    
    r = requests.get(HDBSCAN_URL)
    r.raise_for_status()

    with open(LOCAL_HDBSCAN_PATH, "wb") as f:
        f.write(r.content)

    return LOCAL_HDBSCAN_PATH



def cargar_hdbscan_temporal():
    """
    Carga el modelo por unos segundos, lo devuelve y NO se mantiene en RAM.
    """
    path = ensure_hdbscan_local()
    paquete = joblib.load(path)

    clusterer = paquete["clusterer"]
    X_norm = paquete.get("X_norm", None)

    return clusterer, X_norm



def liberar_hdbscan(clusterer, X_norm=None, paquete=None):
    """
    Libera memoria inmediatamente.
    """
    try:
        del clusterer
    except:
        pass

    try:
        del X_norm
    except:
        pass

    try:
        del paquete
    except:
        pass

    gc.collect()
