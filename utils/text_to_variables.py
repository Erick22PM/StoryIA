import pandas as pd
pd.set_option("display.max_columns", None)

import utils.IngenieriaDeVariables as iv
import utils.embedding_guion as emgu
import re
from utils.data_loader import load_hdbscan_mod
import numpy as np
import hdbscan
from utils.fijar_cpu import forzar_cpu
forzar_cpu()

import tensorflow as tf

def asignar_cluster_produccion(texto, embed_fn, clusterer, threshold=0.10):
    # 1. embedding
    emb = embed_fn(texto)
    emb = np.array(emb, dtype=float)

    # 2. Normalizar
    emb_norm = emb / np.linalg.norm(emb)

    # 3. approximate_predict
    label, prob = hdbscan.approximate_predict(clusterer, emb_norm.reshape(1, -1))

    label = int(label[0])
    prob = float(prob[0])

    if prob < threshold:
        label = -1

    return label, prob


def cargar_modelo_hdbscan():
    return load_hdbscan_mod()

# Lista que ya tienes
lista_palabras_cols = [
    'ahora', 'ahí', 'alguien', 'amor', 'aquí', 'aunque', 'años', 'aún', 'bien',
    'cada', 'cosas', 'cuenta', 'cómo', 'da', 'después', 'dice', 'dos', 'día',
    'embargo', 'entonces', 'forma', 'hace', 'hacer', 'hacia', 'historia',
    'lugar', 'manera', 'mejor', 'mientras', 'mismo', 'momento', 'mundo',
    'nunca', 'parece', 'parte', 'película', 'persona', 'personas', 'pueden',
    'realidad', 'realmente', 'siempre', 'sino', 'tal', 'tan', 'tener', 'tiempo',
    'va', 'veces', 'ver', 'vez'
]
lista_no = [ 'descr_num_urls', 'canal',
 'descr_num_emojis', 'url',
 'descr_num_hashtags',
 'descr_hashtags_list',
 'descr_num_chars',
 'descr_num_chars_clean',
 'descr_num_words',
 'descr_num_words_clean',
 'descr_has_cta',
 'descr_has_question',
 'descr_upper_ratio',
 'descr_hashtag_density', 'descripcion_archivo',
 'descr_emoji_density',
 #'estilo_narrativo', 'densidad_informativa', 'complejidad_gramatical', 'elementos_retencion', 'emocion_principal', 'longitud_tokens',
 'ruta','img_path','embedding_img']

palabras = ['ahora', 'ahí', 'alguien', 'amor', 'aquí', 'aunque', 'años', 'aún', 'bien', 'cada', 'cosas', 'cuenta', 'cómo', 'da', 'después', 'dice', 'dos', 'día', 'embargo', 'entonces', 'forma', 'hace', 'hacer', 'hacia', 'historia', 'lugar', 'manera', 'mejor', 'mientras', 'mismo', 'momento', 'mundo', 'nunca', 'parece', 'parte', 'película', 'persona', 'personas', 'pueden', 'realidad', 'realmente', 'siempre', 'sino', 'tal', 'tan', 'tener', 'tiempo', 'va', 'veces', 'ver', 'vez']

def contar_palabras_en_texto(texto, palabras):
    """Cuenta ocurrencias exactas por palabra usando regex con límites de palabra."""
    if pd.isna(texto):
        texto = ""
    texto = texto.lower()

    counts = {}
    for palabra in palabras:
        patron = r'\b' + re.escape(palabra) + r'\b'
        counts[palabra] = len(re.findall(patron, texto))
    return counts


canales_es = ['faridieck', 'hernanmartinezl']
canales_en = ['kurz_gesagt']

def get_variables_from_text(text):
    data = {
        'id': '12341',
        'canal': 'nuevo',
        'transcripcion': f'''f{text}'''
    }

    df = pd.DataFrame([data])
    df = iv.generar_features_textuales_es(
        df,
        col_texto='transcripcion',
        col_titulo='titulo',
        n_workers=8
    ).copy()

    # Crear nuevas columnas
    for palabra in lista_palabras_cols:
        df[palabra] = 0   # inicializar

    # Rellenar
    for idx, texto in df['transcripcion'].items():
        counts = contar_palabras_en_texto(texto, lista_palabras_cols)
        for palabra, cnt in counts.items():
            df.at[idx, palabra] = cnt
            
    emb = emgu.embed_text_robusto(df.loc[0, "transcripcion"])
    df["embedding_guion"] = None
    df.at[0, "embedding_guion"] = emb.tolist()
    cols_new_model = []

    for col in df.columns:
        if not col in lista_no:
            cols_new_model.append(col)

    df_modelo = df[cols_new_model]
    df = df_modelo.copy()

    columnas_palabras = []
    for col in df.columns:
        if col.startswith("es_") or col.startswith("en_"):
            columnas_palabras.append(col)

    var = [i for i in df.columns if i not in columnas_palabras]

    clusterer, X_norm = cargar_modelo_hdbscan()
    label, prob = asignar_cluster_produccion(
        texto=df.iloc[0]['transcripcion'],
        embed_fn=emgu.embed_text_robusto,
        clusterer=clusterer
    )
    df['Cluster_hdbscan'] = label
    df.drop(columns={'transcripcion'}, inplace=True)

    for pla in palabras:      
        df[pla] = df[pla]/df['transcripcion_palabras']

    return df