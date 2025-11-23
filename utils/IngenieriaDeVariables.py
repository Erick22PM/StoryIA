import pandas as pd
import numpy as np
import re
import emoji
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import util
from sklearn.feature_extraction.text import CountVectorizer

from utils.load_nltk import load_nltk
load_nltk()
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from utils.data_loader import load_spacy, load_embedder
from utils.load_nltk import load_nltk_stop_words

nlp_es = load_spacy()
embedder = load_embedder()

stop_words_es = load_nltk_stop_words()
vect_es = CountVectorizer(stop_words=stop_words_es, ngram_range=(1, 30), min_df=.01, max_df=100)


def generar_features_textuales_es(df, col_texto='transcripcion', col_titulo=None, n_workers=8):

    df = df.copy()
    prefix = f"{col_texto}_"

    # ==============================
    # LIMPIEZA BÁSICA
    # ==============================
    df['texto_limpio'] = (
        df[col_texto]
        .astype(str)
        .str.lower()
        .str.replace(r'[^a-záéíóúüñ0-9\s.,!?]', '', regex=True)
    )

    def safe_len(x): return len(x) if isinstance(x, str) else 0

    # ==============================
    # MÉTRICAS GENERALES
    # ==============================
    df[prefix+'longitud'] = df[col_texto].map(safe_len)
    df[prefix+'palabras'] = df['texto_limpio'].map(lambda x: len(x.split()))
    df[prefix+'palabras_unicas'] = df['texto_limpio'].map(lambda x: len(set(x.split())))
    df[prefix+'ratio_unicas'] = (df[prefix+'palabras_unicas'] / df[prefix+'palabras'].replace(0, np.nan)).fillna(0)
    
    df[prefix+'letras'] = df[col_texto].map(lambda x: sum(map(str.isalpha, x)))
    
    df[prefix+'vocales'] = df['texto_limpio'].map(lambda x: sum([x.count(i) for i in 'aeiouáéíóúü']))
    df[prefix+'consonantes'] = df['texto_limpio'].map(lambda x: sum([x.count(i) for i in 'bcdfghjklmnñpqrstvwxyz']))

    df[prefix+'ratio_vocales'] = df[prefix+'vocales'] / df[prefix+'longitud'].replace(0, np.nan)
    df[prefix+'ratio_consonantes'] = df[prefix+'consonantes'] / df[prefix+'longitud'].replace(0, np.nan)

    # ==============================
    # SIGNOS DE PUNTUACIÓN
    # ==============================
    df[prefix+'puntos'] = df[col_texto].map(lambda x: x.count('.'))
    df[prefix+'comas'] = df[col_texto].map(lambda x: x.count(','))
    df[prefix+'exclamaciones'] = df[col_texto].map(lambda x: x.count('!'))
    df[prefix+'preguntas'] = df[col_texto].map(lambda x: x.count('?'))

    df[prefix+'oraciones'] = df[col_texto].map(lambda x: len(sent_tokenize(str(x))) if isinstance(x, str) else 0)

    df[prefix+'prom_palabras_oracion'] = (
        df[prefix+'palabras'] / df[prefix+'oraciones'].replace(0, np.nan)
    ).fillna(0)

    # === NORMALIZACIONES ===
    df[prefix+'puntos_norm'] = df[prefix+'puntos'] / df[prefix+'longitud'].replace(0, np.nan)
    df[prefix+'comas_norm'] = df[prefix+'comas'] / df[prefix+'longitud'].replace(0, np.nan)
    df[prefix+'exclamaciones_norm'] = df[prefix+'exclamaciones'] / df[prefix+'oraciones'].replace(0, np.nan)
    df[prefix+'preguntas_norm'] = df[prefix+'preguntas'] / df[prefix+'oraciones'].replace(0, np.nan)

    df[prefix+'signos_total'] = (
        df[prefix+'puntos'] +
        df[prefix+'comas'] +
        df[prefix+'exclamaciones'] +
        df[prefix+'preguntas']
    )

    df[prefix+'signos_ratio'] = df[prefix+'signos_total'] / df[prefix+'longitud'].replace(0, np.nan)

    # ==============================
    # EMOJIS
    # ==============================
    df[prefix+'emojis'] = df[col_texto].map(emoji.emoji_count)

    # ==============================
    # PALABRAS NO STOPWORDS
    # ==============================
    df[prefix+'palabras_no_stop'] = [
        len([w for w in str(t).split() if w not in stop_words_es])
        for t in df['texto_limpio']
    ]

    df[prefix+'densidad_lexica'] = (
        df[prefix+'palabras_no_stop'] / df[prefix+'palabras'].replace(0, np.nan)
    ).fillna(0)

    # ==============================
    # REPETITIVIDAD
    # ==============================
    def repetitividad(texto):
        palabras = texto.split()
        if len(palabras) <= 1: return 0
        return 1 - len(set(palabras)) / len(palabras)

    df[prefix+'repetitividad'] = df['texto_limpio'].map(repetitividad)

    # ==============================
    # ÍNDICE NARRATIVO (tuneado)
    # ==============================
    df[prefix+'indice_narrativo'] = (
        0.4 * df[prefix+'densidad_lexica'] +
        0.3 * df[prefix+'ratio_unicas'] +
        0.3 * (1 - df[prefix+'repetitividad'])
    )

    # ==============================
    # spaCy features (todo en español)
    # ==============================
    print("⚙️ Procesando análisis lingüístico con spaCy...")

    f_sustantivos, f_verbos, f_adjetivos, f_pron_personales = [], [], [], []
    pronombres_es = {'yo','tú','vos','usted','él','ella','nosotros','ustedes','ellos','me','te','se','nos'}

    for i, texto in tqdm(df[col_texto].items(), total=len(df)):
        doc = nlp_es(str(texto))
        if len(doc) == 0:
            f_sustantivos.append(0); f_verbos.append(0); f_adjetivos.append(0); f_pron_personales.append(0)
            continue

        tokens = [t.text.lower() for t in doc]
        pos = [t.pos_ for t in doc]

        f_sustantivos.append(pos.count("NOUN") / len(doc))
        f_verbos.append(pos.count("VERB") / len(doc))
        f_adjetivos.append(pos.count("ADJ") / len(doc))
        f_pron_personales.append(sum(t in pronombres_es for t in tokens) / len(doc))

    df[prefix+'f_sustantivos'] = f_sustantivos
    df[prefix+'f_verbos'] = f_verbos
    df[prefix+'f_adjetivos'] = f_adjetivos
    df[prefix+'f_pron_personales'] = f_pron_personales

    # ==============================
    # EMBEDDINGS (paralelizado)
    # ==============================
    print("⚙️ Calculando embeddings...")
    textos = df[col_texto].astype(str).tolist()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(embedder.encode, t, convert_to_tensor=True) for t in textos]
        embeddings_texto = [f.result() for f in tqdm(as_completed(futures), total=len(futures))]

    # ==============================
    # COHESIÓN SEMÁNTICA
    # ==============================
    print("⚙️ Calculando cohesión semántica...")

    def cohesion_texto(texto):
        parrafos = [p.strip() for p in texto.split('\n') if len(p.strip()) > 0]
        if len(parrafos) <= 1: return 1
        emb = embedder.encode(parrafos, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb, emb)
        upper = sim[np.triu_indices(len(parrafos), k=1)]
        return float(upper.mean().item())

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(cohesion_texto, t) for t in textos]
        df[prefix+'cohesion_semantica'] = [f.result() for f in tqdm(as_completed(futures), total=len(futures))]

    # ==============================
    # SIMILARIDAD CON TÍTULO
    # ==============================
    if col_titulo and col_titulo in df.columns:
        print("⚙️ Calculando similaridad texto vs título...")
        embeddings_tit = embedder.encode(df[col_titulo].astype(str).tolist(), convert_to_tensor=True)
        sim_titulo = util.cos_sim(embeddings_tit, embeddings_texto).diagonal().cpu().numpy()
        df[prefix+'similitud_titulo'] = sim_titulo
    else:
        df[prefix+'similitud_titulo'] = np.nan

    # ==============================
    # HOOKS (solo español)
    # ==============================
    patrones_hook = re.compile(
        r"(por qué|cómo|quién|wow|increíble|sorprendente|atención|impactante|descubre|secreto)",
        re.IGNORECASE
    )
    df[prefix+'hooks'] = df[col_texto].map(lambda x: len(re.findall(patrones_hook, str(x))))

    print("✅ Features en español generadas con éxito.")
    return df


def generar_ngram_features_es(
    df,
    col_texto='transcripcion',
    ngram_range=(1, 3),
    min_df=0.01,
    max_df=0.5,
    max_features=None,
):
    """
    Genera columnas de n-gramas frecuentes SOLO PARA ESPAÑOL.
    
    Devuelve:
        df_out, vectorizer_es
    """

    df = df.copy()

    # ==============================
    # STOPWORDS EN ESPAÑOL
    # ==============================
    stop_es = stop_words_es

    # ==============================
    # LIMPIAR NOMBRES DE FEATURES
    # ==============================
    def clean_token(t):
        t = re.sub(r"\s+", "_", t.strip())  # espacios -> _
        t = re.sub(r"[^0-9a-zA-ZáéíóúüñÁÉÍÓÚÜÑ_]", "", t)  # quitar símbolos raros
        return t[:80]  # limitar tamaño

    # ==============================
    # APLICAR VECTORIZADOR ESPAÑOL
    # ==============================
    textos_es = df[col_texto].astype(str)

    vectorizer_es = CountVectorizer(
        stop_words=list(stop_es),
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )

    Xt = vectorizer_es.fit_transform(textos_es)

    # ==============================
    # CREAR COLUMNAS DE FEATURES
    # ==============================
    feature_names = vectorizer_es.get_feature_names_out()

    colnames = [clean_token(tok) for tok in feature_names]

    df_ngram = pd.DataFrame(
        Xt.toarray(),
        index=df.index,
        columns=colnames
    )

    # ==============================
    # CONCATENAR AL DATAFRAME
    # ==============================
    df_out = pd.concat([df, df_ngram], axis=1)

    return df_out, vectorizer_es

