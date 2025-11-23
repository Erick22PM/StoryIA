

from utils.fijar_cpu import forzar_cpu
forzar_cpu()

import tensorflow as tf

# DESHABILITAR COMPILACIÓN JIT DENTRO DE TENSORFLOW
tf.config.optimizer.set_jit(False)

# Fuerza eager execution (evita graph mode = evita JIT)
tf.config.run_functions_eagerly(True)

import numpy as np
import utils.text_to_variables as ttv

from utils.data_loader import load_clas_model, load_r_bajo_model, load_r_normal_model, load_r_viral_model, load_scaler_bajo_y, load_scaler_viral_y, load_scaler_normal_y



def predecir_valor(df):
    """
    df_row: dataframe de 1 sola fila con EXACTAMENTE las columnas que usaste en entrenamiento.
    """

    # -----------------------------
    # 1. Convertir DF a vector
    # -----------------------------
    X = df

    # Convertir embeddings a matriz
    emb_matrix = np.stack(df["embedding_guion"].values)

    # Elimina esa columna del df
    df_no_emb = df.drop(columns=["embedding_guion"])

    # Concatenar features normales + embeddings
    X = np.hstack([df_no_emb.values, emb_matrix])

    # -----------------------------
    # 3. Clasificar la fila
    # -----------------------------
    model_clasificacion = load_clas_model()
    probs = model_clasificacion.predict(X)
    clase_pred = np.argmax(probs, axis=1)[0]   # 0=BAJO, 1=NORMAL, 2=VIRAL

    # -----------------------------
    # 4. Seleccionar modelo respectivo
    # -----------------------------
    if clase_pred == 0:
        modelo = load_r_bajo_model()
        scaler_y = load_scaler_bajo_y()
    elif clase_pred == 1:
        modelo = load_r_normal_model()
        scaler_y = load_scaler_normal_y()
    else:
        modelo = load_r_viral_model()
        scaler_y = load_scaler_viral_y()

    # -----------------------------
    # 5. Predecir valor
    # -----------------------------
    pred_scaled = modelo.predict(X)

    # Si escalaste y → revertir
    pred_final = scaler_y.inverse_transform(pred_scaled)[0][0]

    return {
        "clase_predicha": int(clase_pred),
        "valor_predicho": float(pred_final),
        "probabilidades_clases": probs[0].tolist()
    }

lista_palabras_cols = ['ahora', 'ahí', 'alguien', 'amor', 'aquí', 'aunque', 'años', 'aún', 'bien', 'cada', 'cosas', 'cuenta', 'cómo', 'da', 'después', 'dice', 'dos', 'día', 'embargo', 'entonces', 'forma', 'hace', 'hacer', 'hacia', 'historia', 'lugar', 'manera', 'mejor', 'mientras', 'mismo', 'momento', 'mundo', 'nunca', 'parece', 'parte', 'película', 'persona', 'personas', 'pueden', 'realidad', 'realmente', 'siempre', 'sino', 'tal', 'tan', 'tener', 'tiempo', 'va', 'veces', 'ver', 'vez']

def get_score_from_text(text):
    df = ttv.get_variables_from_text(text)


    transcripcion_cols = []
    for col in df.columns:
        if col.startswith("transcripcion"):
            transcripcion_cols.append(col)

    for col in ['transcripcion_longitud', 'transcripcion_palabras', 'transcripcion_letras', 'transcripcion_prom_palabras_oracion', 'transcripcion_palabras_unicas', 'transcripcion_vocales', 'transcripcion_consonantes', 'transcripcion_puntos', 'transcripcion_comas', 'transcripcion_exclamaciones', 'transcripcion_preguntas', 'transcripcion_signos_total']:
        transcripcion_cols.remove(col)

    df['transcripcion_palabras_no_stop'] = df['transcripcion_palabras_no_stop']/df['transcripcion_palabras']
    df['transcripcion_hooks'] = df['transcripcion_hooks']/df['transcripcion_palabras']
    df['transcripcion_oraciones'] = df['transcripcion_oraciones']/df['transcripcion_palabras']
    df['transcripcion_signos_total'] = df['transcripcion_signos_total']/df['transcripcion_letras']
    for col in lista_palabras_cols:
        df[col] = df[col] / df['transcripcion_palabras']

    df.drop(columns={'transcripcion_longitud', 'transcripcion_palabras', 'transcripcion_letras', 'transcripcion_prom_palabras_oracion', 'transcripcion_palabras_unicas', 'transcripcion_vocales', 'transcripcion_consonantes', 'transcripcion_puntos', 'transcripcion_comas', 'transcripcion_exclamaciones', 'transcripcion_preguntas'}, inplace=True)

    orden_columnas = ['transcripcion_ratio_unicas', 'transcripcion_ratio_vocales', 'transcripcion_ratio_consonantes', 'transcripcion_oraciones', 'transcripcion_puntos_norm', 'transcripcion_comas_norm', 'transcripcion_exclamaciones_norm', 'transcripcion_preguntas_norm', 'transcripcion_signos_total', 'transcripcion_signos_ratio', 'transcripcion_palabras_no_stop', 'transcripcion_densidad_lexica', 'transcripcion_repetitividad', 'transcripcion_indice_narrativo', 'transcripcion_f_sustantivos', 'transcripcion_f_verbos', 'transcripcion_f_adjetivos', 'transcripcion_f_pron_personales', 'transcripcion_cohesion_semantica', 'transcripcion_hooks', 'ahora', 'ahí', 'alguien', 'amor', 'aquí', 'aunque', 'años', 'aún', 'bien', 'cada', 'cosas', 'cuenta', 'cómo', 'da', 'después', 'dice', 'dos', 'día', 'embargo', 'entonces', 'forma', 'hace', 'hacer', 'hacia', 'historia', 'lugar', 'manera', 'mejor', 'mientras', 'mismo', 'momento', 'mundo', 'nunca', 'parece', 'parte', 'película', 'persona', 'personas', 'pueden', 'realidad', 'realmente', 'siempre', 'sino', 'tal', 'tan', 'tener', 'tiempo', 'va', 'veces', 'ver', 'vez', 'Cluster_hdbscan', 'embedding_guion']
    df = df[orden_columnas]


    resultado = predecir_valor(df)
    return resultado['valor_predicho']












