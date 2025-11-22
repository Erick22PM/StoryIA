1. Planteamiento del problema.
Problema: Los creadores novatos en TikTok no comprenden por qué algunos videos triunfan y otros no. Factores como el guion, la dscripción, hashtags o la miniatura influyen en el éxito, pero carecen de herramientas que los ayuden a analizarlos y mejorar.

Solución: Desarrollar un sistema que analice guiones, descripciones, hashtags y miniaturas de videos exitosos (de creadores de contenido enfocados al story telling) identificando patrones narrativos, visuales y semánticos mediante técnicas de machine learning.

Aplicación: El sistema ofrecerá retroalimentación automática y explicativa sobre cómo mejorar nuevos contenidos, calculará un puntaje de probabilidad de éxito. Incluirá un chatbot que traduzca los hallazgos en recomendaciones prácticas para los creadores. Y permitirá a los usuarios tener una plataforma que les ayude a mejorar en sus guiones y tomar inspiración.

2. Fuentes de datos:
Datos primarios (mediante scraping) TikTok:
Miniatura (imagen).
Descripción (incluye hashtags).
Transcripción (guion).
Métricas: likes, duración, compartidos, comentarios, canal.

3. EDA y visualización
Durante el análisis exploratorio (EDA) se buscará:

Tablas y análisis descriptivo:
Distribución de longitud de guiones (palabras, oraciones).
Promedio de likes, comentarios, etc. por categoría, canal, hashtags.
Top palabras más frecuentes en títulos exitosos.
Relación entre duración del video y visualizaciones.

Visualizaciones:
Boxplots: vistas vs. longitud del guión.
Nube de palabras: términos más comunes en guiones exitosos.
t-SNE o PCA 2D: para visualizar embeddings de guiones agrupados por estilo.
Clustering visual de miniaturas (CLIP embeddings): patrones de color o composición.
Gráfico de correlación: entre métricas del video y variables derivadas.

4. Ingeniería de variables
Nivel básico (textual y visual):
Guiones:
Número de palabras, oraciones, párrafos, caracteres, signos de puntuación, etc.
Longitud promedio de oración.
Frecuencia de sustantivos, verbos y adjetivos.
Densidad léxica.
Dominancia de pronombres personales (estilo narrativo).
Descripciones:
Número de palabras, número de urls, número de emojis, número de hashtags, lista de hashtags, número de caracteres, etc.
Ratio de mayusculas
Si hay preguntas, CTA.
Densidad de hashtag, emojis
Miniaturas:
Brillo, saturación y contraste promedio de la miniatura.
Número de caras detectadas o elementos visuales (usando OpenCV o CLIPSeg).
Etc.

Nivel intermedio (semántico y estructural):
Ngram_features con 51 palabras para el guión.
Para el guión: densidad informativa, complejidad gramatical, elementos de retención, emoción principal, longitud de tokens, resumen en una linea, story structure hook, story structure conflict, story structure climax, story structure resolution.
Cohesión semántica del guión (medida con embeddings entre párrafos).
Presencia de “hooks narrativos” (detectar preguntas o frases emocionales).
Clasificación temática del video (LDA o clustering).
Complejidad de lectura (índice Flesch en español).
Embedding de miniaturas.

Nivel avanzado (feature engineering cruzado y score):
Score de éxito de guión:
Entrenado mediante un modelo supervisado con etiquetas derivadas de likes, comentarios y compartidos normalizados.
Vector final multimodal (guión + imagen + métricas + descripción) para predicción y vectorización.


5. Modelación
Modelado no supervisado
Clustering de guiones (text embeddings):
Objetivo: agrupar estilos narrativos (reflexivo, cómico, descriptivo, épico, etc.).
Técnicas: K-Means, HDBSCAN, o UMAP + DBSCAN.
Modelo base: Embeddings CLIP o OpenAI text-embedding-3-large.
Clustering de miniaturas (image embeddings):
Objetivo: detectar patrones visuales exitosos (colores, texturas, estructura).
Modelo: CLIP ViT-B/32 o ViT-L/14.
Salida: agrupaciones de estilos visuales con ejemplos representativos

Modelado supervisado
Regresión / Clasificación (para score de guión):
Objetivo: predecir un “Score Narrativo” (0–100) basado en el guión.
Variable dependiente: vistas normalizadas.
Modelos:
Random Forest / XGBoost (features tabulares).
BERT fine-tuned o RoBERTa para clasificación textual.
Se dividirá en 4 sub modelos, uno de clasificación para ver si es un caso normal, bajo o viral. Y se hará el modelado correspondiente para cada caso. 
Similitud semántica (text + image retrieval):
Objetivo: recomendar guiones, hashtags o miniaturas similares a las ingresadas por el usuario.
Métrica: distancia coseno entre embeddings.

8. Elementos gráficos
Páginas (Streamlit):
Inicio: explicación del proyecto y carga de inputs.
Carga de inputs (guión y miniatura)
Análisis de inputs: 
 Guión: entrada de texto, score narrativo.
 Miniatura: muestra similitudes con otros videos.
Chatbot.
Página donde el usuario podrá recibir feedback del chatbot para mejorar su guión.
Hashtags
Página donde el usuario recibirá recomendaciones de hashtags para su video.
Genereación de audio
Página donde el usuario podrá generar un audio con voz artificial de su guión
Página para buscar videos y sus dashboards. Mediante:
 Busqueda de video por texto [text to embeding y buscar el más cercano].
 Navegación en todos los videos por miniatura y descripción para seleccionar uno.
 Dashboard de cada uno de los videos.
En cada una de estas páginas se mostrarán los 5 videos más parecidos en guión y miniatura.

Navegación:
Navbar para navegar entre las diferentes páginas.

## Stack tecnológico
| Tipo | Herramientas |
| --- | --- |
| Lenguaje | Python 3.11 |
| Frameworks IA | Hugginh Face, OpenAI API, LangChain |
| Embeddings | CLIP (ViT-B32), OpenAI text-embedding-3-large |
| Modelado ML | Keras, Scikit-learn, XGBoost, TensorFlow / PyTorch |
| NLP | spaCy, NLTK, Transformers |
| Visualización | Plotly, Matplotlib, WordCloud |
| Web scrapping | Selenium, Requests, BeautifulSoup |
| UI | Streamlit |
| Audio | ElevenLabs API, gTTS |
| Imagen | Pillow, OpenCV |
| Datos | Pandas, NumPy |
| Deploy | Streamlit Cloud |

## Consumo de modelos
| Tipo de modelo | Propósito | Ejemplo / API | Estado |
| --- | --- | --- | --- |
| LLM | Chatbot explicativo | gpt-4o-mini / gpt-4-turbo | Paga |
| Embeddings multimodales | Comparar texto e imágenes | CLIP (ViT-B/32) | Libre |
| Text-to-Speech | Generar narración del video | ElevenLabs / gTTS | Mixto |
| Modelo de regresión y clasificación / XGBoost | Calcular score narrativo | Entrenado localmente | Propio |
| Modelo de clustering (KMeans/DBSCAN) | Agrupar estilos | Scikit-learn | Propio |

## Pipeline / Inferencia

1. Input del usuario:
    - Guión (texto)
    - Miniatura (Imagen)

2. Extracción de features (CLIP / OpenAI / BERT)
    - Embeddings
    - Variables 

3. Modelos de análisis:
    - Score narrativo
    - Cluster asignado

4. Chatbot explicativo (LLM prompt tuning):
    - Explica mejoras y guión corregido.

5. Output visual: 
    - Score narrativo
    - Feedback textual




