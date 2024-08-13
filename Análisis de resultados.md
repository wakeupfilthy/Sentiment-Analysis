# Análisis de sentimientos
Hernández A.
## Marco teórico
<b>Conjunto de datos:</b> Es un corpus de textos estructurados utilizados para entrenar y evaluar modelos de análisis de sentimientos. Pueden incluir reseñas, comentarios u opiniones etiquetadas con sentimientos (positivo, neutral, negativo).<br>
<b>Análisis de sentimientos:</b> Es el proceso de identificar y clasificar la polaridad emocional de textos, como opiniones de usuarios, para determinar si son positivas, negativas o neutras.<br>
<b>Preprocesamiento de texto:</b> Es el conjunto de técnicas para limpiar y normalizar datos textuales antes del análisis. Incluye eliminación de stopwords, lematización o stemming, y eliminación de caracteres especiales.<br>
<b>Diccionarios:</b> Son listas de palabras o términos asociados con categorías de sentimientos (positivo, negativo, neutro). Se utilizan para asignar polaridades a palabras en el análisis de sentimientos basado en reglas.<br>
<b>Embeddings:</b> Son representaciones numéricas de palabras o frases que capturan significados semánticos. Pueden ser preentrenados (como GloVe, Word2Vec) o aprendidos durante el entrenamiento del modelo.<br>
<b>Regresión logística:</b> Es un modelo de aprendizaje supervisado que se utiliza para la clasificación binaria o multiclase. Estima probabilidades utilizando la función logística y es eficaz para problemas linealmente separables.<br>
<b>Árboles de decisión:</b> Son modelos de aprendizaje supervisado que utilizan decisiones en forma de árbol para clasificar ejemplos. Dividen el espacio de características en regiones que corresponden a diferentes clases.<br>
<b>Máquinas de vectores de soporte (SVM):</b> Son algoritmos de aprendizaje supervisado que construyen hiperplanos en un espacio de alta dimensión para separar clases. Son efectivos en espacios de características no lineales.<br>
<b>Redes neuronales:</b> Son modelos de aprendizaje profundo inspirados en el cerebro humano. Pueden incluir capas de embeddings, convolucionales, recurrentes, etc., para aprender representaciones complejas y realizar clasificaciones precisas.<br>
<b>TF-IDF (Term Frequency-Inverse Document Frequency):</b> es una técnica de vectorización utilizada en procesamiento de lenguaje natural para convertir texto en una representación numérica. La idea básica es evaluar la importancia de una palabra en un documento dentro de un conjunto de documentos. La frecuencia de término (TF) mide cuántas veces aparece una palabra en un documento, mientras que la frecuencia inversa de documentos (IDF) mide la importancia de la palabra en el conjunto completo de documentos. Al multiplicar TF por IDF, se reduce la relevancia de palabras comunes y se aumenta la de términos que son más informativos en el contexto específico del documento.<br>
<b>One-Hot Encoding:</b> es una técnica utilizada para convertir variables categóricas, como palabras o etiquetas de clase, en una representación numérica que pueda ser procesada por la red.<br>

## Selección de herramientas
<b>Diccionarios Harvard IV-4 (HIV4):</b>
Son diccionarios léxicos que contienen palabras asociadas con categorías emocionales y psicológicas. Se utilizan en análisis de sentimientos para asignar polaridades a palabras en textos, ayudando a determinar la orientación emocional de los mensajes.<br>
<b>Opinion Lexicon:</b>
Es un conjunto de palabras etiquetadas manualmente como positivas o negativas. También se utiliza en análisis de sentimientos para asignar polaridades a palabras en textos, permitiendo clasificar opiniones según su tono emocional.<br>
<b>Pandas: </b>
Es una biblioteca de Python utilizada para manipular y analizar datos estructurados de manera eficiente. Proporciona estructuras de datos flexibles y herramientas para trabajar con tablas y series temporales, facilitando la exploración y limpieza de datos.<br>
<b>scikit-learn (sklearn):</b>
Es una biblioteca de aprendizaje automático en Python que ofrece herramientas simples y eficientes para el análisis predictivo de datos. Incluye algoritmos para clasificación, regresión, agrupamiento, entre otros, así como utilidades para preprocesamiento de datos y evaluación de modelos. <br>
<b>NLTK (Natural Language Toolkit):</b>
Es una plataforma en Python para el procesamiento del lenguaje natural (NLP). Proporciona herramientas para tokenizar, etiquetar, analizar y clasificar texto, así como acceso a corpus y recursos léxicos como WordNet y diccionarios sentimentales. <br>
<b>TensorFlow:</b>
Es una plataforma de código abierto desarrollada por Google para machine learning y deep learning. Permite construir y entrenar modelos de redes neuronales de manera eficiente, escalable y flexible, utilizando técnicas como redes convolucionales, recurrentes y embeddings.<br>
<b>Keras:</b>
Es una API de alto nivel para construir y entrenar modelos de deep learning, que puede ejecutarse sobre TensorFlow, Theano u otros backends. Simplifica la creación de redes neuronales mediante una interfaz intuitiva y modular. <br>
<b>GloVe (Global Vectors for Word Representation):</b>
Es un modelo de representación de palabras preentrenado desarrollado por Stanford. Proporciona embeddings de palabras aprendidos a partir de grandes corpus de texto, capturando relaciones semánticas y sintácticas entre palabras.<br>

## Parte 1. Adquisición de datos y análisis exploratorio de datos
El objetivo de esta practica es realizar un análisis de sentimientos, también conocida como polaridad de opinión; haciendo uso de diferentes aproximaciones. El análisis se hará sobre un conjunto de datos de opiniones de productos de Amazon, el cual está disponible en https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews. Con ayuda de la librería de pandas cargaremos este conjunto de datos y observaremos sus características. (figura1.1)
![image](https://github.com/user-attachments/assets/5c1fefbf-389e-41b4-9649-1b4a27749a8d) figura 1.1<br>
Podemos observar que el conjunto cuenta con 10 columnas, es decir, 10 dimensiones. También con más de medio millón de entradas. La interpretación de cada columna, así como su tipo de dato se puede observar en la tabla 1.1<br>

| Columna                 | Descripción                                                           | Tipo de dato |
| ----------------------- | --------------------------------------------------------------------- | ------------ |
| ID                      | Es la identificación única de la reseña                               | Entero       |
| Product ID              | Es la identificación única de un producto                             | Cadena       |
| User ID                 | Es la identificación única de un usuario                              | Cadena       |
| ProfileName             | Es el nombre del perfil del usuario<br>                               | Cadena       |
| Helpfulness Numerator   | Es el índice de usuarios que marcaron como útil la reseña             | Entero       |
| Helpfulness Denominator | Es el inidice de usuarios que visualizaron la reseña<br>              | Entero       |
| Score                   | Es la calificación asignada al producto                               | Entero       |
| Time                    | Es la hora en que se publicó la reseña                                | Entero       |
| Summary                 | Es una breve descripción de el producto o la experiencia del producto | Cadena       |
| Text                    | Es la reseña del producto                                             | Cadena       |
<br>
Tabla 1.1<br>
En la figura 1.2 podemos observar las primeras 5 filas del conjunto<br>
![image](https://github.com/user-attachments/assets/552918e5-d0a2-45f2-82e3-25ac8a64c863) figura 1.2<br>
Y unas estadísticas básicas del conjunto (Figura 1.3)<br>
![image](https://github.com/user-attachments/assets/aa841807-fe6c-4ac0-9205-c9fa3d5c5936) figura 1.3<br>
También será útil observar el total de calificaciones de productos (Figura 1.4) <br>
![image](https://github.com/user-attachments/assets/d62ea941-9897-4ee4-9a2a-96386fb0fc24) figura 1.4<br>

## Parte 2. Preprocesamiento de datos
Después de haber realizado el análisis exploratorio del conjunto, podemos decidir qué columnas nos pueden ser útiles en el análisis de sentimiento. La columna más importante es la de Text, debido a que ahí se encuentra la información que queremos analizar sobre la polaridad de la opinión de los usuarios sobre algún producto. También nos será útil la columna de Score, ya que nos dará una guía sobre la opinión del usuario, si le gustó o no el producto y qué tanto. En este caso no importa quien escribió la reseña, o a que producto, tampoco la hora o demás datos, así que podemos prescindir de ellos.<br>
Debido a la gran cantidad de datos en el conjunto, para agilizar el análisis y sin perder tanta información relevante, usaremos solo una parte del conjunto. Utilizaremos la columna Helpfulness Nominator para obtener las reseñas que resultaron más útiles o relevantes a otros usuarios y obtendremos las primeras 10,000. Se observa una parte del conjunto en la figura 2.1 <br>
![image](https://github.com/user-attachments/assets/847fa3ab-5b97-48d5-b7d3-e5c4bd584f97) figura 2.1<br>
Ahora crearemos una nueva columna llamada Sentiment la cual usaremos cómo etiqueta de clase para el sentimiento de la reseña basándose en la calificación del usuario. Sera Negativo para scores de 1 y 2, Neutral para scores de 3 y Positivo para scores de 4 y 5.  Finalmente, haremos un balance para tener las mismas instancias en cada clase y no tener clases predominantes en el entrenamiento. Entonces obtenemos un total de 700 instancias para cada clase. (Figura 2.2).<br>
![image](https://github.com/user-attachments/assets/d54aa4c3-74bc-4d1f-b65a-5b3e82a8edee) Figura 2.2 <br>

## Parte 3. Limpieza de datos
Al observar el texto del conjunto podemos observar que contiene ruido que puede afectar el procesamiento posterior, como etiquetas HTML, caracteres numéricos o signos de puntuación, por lo que usaremos expresiones regulares para eliminar las etiquetas y todos los caracteres que no representen letras. También normalizaremos el texto convirtiendo todo a letras minúsculas y eliminaremos las palabras vacías que no aportan información relevante para nuestro análisis, conocidas como stopwords. Puede que se requieran de más métodos de normalización para métodos posteriores, pero por ahora basta con esta limpieza básica. Podemos observar el nuevo texto y notar que el ruido ha disminuido bastante y las palabras que contienen puede ser de utilidad para nuestro análisis. (Figura 3.1)<br>
![image](https://github.com/user-attachments/assets/b45cc796-149b-4506-85ca-b695a6d7842f) figura 3.1<br>

## Parte 4. Análisis de sentimientos usando diccionarios.
Para esta aproximación haremos uso de los diccionarios Harvard IV-4 o HIV4 y de Opinion Lexicon.
Estos diccionarios precargados contienen información sobre la polaridad de opinión en palabras y obtener un score positivo y negativo sobre el sentimiento, así como una polaridad final sobre el sentimiento de un texto.<br>
Primero usaremos el diccionario de HIV4, entonces lo que se hará es usar el texto del conjunto previamente limpio, tokenizar el texto y obtener un score de estos diccionarios por cada token. Y realizar este proceso a cada fila del texto del conjunto. Dicha puntuación se guardará en una nueva columna llamada SentimentScore. Podemos observar algunos resultados en las figura 4.1, 4.2 y 4.3.<br>

![image](https://github.com/user-attachments/assets/78e7f936-47c2-489b-9339-b5e03ac6eb23) figura 4.1<br>
![image](https://github.com/user-attachments/assets/a6719fc4-1653-4806-83ae-7dab0c170a32) figura 4.2<br>
![image](https://github.com/user-attachments/assets/ea85bff3-f8e9-4b7c-a836-bb02dfe91332) figura 4.3<br>

Podemos resaltar la similitud de la polaridad de las reseñas con las etiquetas asignadas. En las reseñas negativas podemos observar valores bajos o negativos en la polaridad, mientras que en las reseñas positivas valores más altos. Pero en general los valores en la polaridad no coinciden tanto con la clase de la reseña, por lo que no es una manera tan confiable de realizar el análisis. <br>
Con Opinion Lexicon el proceso es muy similar a HIV4, tokenizamos las palabras y obtenemos un resultado, con la diferencia de que opinion lexicon devuelve los score positivos y negativos por separado, por lo que crearemos un diccionario con ambos scores. Luego obtenemos los scores del texto y obtenemos un total de polaridad restando los valores negativos a los positivos. El resultado se observa en la Figura 4.4, 4.4 y 4.6.<br>
![image](https://github.com/user-attachments/assets/aaffb3d2-c6c9-4b43-bfb1-e973dac15145) figura 4.4<br>
![image](https://github.com/user-attachments/assets/478cafaa-b2cd-4823-a335-6e6906b6afde) figura 4.5<br>
![image](https://github.com/user-attachments/assets/f8282d59-3842-434e-b8f4-35adc348d01f) figura 4.6<br>
De manera similar a HIV4 podemos observar que algunos valores coinciden con las etiquetas, pero muchos otros no. Aunque el desempeño general a simple vista parece mejor que el de HIV4.<br>

## Parte 5. Análisis de sentimientos con algoritmos de Machine Learning.
Para estas aproximaciones es necesario procesar un poco más los datos para obtener mejores resultados en el entrenamiento de los modelos. Por lo que primero habrá que tokenizar el texto, aplicar una lematización para obtener los lemas de las palabras y no tener tantos términos diferentes. También será necesario convertir el texto a una representación numérica y pueda ser interpretado por el modelo, esto lo hacemos con la técnica de TF-IDF (Term Frequency- Inverse Document Frequency). Los modelos por utilizar son el de Regresión Logística, Árboles de decisión y Maquinas de soporte vectorial (SVM). <br>
Una vez teniendo el texto procesado y transformado a vector, podemos entrenar los modelos usando también la columna Sentiment como etiquetas, con un conjunto divido de entrenamiento 80% y prueba 20%. Podemos observar el desempeño de los modelos. <br>
<b>Regresión Logística</b> (Figura 5.1). <br>
![image](https://github.com/user-attachments/assets/c0cd1873-91db-4a0a-b76c-eb225deb9f07) Figura 5.1 <br>
<b>Árboles de decisión</b> (Figura 5.2).<br>
![image](https://github.com/user-attachments/assets/625cd551-d10b-4284-b800-b7ff8bfeea4e) Figura 5.2 <br>
<b>SVM</b> (Figura 5.3).<br>
![image](https://github.com/user-attachments/assets/b7da4c7d-3f47-4506-b8a3-5f18ca9d8a6e) Figura 5.3 <br>
Se hace notar la capacidad de los modelos para clasificar de manera correcta las instancias correspondientes a las clases de negativos y positivos, pero no son tan capaces de identificar las instancias neutrales. Aún así logrando un buen desempeño general para la clasificación tomando en cuenta que fueron entrenadas con un conjunto relativamente pequeño. Comparando los modelos usando cross validation podemos notar que el algoritmo que mejor se desempeña para esta tarea es el de máquinas de soporte vectorial con una precisión de más del 70%. (Figura 5.4) <br>
![image](https://github.com/user-attachments/assets/e9e7988d-7e94-446c-8635-8fd760e92053) Figura 5.4 <br>

## Parte 6. Análisis de sentimientos con Word embeddings y redes neuronales.
Para esta aproximación es necesario transformar los datos de manera diferente, ya que se deberán transformar las palabras a vectores mediante el método de One-Hot Econding, por lo que primero habrá que separar las palabras tokenizando el texto. Con ayuda de los métodos de keras y tensorflow convertir los tokens en secuencias de números, será necesario también un proceso de padding para que todos los vectores o secuencias tengan el mismo tamaño. Igualmente, las etiquetas de clase se transformaran con One-Hot encoding. <br>
Una vez transformados los datos, la primera aproximación será usando Word embeddings precargados de GloVe https://nlp.stanford.edu/projects/glove/, para proporcionar contexto semántico previamente aprendido. Por lo que se hará una matriz de embeddings de tamaño 100 para la red neuronal convolucional. Se divide el conjunto de datos en 80% entrenamiento y 20% prueba. <br>
Se crea un modelo de red secuencial donde se añade una capa de embeddings utilizando la matriz de embeddings preentrenada de GloVe. Esta capa no es entrenable lo que significa que los valores de los embeddings no se actualizarán durante el entrenamiento. También se añade una capa de convolución 1D con 64 filtros y un tamaño de kernel de 5. Utiliza la activación ReLU. Una capa de Global Max Pooling que reduce la dimensionalidad. Una capa densa con 32 unidades y activación ReLU. Se añade dropout con una tasa de 0.5 para prevenir el sobreajuste. Una capa densa con 3 unidades y activación softmax para la clasificación de las tres clases (positivo, neutral, negativo).<br>
Finalmente se compila el modelo usando el optimizador Adam, la pérdida de entropía cruzada categórica y la métrica de precisión. El modelo se entrena durante 10 iteraciones y con un tamaño de batch de 32. Finalmente se obtiene el resultado siguiente. (Figura 6.1).<br>
![image](https://github.com/user-attachments/assets/d2064279-ad66-4e3d-adbf-ab80eb16f5af) Figura 6.1 <br>
Se observa que la precisión del entrenamiento mejora gradualmente desde 34.75% en la primera época hasta 81.71% en la décima época. La precisión de validación también mejora de 41.84% a 61.23% en la misma cantidad de épocas. El modelo tiene un desempeño moderado con una precisión de 61.23%. Quizás se deba a la cantidad baja de datos. <br>
La segunda aproximación será con Word embeddings aprendida del mismo conjunto. A diferencia del modelo anterior que utilizaba embeddings GloVe preentrenados, este modelo aprende los embeddings directamente desde los datos de entrenamiento. La dimensión de los embeddings se establece en 100. <br>
Aplica 64 filtros de convolución con tamaño de kernel de 5 sobre las secuencias de palabras, reduce la dimensionalidad tomando el valor máximo de cada filtro, cuenta con 32 unidades y función de activación ReLU y una tasa de 0.5 para prevenir el sobreajuste. Finalmente se añade una capa con 3 unidades y función de activación softmax para clasificar las reseñas en tres clases (positivo, neutral, negativo). <br>
Luego compilamos el modelo con el optimizador Adam y la función de pérdida categorical_crossentropy, adecuada para la clasificación multiclase. De igual manera el modelo se entrena durante 10 iteraciones con un tamaño de lote de 32. El resultado de este modelo se observa en la Figura 6.2. <br>
![image](https://github.com/user-attachments/assets/995f1e85-9fae-4909-88f1-4fd45500bbd1) Figura 6.2. <br>
Notamos que la precisión general del modelo es bastante buena (77.30%), especialmente en comparación con el modelo anterior que utilizaba embeddings GloVe preentrenados. Además, el modelo muestra un buen desempeño en todas las clases, especialmente en la clase negativa, donde la precisión y el F1-score son más altos. <br>

## Conclusiones.
Durante esta práctica, exploramos varias técnicas y modelos para abordar esta tarea utilizando datos de opiniones de productos de Amazon. A través de diversas aproximaciones, desde el uso de diccionarios predefinidos hasta modelos avanzados de aprendizaje profundo, pudimos observar diferentes niveles de precisión y eficacia en la clasificación de sentimientos. <br>
Las técnicas basadas en diccionarios como HIV4 y Opinion Lexicon proporcionaron una primera aproximación para asignar polaridades a las palabras en las reseñas. Sin embargo, su efectividad fue limitada debido a la subjetividad y ambigüedad en la interpretación de las palabras. <br>
Los modelos de aprendizaje supervisado, como Regresión Logística, Árboles de Decisión y SVM, mostraron una mejora significativa al utilizar representaciones vectoriales como TF-IDF para capturar la información semántica de las reseñas. SVM destacó como el modelo más robusto, superando el 70% de precisión en la clasificación de sentimientos positivos y negativos. <br>
En cuanto a las redes neuronales, experimentamos con embeddings preentrenados de GloVe y embeddings aprendidos directamente desde los datos. Estas aproximaciones demostraron ser efectivas para capturar el contexto semántico y mejorar la precisión de la clasificación, especialmente con la red que aprende embeddings específicos para el dominio del conjunto de datos.












