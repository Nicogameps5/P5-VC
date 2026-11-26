# Práctica 5 VC

Ejercicios de la quinta práctica de laboratorio de VC.

Nicolás Hernández Castro y Gabriel Godoy Navarro - EII ULPGC 26 de noviembre de 2026

## Prototipo 1: Sistema de Detección de Somnolencia

Este primer prototipo consiste en un monitor de somnolencia en tiempo real pensado para prevenir descuidos al volante o durante sesiones prolongadas de estudio. Analizando la entrada de la cámara web, el sistema combina rasgos de apariencia y geometría facial para identificar tanto bostezos como ojos cerrados (microsueños), lanzando avisos visuales dinámicos en pantalla según el nivel de alerta.

### 1. Selección y Adecuación del Conjunto de Datos

Para entrenar el clasificador biométrico, se empleó el conjunto de datos **OPENORCLOSEDEYES (CEW)**, disponible en https://www.kaggle.com/datasets/theabah0/openorclosedeyes/data.

Tras filtrar las imágenes en las que MediaPipe localiza el rostro con suficiente confianza `min_face_detection_confidence=0.4`, se obtuvieron 2.406 muestras válidas equilibradas entre ambas clases:

* **Ojos Cerrados (`ClosedFace`):** 1.176 muestras (48,9%)
* **Ojos Abiertos (`OpenFace`):** 1.230 muestras (51,1%)

Con el fin de reducir la variabilidad espacial y lumínica antes de extraer características, cada imagen pasa por un proceso de normalización geométrica y radiométrica a través de la función `extraer_roi_ocular(frame_bgr)`:

1. **Alineación horizontal:** Se calculan los centros de ambos ojos mediante los landmarks de MediaPipe (`[33, 133]` para el ojo izquierdo y `[362, 263]` para el derecho) y se aplica una transformación afín `cv2.warpAffine` para corregir cualquier inclinación de la cabeza.
2. **Recorte de la región de interés (ROI):** Se extrae un recorte horizontal de ambos ojos proporcional a la distancia que los separa `dist_ojos * 0.95` y se escala a 64x24 píxeles en escala de grises.
3. **Normalización de contraste:** Se utiliza ecualización adaptativa CLAHE `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))` para homogeneizar la iluminación de la zona ocular y reducir el impacto de las sombras o reflejos.


### 2. Extracción de Características y Entrenamiento del Modelo

Para no pasarle los píxeles en bruto al clasificador, cada recorte de los ojos se transforma primero en un vector numérico con la función `extraer_caracteristicas(img_gray)` que combina información de forma y textura:

* **HOG (*Histogram of Oriented Gradients*):** Captura la forma y orientación de los bordes (como la curvatura de los párpados, las pestañas y las cejas) trabajando con 9 orientaciones, celdas de 8x8 píxeles y bloques de 2x2.
* **LBP (*Local Binary Patterns*):** Analiza la textura de la imagen en escala de grises `local_binary_pattern(img_gray, P=8, R=1, method="uniform")`, dividiendo el recorte en una cuadrícula de 2x4 celdas para conservar la distribución espacial (diferenciando así el contraste del iris frente a la piel lisa del párpado cerrado).

Una vez extraídas y escaladas estas características `StandardScaler()`, se entrenó un clasificador SVM con kernel RBF `SVC(kernel="rbf", C=10, probability=True)`, evaluando su rendimiento mediante validación cruzada en 5 particiones `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` utilizando `cross_val_predict(modelo, X, y, cv=skf, n_jobs=-1)` para obtener las predicciones de todo el conjunto sin fuga de datos. Tras comprobar su rendimiento, el modelo se entrenó con el 100% de las muestras `modelo.fit(X, y)` y se guardó en disco (modelo_atencion.pkl) mediante `joblib` para utilizarlo en tiempo real.

![Resultados del entrenamiento y matriz de confusión](entrenamiento.png)

Al analizar los resultados obtenidos, se observa que el modelo logra un rendimiento muy sólido y equilibrado:

- **Precisión global y equilibrio entre clases:** El clasificador alcanza un 95% de acierto global (accuracy) sobre las 2.406 imágenes, manteniendo valores prácticamente idénticos de precision, recall y f1-score (entre 0,95 y 0,96) tanto en ojos abiertos como en ojos cerrados. Esto confirma que el modelo no tiene sesgo hacia ninguna de las dos categorías.
- **Análisis de la matriz de confusión:** De las 1.176 muestras de ojos cerrados, el sistema clasifica correctamente 1.114 y solo confunde 62. En el caso de los ojos abiertos, acierta 1.175, cometiendo únicamente 55 fallos.


### 3. Apartado Extra: Comparativa de Modelos Clásicos frente a Red Neuronal

Como apartado extra, se realizó una comparativa para observar cómo se comportan distintos modelos clásicos (SVM, KNN y Random Forest) frente a una red neuronal (MLP) al resolver este mismo problema.

Para evaluar ambos enfoques en igualdad de condiciones, se utilizaron exactamente los mismos vectores de características HOG + LBP estandarizados con `StandardScaler` y la misma validación cruzada de 5 particiones (`StratifiedKFold`), midiendo tanto el acierto como los tiempos de entrenamiento y predicción imagen a imagen (promediada sobre 500 muestras):

![Comparativa de modelos](comparativa.png)

Al contrastar los modelos clásicos con la red neuronal se observan diferencias en su comportamiento:

- **Capacidad de clasificación:** El SVM (95,14%) y la Red Neuronal MLP (94,01%) alcanzan un rendimiento muy similar y superior a Random Forest (92,02%) y KNN (91,52%). Esto refleja que mientras KNN se ve afectado por el ruido al comparar distancias directas entre tantos datos, tanto el kernel RBF del SVM como las capas ocultas de la red neuronal aprenden a separar patrones complejos dando más peso a los rasgos relevantes.
- **Velocidad de inferencia en tiempo real:** La Red Neuronal (MLP) es la más veloz prediciendo fotograma a fotograma (0,508 ms/imagen), ya que una vez entrenada solo realiza operaciones matriciales muy ligeras, seguida de cerca por el SVM (1,751 ms/imagen). Por el contrario, otros enfoques clásicos penalizan mucho la latencia en directo, ya que Random Forest (41,372 ms) debe evaluar 100 árboles por imagen y KNN (15,107 ms) tiene que calcular distancias contra las 2.406 muestras de entrenamiento en cada cuadro.
- **Facilidad de uso y configuración:** A nivel de programación, todos resultan igual de sencillos de implementar gracias a la interfaz unificada de `scikit-learn` (`make_pipeline`, `fit` y `predict`). La diferencia está en el ajuste previo, ya que los modelos clásicos funcionan prácticamente listos para usar con muy pocos parámetros (como el número de vecinos en KNN, los árboles en Random Forest o el kernel en SVM) y entrenan en apenas un segundo (entre 0,05 s y 1,30 s). Por su parte, la Red Neuronal (MLP) exige tomar más decisiones de diseño, como definir la arquitectura de capas ocultas, la tasa de aprendizaje o el límite de iteraciones para asegurar su convergencia, además de requerir algo más de tiempo de entrenamiento (2,76 s).


### 4. Demostración en tiempo real

Para llevar el sistema a tiempo real, se carga el clasificador entrenado (modelo_atencion.pkl) junto con el detector facial de MediaPipe y los emojis (durmiendo.png y bostezando.png). En cada fotograma capturado por la cámara web, la función `analizar_rostro_completo` aprovecha una única detección de landmarks para extraer dos señales simultáneas:

- **Detección ocular suavizada:** Se recorta y alinea la franja de los ojos con el mismo preprocesado del entrenamiento, se extraen sus características HOG + LBP y se calcula la probabilidad con el SVM (exigiendo `probs[0] > 0.65` para considerar el ojo cerrado). Para evitar que un parpadeo natural rápido dispare falsas alarmas, las predicciones pasan por una ventana deslizante de los últimos 5 fotogramas `deque(maxlen=5)` mediante votación mayoritaria.
- **Detección geométrica de bostezos:** Se calcula la apertura relativa de la boca dividiendo la distancia vertical de los labios interiores (landmarks 13 y 14) entre el ancho de las comisuras (landmarks 78 y 308). Si esta relación supera el umbral `0.58` en al menos 3 de los últimos 4 fotogramas `deque(maxlen=4)`, se activa el aviso de bostezo.

A partir de ambas señales, el sistema decide el estado dando prioridad al cierre ocular sobre el bostezo y actualiza la interfaz en pantalla:

- **Estado Despierto (`OpenFace`):** Muestra un marco verde alrededor de la imagen y el mensaje `"OJOS ABIERTOS - DESPIERTO"` en la barra superior.
- **Alerta Temprana de Fatiga (`Bostezo`):** Aplica un tinte naranja translúcido sobre el vídeo (`cv2.addWeighted`), cambia el borde a naranja con el texto `"BOSTEZO DETECTADO"` y superpone el emoji `bostezando.png` con un ligero movimiento vertical flotante.
- **Alerta de Microsueño (`ClosedFace`):** Aplica un filtro azulado nocturno sobre la escena, muestra el aviso `"OJOS CERRADOS - DURMIENDO... Zzz"` y anima en pantalla el emoji `durmiendo.png`.

![Demostración Prototipo 1](Prototipo1.gif)

