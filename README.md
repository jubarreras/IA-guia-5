# IA-guia-5
Solución a la guía #5 de la materia de Inteligencia Artificial en la Universidad Nacional de Colombia, hecha por los estudiante: Juan David Barrera Salamanca, Yeray Dario Moreno Rangel y Juan Diego Lozano Colmenares

### Cada uno de ustedes consiga 100 imágenes de perros y 100 imágenes de gatos. Conformen un dataset de perros y gatos del curso. Desarrollen un modelo de clasificación de perros y gatos, usando redes neuronales.

El proceso incluirá la preparación de los datos, la construcción de la arquitectura de la red neuronal, el entrenamiento del modelo y la evaluación de su rendimiento. Para guiar este proceso, he encontrado un código en Kaggle que implementa un modelo de clasificación de imágenes utilizando redes neuronales convolucionales (CNN), una variante especialmente efectiva para tareas de visión por computadora. A lo largo de este proyecto, explicaremos paso a paso cómo funciona este código, desde la carga y preprocesamiento de las imágenes hasta el entrenamiento y la validación del modelo.
**1. Importación de librerias**
```python
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from  tensorflow.keras import models, optimizers, regularizers
```

**1. Importación de librerias**
```python
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from  tensorflow.keras import models, optimizers, regularizers
```
**2. Define una red neuronal convolucional (CNN) para clasificar imágenes de perros y gatos. La red comienza con capas convolucionales (Conv2D) que extraen características de las imágenes usando filtros, seguidas de capas de pooling (MaxPooling2D) que reducen la dimensionalidad. Luego, una capa Flatten convierte los datos en un vector 1D, y se añade una capa densa (Dense) con 512 neuronas para combinar características. Un Dropout del 50% ayuda a prevenir el sobreajuste. Finalmente, una capa densa de salida con activación sigmoide (sigmoid) clasifica la imagen como perro o gato. El modelo resume su arquitectura con model.summary().**
```python
model = models.Sequential()

model.add(Conv2D(32, (3,3), activation= 'relu', input_shape=(150,150,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
```
**Este código utiliza ImageDataGenerator de Keras para preprocesar y aumentar datos en un problema de clasificación binaria de imágenes (perros vs. gatos). Para el entrenamiento, train_datagen normaliza las imágenes (escalando los píxeles a [0, 1]) y aplica transformaciones aleatorias (rotación, desplazamiento, zoom, etc.) para aumentar la diversidad del dataset y evitar el sobreajuste. Para la validación, test_datagen solo normaliza las imágenes sin transformaciones. Luego, flow_from_directory carga las imágenes desde directorios organizados por clase, las redimensiona a 150x150 píxeles, y genera lotes de 32 imágenes con etiquetas binarias (0 o 1). Los generadores resultantes (train_generator y validation_generator) se usan para entrenar y validar el modelo, proporcionando datos listos para la red neuronal**

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('../input/cnn-data-sources/cats_and_dogs/train',
                                 target_size=(150,150),
                                 batch_size=32,
                                 class_mode='binary'
                                 )

validation_generator = test_datagen.flow_from_directory('../input/cnn-data-sources/cats_and_dogs/validation',
                                 target_size=(150,150),
                                 batch_size=32,
                                 class_mode='binary'
                                 )
```
**Primero, se configura un callback ModelCheckpoint para guardar el mejor modelo durante el entrenamiento (basado en la precisión de validación val_accuracy) en un archivo llamado modelo_perros_gatos.hdf5. Luego, se compila el modelo especificando la función de pérdida (binary_crossentropy para problemas binarios), el optimizador (Adam) y la métrica a monitorear (accuracy). Finalmente, se entrena el modelo con model.fit, utilizando los generadores de datos (train_generator y validation_generator), definiendo el número de pasos por época y épocas totales, y aplicando el callback para guardar el mejor modelo automáticamente.**
```python
checkpoint = ModelCheckpoint('modelo_perros_gatos.hdf5',monitor='val_accuracy', verbose= 1, save_best_only=True)
model.compile(loss='binary_crossentropy', optimizer =optimizers.Adam(),
             metrics=['accuracy'])
hist = model.fit(train_generator, steps_per_epoch=2000//32,
                epochs=100,
                validation_data=validation_generator,
                 validation_steps= 1000//32,
                 callbacks=[checkpoint])
```
**Para graficar la precisión (accuracy)del modelo durante el entrenamiento y la validación. La línea `plt.plot(hist.history['accuracy'])` muestra la precisión en el conjunto de entrenamiento, mientras que `plt.plot(hist.history['val_accuracy'])` muestra la precisión en el conjunto de validación. La función `plt.legend()` añade una leyenda para distinguir ambas curvas, y `plt.show()` despliega la gráfica. Esto permite visualizar cómo evoluciona el rendimiento del modelo a lo largo de las épocas, ayudando a identificar si hay sobreajuste (overfitting) o si el modelo está aprendiendo adecuadamente.**
```python
plt.plot(hist.history['accuracy'], label = 'Train')
plt.plot(hist.history['val_accuracy'], label = 'Val')
plt.legend()
plt.show()
```

El resultado es:

![Resultado](https://github.com/jubarreras/IA-guia-5/blob/main/Captura%20de%20pantalla%202025-03-05%20233930.png)
