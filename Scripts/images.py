from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#Inicializar la CNN
classifier = Sequential()
#Convolucion

#Vamos a ocupar 32 filtros para detectar rasgos, el 3x3 definimos el tama;o
#metodo add sirve apara a;adir una capa, en este caso la capa de convolution, no dense que se utiliza cuando se hace una rna mas generica
classifier.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(64,64,3), activation="relu")) #Aqui agregamos una capa de convolucion
#Es importante ver la documentacion
#El formato o tamano de las imagenes input_shape, para especificar si los datos no vienen cuadrados, para que todas tengan el mismo formato
#input_shape(255,255,1) en escala de grises
#input_shape(128,128,3) en colores todo esto se puede modificar(es como el canal de color) (primero numero de filas, numero de columnas, numero de canales de color)

#Paso 2 vamos con el maxpooling (reducimos el tama;o de la imagen)
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Flattening
classifier.add(Flatten())
#capa full conectada
classifier.add(Dense(units= 128,activation = "relu")) #Esto es la capa oculta
classifier.add(Dense(units = 1,activation="sigmoid")) #Esta es la capa de salida, como queremos en base a probabilidades ocupamos sigmoide (funciones de softmax o entropia cruzada en caso de que se ocupe mas de una categoira)
#En este caso, es uno porque estamos usando una clasificacion binaria

#Aqui estamos compilando la red neuronal
#En este caso no estamos usando el gradiente grecedentiente sino el adam,
#en loss es la funcion de perdida, en este caso estamos usando binary_crossen q es especial para clasificacion bianria, en caso de que hayan mas se utiliza otra
#lo demas es la metrica de presicion
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Ajustamos la CNN  a las imagenes a entrenar

from keras.preprocessing.image import ImageDataGenerator #Esto es para limpiar las imagenes y evitar el sobre ajuste

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/ignac/OneDrive/Escritorio/Ignacia/UTEM/Machine Learning/Coloracion/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32, #Lotes de 32 imagenes
                                                 class_mode = 'binary') #Yo se que tengo dos clases

test_set = test_datagen.flow_from_directory('C:/Users/ignac/OneDrive/Escritorio/Ignacia/UTEM/Machine Learning/Coloracion/test_set',
                                            target_size = (64, 64),#tama;o de carga
                                            batch_size = 32,#tama;o de lotes de imagenes
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = len(training_set),#cuantas muestras tiene que tomar en un epoch completo
                         epochs = 100,#cantidad de epoch q le dedicamos a la fase de entrenamiento
                         validation_data = test_set,
                         validation_steps = len(test_set))

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/ignac/OneDrive/Escritorio/Ignacia/UTEM/Machine Learning/Coloracion/test_set/Banana/banana3.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'banana'
else:
    prediction = 'apple'

print(prediction)