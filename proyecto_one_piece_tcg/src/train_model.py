import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

# Ruta a los datos de entrenamiento
train_dir = 'data/train'

# Preprocesamiento de imágenes
image_size = (150, 150)  # Redimensionar las imágenes a 150x150
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  # Para clasificación multiclase
)

# Construcción del modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Número de clases
])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint para guardar el mejor modelo
checkpoint = ModelCheckpoint('models/one_piece_card_classifier.h5', save_best_only=True)

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    epochs=10,  # Ajusta según tus necesidades
    callbacks=[checkpoint]
)

# Guardar el modelo
model.save('models/one_piece_card_classifier.h5')
