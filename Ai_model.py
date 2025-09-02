import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import cv2  # OpenCV para processamento de imagens (ex: visualização)
import matplotlib.pyplot as plt
import os

# Configurações
IMG_SIZE = 48  # Tamanho das imagens no FER2013
BATCH_SIZE = 64
EPOCHS = 20  # Ajuste conforme sua máquina
NUM_CLASSES = 7  # Emoções: angry, disgust, fear, happy, neutral, sad, surprise

# Caminhos do dataset (ajuste para o seu diretório)
train_dir = 'images/train/'  # Pasta com subpastas de emoções
test_dir = 'images/test/'    # Pasta de teste

# Verifica se as pastas existem
if not os.path.exists(train_dir):
    raise ValueError("Baixe e extraia o dataset FER2013 em 'fer2013/'")

# Data Augmentation e Geradores (usando Keras para carregar imagens)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Novo: varia brilho
    channel_shift_range=0.2,      # Novo: shift em canais (mesmo grayscale)
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',  # Imagens em cinza
    batch_size=BATCH_SIZE,
    class_mode='categorical'  # One-hot encoding para 7 classes
)

test_generator = train_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# Construção do Modelo CNN com Keras
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),  # Novo
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),  # Novo
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),  # Novo
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),  # Nova camada
    BatchNormalization(),  # Novo
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),  # Aumente de 128 para 256
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
# Compilação
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()  # Mostra a arquitetura

# Treinamento
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,  # ~448
    epochs=50,  # Aumente para mais aprendizado
    validation_data=test_generator,
    validation_steps=test_generator.samples // BATCH_SIZE  # ~112
)

# Salva o modelo treinado
model.save('emotion_model.h5')
print("Modelo salvo como 'emotion_model.h5'")

# Plot de Acurácia e Loss (opcional)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()