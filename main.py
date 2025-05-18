
# Parâmetros:

MODEL_PATH = "saved_model/"
MODEL_NAME = "modelo1"

MODEL_DIR = f"{MODEL_PATH}{MODEL_NAME}.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32



# Carregar datasets-template existentes no TensorFlow Datasets

import tensorflow_datasets as tfds

(ds_train, ds_val), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

# Começar a preparar os dados

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import random
import pickle
import pandas as pd

tf.config.run_functions_eagerly(True)


def preprocess_image(image, label): # redimensionando as imagens e normalizando os valores dos pixels
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

ds_train = ds_train.map(preprocess_image).shuffle(1000).batch(BATCH_SIZE).prefetch(1)
ds_val = ds_val.map(preprocess_image).batch(BATCH_SIZE).prefetch(1)
val_list = list(ds_val)


model = None
history = None

from tensorflow.python.client import device_lib
print("Devices Detected:")
print("=====================================")
print(device_lib.list_local_devices())

# Começar a programar o modelo
def criar_modelo():
    model = models.Sequential([
        
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)), # aplica 32 filtros de 3x3 na imagem 128x128x3
        layers.MaxPooling2D(2, 2), # 128x128 => 64x64

        layers.Conv2D(64, (3, 3), activation='relu'), # aplica 64 filtros de 3x3 na imagem 64x64
        layers.MaxPooling2D(2, 2), # 64x64 => 32x32

        layers.Conv2D(128, (3, 3), activation='relu'), # aplica 128 filtros de 3x3 na imagem 32x32
        layers.MaxPooling2D(2, 2), # 32x32 => 16x16

        layers.Flatten(), # Converte a imagem 16x16x128 para um vetor 32768
        
        layers.Dense(128, activation='relu'), # 128 neurônios vão processar o vetor
        
        layers.Dense(1, activation='sigmoid') # Saída binária (0 ou 1)
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_DIR)
    print(f"✅ Modelo salvo em: {MODEL_DIR}")


    print(model.summary())
    return model


def carregar_modelo():
    if os.path.exists(MODEL_DIR):
        print("✅ Carregando modelo salvo...")
        model = tf.keras.models.load_model(MODEL_DIR, compile=True)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        print("🔁 Criando novo modelo...")
        model = criar_modelo()
    return model



def treinar(model,epochs=5,path=MODEL_DIR):
    if model is None:
        print("❌ Não há modelo carregado")
        return
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs
    )
    print("✅ Treinamento concluído")
    model.save(path)
    print(f"✅ Modelo salvo em: {path}")
    
    salvar_history(history)
    return history


def salvar_history(history, path=f'{MODEL_PATH}{MODEL_NAME}.pkl'):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✅ Histórico salvo em: {path}")

def carregar_history(path=f'{MODEL_PATH}{MODEL_NAME}.pkl'):
    if not os.path.exists(path):
        print(f"❌ Arquivo não encontrado: {path}")
        return None
    with open(path, 'rb') as f:
        history_dict = pickle.load(f)
    print(f"🔁 Histórico carregado de: {path}")
    return history_dict

def visualizar_history(history):
    if type(history) is not dict:
        history = history.history
    df=pd.DataFrame(history.history)
    df.plot(y=['accuracy', 'val_accuracy'], title='Accuracy')
    df.plot(y=['loss', 'val_loss'], title='Loss')
    plt.show()


# deprecated
def _plot_history(history):
    plt.plot(history['accuracy'], label='train acc')
    plt.plot(history['val_accuracy'], label='val acc')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()



def test_image():
    idx = random.randint(0, len(val_list) - 1)
    img_batch, label_batch = val_list[idx]
    img = img_batch[0].numpy()
    label = label_batch[0].numpy()
    
    pred = model.predict(tf.expand_dims(img, axis=0))[0][0]
    pred_label = 1 if pred > 0.5 else 0
    print(f"Predição: {'Dog 🐶' if pred_label == 1 else 'Cat 🐱'} ({pred:.2f})")
    print("✅" if pred_label == label else "❌")
    
    plt.imshow(img)
    plt.title("Imagem: " + ("Dog 🐶" if label == 1 else "Cat 🐱"))
    plt.axis('off')
    plt.show()
    
    
    
    

import sys
if __name__ == "__main__":
    if sys.flags.interactive:
        print("Executando em um ambiente interativo")
    else:
        model = carregar_modelo()   
        history = treinar(model, epochs=5)
        visualizar_history(history)
        test_image()
        test_image()
        test_image()
