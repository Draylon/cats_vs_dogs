
# Parâmetros:

MODEL_PATH = "saved_model/"
MODEL_NAME = "modelo"

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
from tensorflow.keras import layers, models, Model
import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.rcParams['font.family'] = ['DejaVu Sans','Noto Sans', 'Noto Sans Symbols2', 'DejaVu Sans', 'Arial', 'sans-serif','DejaVu Sans']

import os
import random
import pickle
import pandas as pd

tf.config.run_functions_eagerly(True)

import time
import csv
from tensorflow.keras.callbacks import Callback

class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.train_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)

    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start_time
        print(f"Treinamento completo em {total_time:.2f}s.")


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
        
        # Primeira camada - dados de entrada
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Segunda camada - controle do crescimento de parâmetros
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.35),
        
        # Terceira camada - reduzindo complexidade
        layers.Conv2D(96, (3, 3), activation='relu'),  # Reduzido de 128 para 96
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),
        
        # Camada adicional para melhorar extração de features
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        
        # layers.Flatten(),  # Flatten para converter a saída 2D em 1D
        layers.GlobalAveragePooling2D(),  # Substitui Flatten
        
        # Camadas densas mais controladas
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),  # Reduzido de 128 para 64
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
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
    
    time_callback = TimeHistory()
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[time_callback]
    )
    print("✅ Treinamento concluído")
    model.save(path)
    print(f"✅ Modelo salvo em: {path}")
    
    salvar_history(history,time_callback)
    return history


def salvar_history(history,time_callback=None, path=f'{MODEL_PATH}{MODEL_NAME}.pkl'):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✅ Histórico salvo em: {path}")
    
    if time_callback is not None:
        df = pd.DataFrame()
        df['epoch_time_seconds'] = time_callback.epoch_times
        df.to_csv(f"{MODEL_PATH}{MODEL_NAME}.time.csv", index_label='epoch')

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
    df=pd.DataFrame(history)
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


def batch_test():
    pred_c = 0
    for i in range(50):
        idx = random.randint(0, len(val_list) - 1)
        img_batch, label_batch = val_list[idx]
        img = img_batch[0].numpy()
        label = label_batch[0].numpy()
        
        pred = model.predict(tf.expand_dims(img, axis=0))[0][0]
        pred_label = 1 if pred > 0.5 else 0
        if pred_label == label:
            pred_c += 1
    print(f"Predições corretas: {pred_c} de 50 ({pred_c / 50 * 100:.2f}%)")

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
    plt.title("Imagem: " + ("Dog" if label == 1 else "Cat 🐱") + "\nPredição:" +("Dog" if pred_label == 1 else "Cat"))
    plt.axis('off')
    plt.show()



def test_custom(img_path):
    model = carregar_modelo()
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0) / 255.0
    pred = model.predict(img)[0][0]
    pred_label = 1 if pred > 0.5 else 0
    print(f"Predição: {'Dog 🐶' if pred_label == 1 else 'Cat 🐱'} ({pred:.2f})")
    
    plt.imshow(img[0])
    plt.title("Imagem: " + ("Dog" if pred_label == 1 else "Cat"))
    plt.axis('off')
    plt.show()




def visualizar_modelo():
    #img_batch, _ = next(iter(ds_val))
    #img = img_batch[0:1]
    img = random.choice(val_list)[0]
    layer_outputs = [layer.output for layer in model.layers if 'conv2d' in layer.name]
    activation_model = Model(model.inputs, layer_outputs)
    
    activations = activation_model.predict(img)
    
    first_layer_activation = activations[0]
    n_filters = first_layer_activation.shape[-1]
    
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i in range(n_filters):
        ax = axes[i//8, i%8]
        feature_map = first_layer_activation[0, :, :, i]
        ax.imshow(feature_map, cmap='gray')
        ax.axis('off')
    
    plt.suptitle('Feature Maps da primeira Conv2D')
    plt.show()



import sys
if __name__ == "__main__":
    if sys.flags.interactive:
        print("Executando em um ambiente interativo")
    else:
        model = carregar_modelo()
        history = treinar(model, epochs=10)
        visualizar_history(history)






import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def gradcam_visualization(model, img, last_conv_layer_name, alpha=0.4):
    """
    Aplica Grad-CAM em uma imagem para um modelo de classificação binária.
    
    Parâmetros:
    - model: modelo Keras treinado.
    - img: imagem de entrada (array numpy, shape [H, W, 3], valores [0, 1]).
    - last_conv_layer_name: nome da última camada convolucional do modelo.
    - alpha: intensidade da sobreposição do heatmap.
    
    Exibe: imagem original com Grad-CAM sobreposto.
    """
    
    # Prepara a imagem
    img_array = np.expand_dims(img, axis=0)
    
    # Modelo intermediário: camada conv e saída
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Calcula gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normaliza o heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Redimensiona o heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Prepara imagem original
    img_uint8 = np.uint8(255 * img)
    
    # Superimposição
    superimposed_img = heatmap_colored * alpha + img_uint8
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img.astype('uint8'))
    plt.axis('off')
    plt.title('Grad-CAM Visualization')
    plt.show()



# Exemplo de uso após o treinamento do modelo
for img, label in train.take(1):
    sample_img = img[0].numpy()
    break

gradcam_visualization(model, sample_img, last_conv_layer_name='conv2d_2')
