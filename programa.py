# vae_launcher.py

import warnings
warnings.filterwarnings("ignore")  # Deshabilitar todos los warnings

from VAE import VAE, open_file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pickletools
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

def load_model(hp_path, weights_path):
    print("Cargando hiperparámetros...")
    hp = open_file(hp_path)
    print("Construyendo modelo VAE...")
    model = VAE(
        encoder_dims=hp['encoder'],
        latent_dim=hp['latent'],
        decoder_dims=hp['decoder'],
        classifier_dims=hp['classifier'],
        output_weights=hp['loss_weights']
    )
    model.build()
    print("Cargando pesos entrenados...")
    model.full_vae.load_weights(weights_path)
    return model

def evaluate(model, x_test, y_test, class_names):
    print("Haciendo predicciones...")

    _, predictions = model.full_vae.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    if y_test.ndim > 1:  # Si y_test está codificado one-hot
        true_classes = np.argmax(y_test, axis=1)
    else:
        true_classes = y_test

    # Verificar que el número de muestras es consistente
    print(f"Predicted classes: {predicted_classes.shape}")
    print(f"True classes: {true_classes.shape}")
    
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"\nAccuracy: {accuracy:.4f}")

    cm = confusion_matrix(true_classes, predicted_classes)

    # Mostrar matriz de confusión
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión: Clasificación de Tumores Primarios TCGA')
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    # Rutas de los archivos
    hp_file = 'cancernet_hyperparams.pickle'
    weights_file = '34_class_CancerNet_weights.h5'
    x_test_file = 'sample_x.pickle'
    y_test_file = 'sample_y.pickle'

    # Lista de clases (orden alfabético)
    class_names = sorted([
        'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
        'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO',
        'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT',
        'THCA', 'THYM', 'UCEC', 'UCS', 'UVM', 'NORM'
    ])

    # Cargar modelo
    model = load_model(hp_file, weights_file)

    # Cargar datos
    print("Cargando datos de prueba...")
    #x_test = pickle.load(open("sample_x.pickle", "rb"))
    #y_test = pickle.load(open("sample_y.pickle", "rb"))
    x_test = open_file(x_test_file)
    y_test = open_file(y_test_file)
    
    y_test = np.array(y_test)  # Convertir a NumPy
    print(f"y_test shape después de conversión: {y_test.shape}")  # (200, 34)

    # con esto no iba, solo es si y_test son etiquetas enteras
    # y_test = to_categorical(np.argmax(y_test, axis=1), num_classes=34)   
    #y_test = to_categorical(y_test, num_classes=34)
    

    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")



    # Evaluar modelo
    evaluate(model, x_test, y_test, class_names)

if __name__ == "__main__":
    main()
