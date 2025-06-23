# entrenar_para_web.py

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# --- PARÁMETROS IGUALES A LOS DE TU RECONOCEDOR ---
FRAMES_PALABRA_ESTATICA = 10
FRAMES_PALABRA_DINAMICA = 40
CARACTERISTICAS_SIMPLIFICADAS = 21 * 3  # Solo usamos posiciones x, y, z

def cargar_y_simplificar_datos(modo):
    """
    Carga los datos originales pero extrae solo las primeras 63 características (posiciones)
    de cada frame para simular los datos que vienen de la web.
    """
    X, y = [], []
    palabras = []
    
    carpeta = f'datos/{modo}'
    if not os.path.exists(carpeta):
        return X, y, palabras
    
    print(f"🔄 Cargando y simplificando datos para el modo: {modo}")
    
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.pkl'):
            palabra = archivo[:-4]
            if palabra not in palabras:
                palabras.append(palabra)
            
            with open(f'{carpeta}/{archivo}', 'rb') as f:
                muestras_originales = pickle.load(f)
            
            for muestra in muestras_originales:
                muestra_simplificada = []
                for frame in muestra:
                    # Extraemos SOLO las primeras 63 características
                    frame_simplificado = frame[:CARACTERISTICAS_SIMPLIFICADAS]
                    muestra_simplificada.append(frame_simplificado)
                X.append(muestra_simplificada)
                y.append(palabra)
    
    return np.array(X), np.array(y), palabras

def crear_y_entrenar_modelo(modo, X, y):
    """Crea, compila y entrena un modelo para el modo especificado."""
    if len(X) == 0:
        print(f"❌ No hay datos para entrenar en modo {modo}")
        return

    if len(np.unique(y)) < 2:
        print(f"❌ Se necesitan al menos 2 señas diferentes para entrenar el modo {modo}")
        return

    print(f"\n🧠 Preparando entrenamiento del modelo web: {modo.upper()}")
    print(f"   - Muestras: {len(X)}")
    print(f"   - Clases: {len(np.unique(y))}")
    print(f"   - Forma de los datos (X): {X.shape}")

    # Codificar etiquetas
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Definir modelo
    num_clases = len(np.unique(y))
    frames_necesarios = FRAMES_PALABRA_ESTATICA if modo == 'estatico' else FRAMES_PALABRA_DINAMICA
    
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(frames_necesarios, CARACTERISTICAS_SIMPLIFICADAS)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_clases, activation='softmax')
    ])
    
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    modelo.summary()
    
    print("\n▶️  Iniciando entrenamiento...")
    modelo.fit(X, y_encoded, epochs=80, batch_size=16, validation_split=0.2)
    
    # Guardar modelo y encoder con nuevo nombre
    ruta_modelo = f'modelos/modelo_{modo}_web.h5'
    ruta_encoder = f'modelos/encoder_{modo}_web.pkl'
    
    modelo.save(ruta_modelo)
    with open(ruta_encoder, 'wb') as f:
        pickle.dump(encoder, f)
        
    print(f"✅ ¡Modelo guardado en {ruta_modelo} y {ruta_encoder}!")

if __name__ == '__main__':
    # Entrenar modelo estático
    X_est, y_est, _ = cargar_y_simplificar_datos('estatico')
    crear_y_entrenar_modelo('estatico', X_est, y_est)
    
    # Entrenar modelo dinámico
    X_din, y_din, _ = cargar_y_simplificar_datos('dinamico')
    crear_y_entrenar_modelo('dinamico', X_din, y_din)
    
    print("\n🎉 Proceso de re-entrenamiento para la web completado.")