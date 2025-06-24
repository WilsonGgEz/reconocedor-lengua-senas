#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RECONOCEDOR DE LENGUA DE SEÑAS - MODO DUAL SIMPLIFICADO
=======================================================
Estático: Letras, números (posición instantánea)
Dinámico: Palabras complejas (posición + movimiento)
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime
#import pyttsx3
import math

class LenguaSeñasDual:
    def __init__(self):
        # Configuración
        self.FRAMES_PALABRA_DINAMICA = 40
        self.FRAMES_PALABRA_ESTATICA = 10
        self.PUNTOS_MANO = 21
        
        # Características según el modo
        self.CARACTERISTICAS_ESTATICAS = self.PUNTOS_MANO * 3
        self.CARACTERISTICAS_DINAMICAS = (
            self.PUNTOS_MANO * 3 +      # Posiciones
            self.PUNTOS_MANO * 3 +      # Velocidades
            self.PUNTOS_MANO * 3 +      # Aceleraciones
            10                          # Características globales
        )
        
        # Umbrales de confianza simplificados
        self.UMBRAL_CONFIANZA_ESTATICO = 0.70
        self.UMBRAL_CONFIANZA_DINAMICO = 0.65
        
        # Modos disponibles
        self.MODO_ESTATICO = "estatico"
        self.MODO_DINAMICO = "dinamico"
        self.modo_actual = self.MODO_ESTATICO
        
        # Crear carpetas
        os.makedirs('datos/estatico', exist_ok=True)
        os.makedirs('datos/dinamico', exist_ok=True)
        os.makedirs('modelos', exist_ok=True)
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Modelos separados
        self.modelo_estatico = None
        self.modelo_dinamico = None
        self.encoder_estatico = LabelEncoder()
        self.encoder_dinamico = LabelEncoder()
        
        # Estados de los modelos
        self.modelo_estatico_entrenado = False
        self.modelo_dinamico_entrenado = False
        
        # Text-to-speech
        #try:
        #    self.tts = pyttsx3.init()
        #    self.tts.setProperty('rate', 150)
        #    self.tts.setProperty('volume', 0.9)
        #    self.tts_disponible = True
        #except:
        #    print("⚠️ Text-to-speech no disponible")
        #    self.tts_disponible = False
        
        # Buffers
        self.buffer_frames_estatico = []
        self.buffer_frames_dinamico = []
        self.buffer_posiciones = []
        self.buffer_velocidades = []
        self.buffer_aceleraciones = []
        self.texto_completo = []
        self.ultima_palabra_reconocida = ""
        
        # Variables de debug mejoradas
        self.debug_mode = True
        self.debug_info = {
            'estatico': {'prediccion': '', 'confianza': 0.0},
            'dinamico': {'prediccion': '', 'confianza': 0.0},
            'mano_detectada': False,
            'velocidad_promedio': 0.0,
            'energia_movimiento': 0.0,
            'hay_movimiento': False
        }
        
        # Variables de reconocimiento automático
        self.auto_reconocimiento = True
        self.contador_estable = 0
        self.FRAMES_ESTABLE_REQUERIDOS = 8
        self.ultima_prediccion_estatica = ""
        
        # Cargar modelos
        self._cargar_modelos()
        
        print("✅ Sistema dual simplificado inicializado")
        print(f"📊 Modo estático: {self.CARACTERISTICAS_ESTATICAS} características")
        print(f"📊 Modo dinámico: {self.CARACTERISTICAS_DINAMICAS} características")
    
    def alternar_modo(self):
        """Alternar entre modo estático y dinámico"""
        if self.modo_actual == self.MODO_ESTATICO:
            self.modo_actual = self.MODO_DINAMICO
            print("🔄 Cambiado a MODO DINÁMICO (palabras con movimiento)")
        else:
            self.modo_actual = self.MODO_ESTATICO
            print("🔄 Cambiado a MODO ESTÁTICO (letras, números)")
        
        # Limpiar buffers al cambiar modo
        self.buffer_frames_estatico = []
        self.buffer_frames_dinamico = []
        self.buffer_posiciones = []
        self.buffer_velocidades = []
        self.buffer_aceleraciones = []
    
    def extraer_caracteristicas_estaticas(self, frame):
        """Extraer solo posiciones para señas estáticas"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = self.hands.process(rgb)
        
        posiciones = []
        mano_detectada = False
        
        if resultado.multi_hand_landmarks:
            for landmarks in resultado.multi_hand_landmarks:
                mano_detectada = True
                self.mp_draw.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Solo posiciones normalizadas
                for punto in landmarks.landmark:
                    posiciones.extend([punto.x, punto.y, punto.z])
        
        # Si no hay mano, llenar con ceros
        if not posiciones:
            posiciones = [0.0] * (self.PUNTOS_MANO * 3)
        
        self.debug_info['mano_detectada'] = mano_detectada
        return posiciones, frame, mano_detectada
    
    def extraer_caracteristicas_dinamicas(self, frame):
        """Extraer posiciones + movimiento para señas dinámicas"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = self.hands.process(rgb)
        
        posiciones_actuales = []
        mano_detectada = False
        
        if resultado.multi_hand_landmarks:
            for landmarks in resultado.multi_hand_landmarks:
                mano_detectada = True
                self.mp_draw.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extraer posiciones normalizadas
                for punto in landmarks.landmark:
                    posiciones_actuales.extend([punto.x, punto.y, punto.z])
        
        if not posiciones_actuales:
            posiciones_actuales = [0.0] * (self.PUNTOS_MANO * 3)
        
        # Agregar al buffer de posiciones
        self.buffer_posiciones.append(posiciones_actuales)
        if len(self.buffer_posiciones) > 3:
            self.buffer_posiciones.pop(0)
        
        # Calcular características completas con movimiento
        caracteristicas_completas = self._calcular_caracteristicas_movimiento(posiciones_actuales)
        
        self.debug_info['mano_detectada'] = mano_detectada
        return caracteristicas_completas, frame, mano_detectada
    
    def _calcular_caracteristicas_movimiento(self, posiciones_actuales):
        """Calcular velocidades, aceleraciones y características globales"""
        caracteristicas = posiciones_actuales.copy()
        
        # VELOCIDADES
        if len(self.buffer_posiciones) >= 2:
            pos_anterior = self.buffer_posiciones[-2]
            velocidades = []
            for i in range(len(posiciones_actuales)):
                vel = posiciones_actuales[i] - pos_anterior[i]
                velocidades.append(vel)
            caracteristicas.extend(velocidades)
            
            self.buffer_velocidades.append(velocidades)
            if len(self.buffer_velocidades) > 2:
                self.buffer_velocidades.pop(0)
                
            # Actualizar debug
            self.debug_info['velocidad_promedio'] = np.mean([abs(v) for v in velocidades])
            self.debug_info['hay_movimiento'] = any(abs(v) > 0.01 for v in velocidades[:6])
        else:
            caracteristicas.extend([0.0] * (self.PUNTOS_MANO * 3))
        
        # ACELERACIONES
        if len(self.buffer_velocidades) >= 2:
            vel_anterior = self.buffer_velocidades[-2]
            vel_actual = self.buffer_velocidades[-1]
            aceleraciones = []
            for i in range(len(vel_actual)):
                acel = vel_actual[i] - vel_anterior[i]
                aceleraciones.append(acel)
            caracteristicas.extend(aceleraciones)
        else:
            caracteristicas.extend([0.0] * (self.PUNTOS_MANO * 3))
        
        # CARACTERÍSTICAS GLOBALES
        caracteristicas_globales = self._calcular_caracteristicas_globales()
        caracteristicas.extend(caracteristicas_globales)
        
        # Actualizar debug
        if len(caracteristicas_globales) > 0:
            self.debug_info['energia_movimiento'] = caracteristicas_globales[-1]
        
        return caracteristicas
    
    def _calcular_caracteristicas_globales(self):
        """Características globales de movimiento"""
        if len(self.buffer_posiciones) < 2:
            return [0.0] * 10
        
        pos_actual = np.array(self.buffer_posiciones[-1]).reshape(-1, 3)
        pos_anterior = np.array(self.buffer_posiciones[-2]).reshape(-1, 3)
        
        caracteristicas = []
        
        # Velocidad total promedio
        velocidades = pos_actual - pos_anterior
        velocidad_total = np.mean(np.linalg.norm(velocidades, axis=1))
        caracteristicas.append(velocidad_total)
        
        # Velocidad máxima
        velocidad_maxima = np.max(np.linalg.norm(velocidades, axis=1))
        caracteristicas.append(velocidad_maxima)
        
        # Dirección de movimiento dominante
        movimiento_x = np.mean(velocidades[:, 0])
        movimiento_y = np.mean(velocidades[:, 1])
        caracteristicas.extend([movimiento_x, movimiento_y])
        
        # Área de la mano
        if np.any(pos_actual):
            area = np.std(pos_actual[:, 0]) * np.std(pos_actual[:, 1])
        else:
            area = 0.0
        caracteristicas.append(area)
        
        # Centro de masa
        centro_x = np.mean(pos_actual[:, 0])
        centro_y = np.mean(pos_actual[:, 1])
        caracteristicas.extend([centro_x, centro_y])
        
        # Movimiento del centro de masa
        if len(self.buffer_posiciones) >= 2:
            pos_ant = np.array(self.buffer_posiciones[-2]).reshape(-1, 3)
            centro_ant_x = np.mean(pos_ant[:, 0])
            centro_ant_y = np.mean(pos_ant[:, 1])
            mov_centro_x = centro_x - centro_ant_x
            mov_centro_y = centro_y - centro_ant_y
        else:
            mov_centro_x = mov_centro_y = 0.0
        caracteristicas.extend([mov_centro_x, mov_centro_y])
        
        # Energía de movimiento
        energia = np.sum(velocidades ** 2)
        caracteristicas.append(energia)
        
        return caracteristicas
    
    def modo_entrenamiento(self):
        """Modo entrenamiento con selección de tipo"""
        print("\n🎯 MODO ENTRENAMIENTO")
        print("=" * 30)
        
        # Seleccionar tipo de seña
        print("🔄 TIPO DE SEÑA:")
        print("1. 📍 ESTÁTICA (letras, números, símbolos)")
        print("2. 🌊 DINÁMICA (palabras con movimiento)")
        
        tipo = input("Selecciona tipo (1/2): ").strip()
        
        if tipo == "1":
            self.modo_actual = self.MODO_ESTATICO
            print("📍 Modo estático seleccionado")
        elif tipo == "2":
            self.modo_actual = self.MODO_DINAMICO
            print("🌊 Modo dinámico seleccionado")
        else:
            print("❌ Selección inválida")
            return
        
        # Solicitar palabra
        palabra = input(f"🏷️ {'Letra/número' if self.modo_actual == self.MODO_ESTATICO else 'Palabra'} a entrenar: ").strip().upper()
        
        if not palabra:
            print("❌ Entrada inválida")
            return
        
        cantidad = int(input(f"📊 Número de muestras (recomendado: {30 if self.modo_actual == self.MODO_ESTATICO else 25}): ") or 
                           (30 if self.modo_actual == self.MODO_ESTATICO else 25))
        
        self._entrenar_seña(palabra, cantidad)
    
    def _entrenar_seña(self, palabra, cantidad):
        """Entrenar una seña específica"""
        frames_necesarios = (self.FRAMES_PALABRA_ESTATICA if self.modo_actual == self.MODO_ESTATICO 
                           else self.FRAMES_PALABRA_DINAMICA)
        
        print(f"\n🎬 Entrenando {self.modo_actual}: {palabra}")
        print("📋 INSTRUCCIONES:")
        if self.modo_actual == self.MODO_ESTATICO:
            print(f"   • Haz la seña '{palabra}' y manténla estable")
            print("   • No necesitas movimiento especial")
        else:
            print(f"   • Haz la seña completa '{palabra}' con su movimiento natural")
            print("   • Incluye el movimiento característico")
        
        print("   • Presiona ESPACIO cuando estés listo")
        print("   • Presiona Q para salir")
        
        cap = cv2.VideoCapture(0)
        muestras_guardadas = 0
        grabando = False
        frames_actuales = []
        
        # Limpiar buffers
        self._limpiar_buffers()
        
        while muestras_guardadas < cantidad:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extraer características según el modo
            if self.modo_actual == self.MODO_ESTATICO:
                caracteristicas, frame, mano_detectada = self.extraer_caracteristicas_estaticas(frame)
            else:
                caracteristicas, frame, mano_detectada = self.extraer_caracteristicas_dinamicas(frame)
            
            if grabando:
                frames_actuales.append(caracteristicas)
                
                if len(frames_actuales) >= frames_necesarios:
                    self._guardar_muestra(palabra, frames_actuales, self.modo_actual)
                    muestras_guardadas += 1
                    frames_actuales = []
                    grabando = False
                    self._limpiar_buffers()
                    print(f"✅ Muestra {muestras_guardadas}/{cantidad} guardada")
            
            # Dibujar interfaz
            self._dibujar_interfaz_entrenamiento(
                frame, palabra, muestras_guardadas, cantidad, 
                len(frames_actuales), frames_necesarios, grabando, mano_detectada
            )
            
            cv2.imshow('ENTRENAMIENTO - Lengua de Señas', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not grabando and mano_detectada:
                grabando = True
                frames_actuales = []
                print(f"🔴 Grabando muestra {muestras_guardadas + 1}...")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if muestras_guardadas > 0:
            print(f"\n🎉 ¡Entrenamiento completado!")
            print(f"📊 {muestras_guardadas} muestras guardadas para '{palabra}' ({self.modo_actual})")
            
            entrenar_ahora = input(f"\n🧠 ¿Entrenar el modelo {self.modo_actual} ahora? (s/N): ").strip().lower()
            if entrenar_ahora in ['s', 'si', 'yes', 'y']:
                self._entrenar_modelo(self.modo_actual)
    
    def _dibujar_interfaz_entrenamiento(self, frame, palabra, guardadas, total, frames_actuales, frames_max, grabando, mano_detectada):
        """Interfaz de entrenamiento mejorada"""
        h, w = frame.shape[:2]
        
        # Overlay principal
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Información principal
        modo_texto = "ESTÁTICA" if self.modo_actual == self.MODO_ESTATICO else "DINÁMICA"
        
        cv2.putText(frame, f"ENTRENANDO {modo_texto}: {palabra}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        estado = "🔴 GRABANDO" if grabando else "⏸️ ESPERANDO"
        color_estado = (0, 255, 0) if grabando else (0, 0, 255)
        cv2.putText(frame, estado, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_estado, 2)
        
        cv2.putText(frame, f"Progreso: {guardadas}/{total}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frames: {frames_actuales}/{frames_max}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Estado de la mano
        mano_estado = "✅ MANO DETECTADA" if mano_detectada else "❌ SIN MANO"
        color_mano = (0, 255, 0) if mano_detectada else (0, 0, 255)
        cv2.putText(frame, mano_estado, (w - 250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_mano, 2)
        
        # Controles
        cv2.putText(frame, "ESPACIO: Grabar | Q: Salir", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Barra de progreso
        if grabando:
            progreso = frames_actuales / frames_max
            ancho_barra = 400
            cv2.rectangle(frame, (w - ancho_barra - 10, 60), 
                         (w - 10, 80), (50, 50, 50), -1)
            cv2.rectangle(frame, (w - ancho_barra - 10, 60), 
                         (w - ancho_barra - 10 + int(ancho_barra * progreso), 80), 
                         (0, 255, 255), -1)
    
    def _limpiar_buffers(self):
        """Limpiar todos los buffers"""
        self.buffer_frames_estatico = []
        self.buffer_frames_dinamico = []
        self.buffer_posiciones = []
        self.buffer_velocidades = []
        self.buffer_aceleraciones = []
    
    def _guardar_muestra(self, palabra, frames, modo):
        """Guardar muestra según el modo"""
        archivo = f"datos/{modo}/{palabra}.pkl"
        
        muestras = []
        if os.path.exists(archivo):
            with open(archivo, 'rb') as f:
                muestras = pickle.load(f)
        
        muestras.append(frames)
        
        with open(archivo, 'wb') as f:
            pickle.dump(muestras, f)
    
    def _entrenar_modelo(self, modo):
        """Entrenar modelo simplificado"""
        print(f"\n🧠 ENTRENANDO MODELO {modo.upper()}...")
        print("=" * 40)
        
        X, y, palabras = self._cargar_datos(modo)
        
        if len(X) == 0:
            print(f"❌ No hay datos para entrenar en modo {modo}")
            return False
        
        if len(palabras) < 2:
            print(f"❌ Necesitas al menos 2 {('letras/símbolos' if modo == self.MODO_ESTATICO else 'palabras')} diferentes")
            return False
        
        print(f"📊 Datos cargados:")
        print(f"   • Muestras totales: {len(X)}")
        print(f"   • {('Letras/símbolos' if modo == self.MODO_ESTATICO else 'Palabras')}: {len(palabras)}")
        print(f"   • Vocabulario: {', '.join(palabras)}")
        
        X = np.array(X)
        
        if modo == self.MODO_ESTATICO:
            encoder = self.encoder_estatico
        else:
            encoder = self.encoder_dinamico
            
        y_encoded = encoder.fit_transform(y)
        
        print(f"📏 Forma de datos: {X.shape}")
        
        # Crear y compilar modelo
        if modo == self.MODO_ESTATICO:
            modelo = self._crear_modelo_estatico(len(palabras))
        else:
            modelo = self._crear_modelo_dinamico(len(palabras))
        
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n🎯 Iniciando entrenamiento {modo}...")
        
        epochs = 80 if modo == self.MODO_ESTATICO else 100
        batch_size = 16 if modo == self.MODO_ESTATICO else 8
        
        try:
            history = modelo.fit(
                X, y_encoded,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1
            )
            
            # Guardar modelo
            modelo.save(f'modelos/modelo_{modo}.h5')
            with open(f'modelos/encoder_{modo}.pkl', 'wb') as f:
                pickle.dump(encoder, f)
            
            # Asignar modelo
            if modo == self.MODO_ESTATICO:
                self.modelo_estatico = modelo
                self.modelo_estatico_entrenado = True
            else:
                self.modelo_dinamico = modelo
                self.modelo_dinamico_entrenado = True
            
            print(f"\n✅ ¡Modelo {modo} entrenado!")
            print(f"📈 Precisión final: {history.history['accuracy'][-1]:.2%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            return False
    
    def _crear_modelo_estatico(self, num_clases):
        """Crear modelo para señas estáticas"""
        modelo = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.FRAMES_PALABRA_ESTATICA, self.CARACTERISTICAS_ESTATICAS)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_clases, activation='softmax')
        ])
        
        return modelo
    
    def _crear_modelo_dinamico(self, num_clases):
        """Crear modelo para señas dinámicas"""
        modelo = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.FRAMES_PALABRA_DINAMICA, self.CARACTERISTICAS_DINAMICAS)),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_clases, activation='softmax')
        ])
        
        return modelo
    
    def _cargar_datos(self, modo):
        """Cargar datos según el modo"""
        X, y = [], []
        palabras = []
        
        carpeta = f'datos/{modo}'
        if not os.path.exists(carpeta):
            return X, y, palabras
        
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.pkl'):
                palabra = archivo[:-4]
                palabras.append(palabra)
                
                with open(f'{carpeta}/{archivo}', 'rb') as f:
                    muestras = pickle.load(f)
                
                for muestra in muestras:
                    X.append(muestra)
                    y.append(palabra)
        
        return X, y, palabras
    
    def _cargar_modelos(self):
        """Cargar ambos modelos (versión WEB) con diagnóstico mejorado."""
        print("--- INICIANDO CARGA DE MODELOS ---")
        
        modelo_estatico_path = 'modelos/modelo_estatico.h5'
        encoder_estatico_path = 'modelos/encoder_estatico.pkl'
        modelo_dinamico_path = 'modelos/modelo_dinamico.h5'
        encoder_dinamico_path = 'modelos/encoder_dinamico.pkl'

        # Modelo estático
        print(f"Buscando modelo estático en: '{os.path.abspath(modelo_estatico_path)}'")
        if os.path.exists(modelo_estatico_path):
            try:
                self.modelo_estatico = tf.keras.models.load_model(modelo_estatico_path)
                with open(encoder_estatico_path, 'rb') as f:
                    self.encoder_estatico = pickle.load(f)
                self.modelo_estatico_entrenado = True
                print("✅ Modelo estático (WEB) cargado con éxito.")
            except Exception as e:
                print(f"⚠️ Error al procesar el archivo del modelo estático: {e}")
        else:
            print("❌ ¡ERROR CRÍTICO! No se encontró el archivo del modelo estático.")

        # Modelo dinámico
        print(f"Buscando modelo dinámico en: '{os.path.abspath(modelo_dinamico_path)}'")
        if os.path.exists(modelo_dinamico_path):
            try:
                self.modelo_dinamico = tf.keras.models.load_model(modelo_dinamico_path)
                with open(encoder_dinamico_path, 'rb') as f:
                    self.encoder_dinamico = pickle.load(f)
                self.modelo_dinamico_entrenado = True
                print("✅ Modelo dinámico (WEB) cargado con éxito.")
            except Exception as e:
                print(f"⚠️ Error al procesar el archivo del modelo dinámico: {e}")
        else:
            print("❌ ¡ERROR CRÍTICO! No se encontró el archivo del modelo dinámico.")

        print("--- CARGA DE MODELOS FINALIZADA ---")

    def modo_reconocimiento(self):
        """Reconocimiento simplificado con visualización en webcam"""
        print("\n🎯 MODO RECONOCIMIENTO")
        print("=" * 25)
        
        modelos_disponibles = []
        if self.modelo_estatico_entrenado:
            modelos_disponibles.append("estático")
        if self.modelo_dinamico_entrenado:
            modelos_disponibles.append("dinámico")
        
        if not modelos_disponibles:
            print("❌ No hay modelos entrenados disponibles")
            print("💡 Usa el Modo 1 para entrenar modelos")
            input("⏎ Presiona Enter para continuar...")
            return
        
        print(f"✅ Modelos disponibles: {', '.join(modelos_disponibles)}")
        print("\n📋 CONTROLES:")
        print("   • ESPACIO: Reconocimiento manual")
        print("   • A: Toggle auto-reconocimiento")
        print("   • M: Cambiar modo (estático/dinámico)")
        print("   • +/-: Ajustar umbral de confianza")
        print("   • C: Limpiar texto")
        print("   • G: Guardar texto")
        print("   • Q: Salir")
        
        cap = cv2.VideoCapture(0)
        self._limpiar_buffers()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            try:
                # Extraer características según modo actual
                if self.modo_actual == self.MODO_ESTATICO:
                    caracteristicas, frame, mano_detectada = self.extraer_caracteristicas_estaticas(frame)
                    self._procesar_reconocimiento_estatico(caracteristicas, mano_detectada)
                else:
                    caracteristicas, frame, mano_detectada = self.extraer_caracteristicas_dinamicas(frame)
                    self._procesar_reconocimiento_dinamico(caracteristicas, mano_detectada)
                
                # Dibujar interfaz EN LA WEBCAM
                self._dibujar_interfaz_reconocimiento(frame)
                
            except Exception as e:
                print(f"⚠️ Error: {e}")
                cv2.putText(frame, "ERROR: Reiniciando...", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('RECONOCIMIENTO - Lengua de Señas', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                self._reconocimiento_manual()
            elif key == ord('a'):
                self.auto_reconocimiento = not self.auto_reconocimiento
                print(f"🔄 Auto-reconocimiento: {'ON' if self.auto_reconocimiento else 'OFF'}")
            elif key == ord('m'):
                self.alternar_modo()
                self._limpiar_buffers()
            elif key == ord('+') or key == ord('='):
                if self.modo_actual == self.MODO_ESTATICO:
                    self.UMBRAL_CONFIANZA_ESTATICO = min(0.95, self.UMBRAL_CONFIANZA_ESTATICO + 0.05)
                    print(f"📈 Umbral estático: {self.UMBRAL_CONFIANZA_ESTATICO:.0%}")
                else:
                    self.UMBRAL_CONFIANZA_DINAMICO = min(0.95, self.UMBRAL_CONFIANZA_DINAMICO + 0.05)
                    print(f"📈 Umbral dinámico: {self.UMBRAL_CONFIANZA_DINAMICO:.0%}")
            elif key == ord('-'):
                if self.modo_actual == self.MODO_ESTATICO:
                    self.UMBRAL_CONFIANZA_ESTATICO = max(0.50, self.UMBRAL_CONFIANZA_ESTATICO - 0.05)
                    print(f"📉 Umbral estático: {self.UMBRAL_CONFIANZA_ESTATICO:.0%}")
                else:
                    self.UMBRAL_CONFIANZA_DINAMICO = max(0.50, self.UMBRAL_CONFIANZA_DINAMICO - 0.05)
                    print(f"📉 Umbral dinámico: {self.UMBRAL_CONFIANZA_DINAMICO:.0%}")
            elif key == ord('c'):
                self.texto_completo = []
                self.ultima_palabra_reconocida = ""
                print("🗑️ Texto limpiado")
            elif key == ord('g'):
                self._guardar_texto_final()
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _procesar_reconocimiento_estatico(self, caracteristicas, mano_detectada):
        """Procesar reconocimiento estático"""
        if not self.modelo_estatico_entrenado or not mano_detectada:
            self.debug_info['estatico'] = {'prediccion': '', 'confianza': 0.0}
            return
        
        # Agregar al buffer
        self.buffer_frames_estatico.append(caracteristicas)
        if len(self.buffer_frames_estatico) > self.FRAMES_PALABRA_ESTATICA:
            self.buffer_frames_estatico.pop(0)
        
        # Predecir si tenemos suficientes frames
        if len(self.buffer_frames_estatico) == self.FRAMES_PALABRA_ESTATICA:
            try:
                secuencia = np.expand_dims(self.buffer_frames_estatico, axis=0)
                pred = self.modelo_estatico.predict(secuencia, verbose=0)
                probabilidades = pred[0]
                
                clase_idx = np.argmax(probabilidades)
                confianza = probabilidades[clase_idx]
                prediccion_raw = self.encoder_estatico.inverse_transform([clase_idx])[0]
                
                self.debug_info['estatico'] = {
                    'prediccion': prediccion_raw,
                    'confianza': confianza
                }
                
                # Auto-reconocimiento con umbral
                if self.auto_reconocimiento and confianza >= self.UMBRAL_CONFIANZA_ESTATICO:
                    if prediccion_raw == self.ultima_prediccion_estatica:
                        self.contador_estable += 1
                        if self.contador_estable >= self.FRAMES_ESTABLE_REQUERIDOS:
                            self._procesar_palabra_reconocida(prediccion_raw)
                            self.contador_estable = 0
                            self.buffer_frames_estatico = []
                    else:
                        self.contador_estable = 0
                        self.ultima_prediccion_estatica = prediccion_raw
                
            except Exception as e:
                print(f"⚠️ Error en reconocimiento estático: {e}")
                self.debug_info['estatico'] = {'prediccion': '', 'confianza': 0.0}
    
    def _procesar_reconocimiento_dinamico(self, caracteristicas, mano_detectada):
        """Procesar reconocimiento dinámico"""
        if not self.modelo_dinamico_entrenado or not mano_detectada:
            self.debug_info['dinamico'] = {'prediccion': '', 'confianza': 0.0}
            return
        
        # Agregar al buffer
        self.buffer_frames_dinamico.append(caracteristicas)
        if len(self.buffer_frames_dinamico) > self.FRAMES_PALABRA_DINAMICA:
            self.buffer_frames_dinamico.pop(0)
        
        # Predecir si tenemos suficientes frames
        if len(self.buffer_frames_dinamico) == self.FRAMES_PALABRA_DINAMICA:
            try:
                secuencia = np.expand_dims(self.buffer_frames_dinamico, axis=0)
                pred = self.modelo_dinamico.predict(secuencia, verbose=0)
                probabilidades = pred[0]
                
                clase_idx = np.argmax(probabilidades)
                confianza = probabilidades[clase_idx]
                prediccion_raw = self.encoder_dinamico.inverse_transform([clase_idx])[0]
                
                self.debug_info['dinamico'] = {
                    'prediccion': prediccion_raw,
                    'confianza': confianza
                }
                
            except Exception as e:
                print(f"⚠️ Error en reconocimiento dinámico: {e}")
                self.debug_info['dinamico'] = {'prediccion': '', 'confianza': 0.0}
    
    def _dibujar_interfaz_reconocimiento(self, frame):
        """Interfaz de reconocimiento en la webcam con confianza y texto"""
        h, w = frame.shape[:2]
        
        # Overlay principal
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-120), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        y_pos = 30
        
        # Modo actual
        modo_color = (0, 255, 255) if self.modo_actual == self.MODO_ESTATICO else (255, 0, 255)
        cv2.putText(frame, f"MODO: {self.modo_actual.upper()}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, modo_color, 2)
        y_pos += 40
        
        # Predicción y confianza según el modo actual
        if self.modo_actual == self.MODO_ESTATICO:
            info = self.debug_info['estatico']
            buffer_actual = len(self.buffer_frames_estatico)
            buffer_max = self.FRAMES_PALABRA_ESTATICA
            umbral_actual = self.UMBRAL_CONFIANZA_ESTATICO
        else:
            info = self.debug_info['dinamico']
            buffer_actual = len(self.buffer_frames_dinamico)
            buffer_max = self.FRAMES_PALABRA_DINAMICA
            umbral_actual = self.UMBRAL_CONFIANZA_DINAMICO
        
        # MOSTRAR PREDICCIÓN Y CONFIANZA EN WEBCAM
        if info['prediccion'] and info['confianza'] > 0:
            # Color según confianza
            if info['confianza'] >= umbral_actual:
                color_pred = (0, 255, 0)  # Verde - alta confianza
                status = "✅"
            elif info['confianza'] >= 0.5:
                color_pred = (0, 255, 255)  # Amarillo - media
                status = "⚠️"
            else:
                color_pred = (0, 0, 255)  # Rojo - baja
                status = "❌"
            
            cv2.putText(frame, f"{status} {info['prediccion']}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_pred, 3)
            y_pos += 40
            
            # PORCENTAJE DE CONFIANZA MUY VISIBLE
            cv2.putText(frame, f"CONFIANZA: {info['confianza']:.1%}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_pred, 2)
        else:
            cv2.putText(frame, "Esperando gesto...", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_pos += 40
        
        y_pos += 30
        
        # Estado de mano y auto-reconocimiento
        mano_estado = "✅ MANO" if self.debug_info['mano_detectada'] else "❌ SIN MANO"
        color_mano = (0, 255, 0) if self.debug_info['mano_detectada'] else (0, 0, 255)
        cv2.putText(frame, mano_estado, (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_mano, 2)
        
        auto_estado = "🤖 AUTO" if self.auto_reconocimiento else "👤 MANUAL"
        cv2.putText(frame, auto_estado, (w - 200, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Buffer status
        color_buffer = (0, 255, 0) if buffer_actual == buffer_max else (255, 255, 255)
        cv2.putText(frame, f"Buffer: {buffer_actual}/{buffer_max}", (w - 200, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_buffer, 2)
        
        # Umbral actual
        cv2.putText(frame, f"Umbral: {umbral_actual:.0%}", (w - 200, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # TEXTO RECONOCIDO EN LA PARTE INFERIOR
        cv2.putText(frame, "TEXTO RECONOCIDO:", (10, h-90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Mostrar texto completo (limitado)
        texto_mostrar = " ".join(self.texto_completo)
        if len(texto_mostrar) > 50:
            texto_mostrar = "..." + texto_mostrar[-47:]
        
        cv2.putText(frame, texto_mostrar, (10, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Última palabra reconocida (si está hablando)
        if self.ultima_palabra_reconocida:
            cv2.putText(frame, f"HABLANDO: {self.ultima_palabra_reconocida}", (10, h-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Controles
        cv2.putText(frame, "ESPACIO: Manual | A: Auto | M: Modo | +/-: Umbral | C: Limpiar | G: Guardar | Q: Salir", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Indicador de listo para reconocimiento manual
        if buffer_actual == buffer_max and info['confianza'] >= umbral_actual:
            cv2.putText(frame, "PRESIONA ESPACIO!", (w - 250, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _reconocimiento_manual(self):
        """Reconocimiento manual con ESPACIO"""
        if self.modo_actual == self.MODO_ESTATICO:
            info = self.debug_info['estatico']
            buffer_listo = len(self.buffer_frames_estatico) == self.FRAMES_PALABRA_ESTATICA
            umbral = self.UMBRAL_CONFIANZA_ESTATICO
        else:
            info = self.debug_info['dinamico']
            buffer_listo = len(self.buffer_frames_dinamico) == self.FRAMES_PALABRA_DINAMICA
            umbral = self.UMBRAL_CONFIANZA_DINAMICO
        
        if buffer_listo and info['prediccion'] and info['confianza'] >= umbral:
            print(f"🔥 MANUAL: {info['prediccion']} ({info['confianza']:.1%})")
            self._procesar_palabra_reconocida(info['prediccion'])
            self._limpiar_buffers()
        else:
            if not buffer_listo:
                print(f"❌ Buffer incompleto para {self.modo_actual}")
            elif not info['prediccion']:
                print("❌ No hay predicción válida")
            else:
                print(f"❌ Confianza baja: {info['confianza']:.1%} < {umbral:.0%}")
    
    def _procesar_palabra_reconocida(self, palabra):
        """Procesar palabra reconocida"""
        self.texto_completo.append(palabra)
        self.ultima_palabra_reconocida = palabra
        
        print(f"🗣️ Reconocido: {palabra}")
        print(f"📝 Texto: {' '.join(self.texto_completo)}")
        
        #if self.tts_disponible:
        #    try:
        #        self.tts.say(palabra)
        #        self.tts.runAndWait()
        #        self.ultima_palabra_reconocida = ""
        #    except Exception as e:
        #        print(f"⚠️ Error TTS: {e}")
        #        self.ultima_palabra_reconocida = ""
    
    def _guardar_texto_final(self):
        """Guardar texto reconocido"""
        if not self.texto_completo:
            print("❌ No hay texto para guardar")
            return
        
        texto = " ".join(self.texto_completo)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archivo = f'texto_reconocido_{timestamp}.txt'
        
        with open(archivo, 'w', encoding='utf-8') as f:
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sistema: Reconocimiento dual simplificado\n")
            f.write(f"Texto reconocido: {texto}\n")
            f.write(f"Palabras: {len(self.texto_completo)}\n")
        
        print(f"💾 Texto guardado en: {os.path.abspath(archivo)}")
    
    def mostrar_menu(self):
        """Menú principal"""
        while True:
            # Estado de los modelos
            estado_estatico = "✅ LISTO" if self.modelo_estatico_entrenado else "❌ NO ENTRENADO"
            estado_dinamico = "✅ LISTO" if self.modelo_dinamico_entrenado else "❌ NO ENTRENADO"
            
            print(f"""
🤟 RECONOCEDOR DUAL SIMPLIFICADO
===============================
📍 Modelo estático (letras/números): {estado_estatico}
🌊 Modelo dinámico (palabras): {estado_dinamico}

1️⃣  ENTRENAR - Enseñar nuevas señas
2️⃣  RECONOCER - Traducir señas a texto y voz
3️⃣  Ver datos de entrenamiento
4️⃣  Cambiar modo actual ({self.modo_actual})
0️⃣  Salir
""")
            
            opcion = input("🎯 Selecciona opción: ").strip()
            
            if opcion == "1":
                self.modo_entrenamiento()
            elif opcion == "2":
                self.modo_reconocimiento()
            elif opcion == "3":
                self._mostrar_datos()
            elif opcion == "4":
                self.alternar_modo()
            elif opcion == "0":
                print("👋 ¡Hasta luego!")
                break
            else:
                print("❌ Opción inválida")
    
    def _mostrar_datos(self):
        """Mostrar datos de entrenamiento"""
        print("\n📊 DATOS DE ENTRENAMIENTO")
        print("=" * 30)
        
        # Datos estáticos
        print("📍 SEÑAS ESTÁTICAS:")
        if os.path.exists('datos/estatico') and os.listdir('datos/estatico'):
            total_estatico = 0
            for archivo in os.listdir('datos/estatico'):
                if archivo.endswith('.pkl'):
                    palabra = archivo[:-4]
                    with open(f'datos/estatico/{archivo}', 'rb') as f:
                        muestras = pickle.load(f)
                    cantidad = len(muestras)
                    total_estatico += cantidad
                    print(f"   📂 {palabra}: {cantidad} muestras")
            print(f"   📈 Total estático: {total_estatico} muestras")
        else:
            print("   ❌ Sin datos estáticos")
        
        # Datos dinámicos
        print("\n🌊 SEÑAS DINÁMICAS:")
        if os.path.exists('datos/dinamico') and os.listdir('datos/dinamico'):
            total_dinamico = 0
            for archivo in os.listdir('datos/dinamico'):
                if archivo.endswith('.pkl'):
                    palabra = archivo[:-4]
                    with open(f'datos/dinamico/{archivo}', 'rb') as f:
                        muestras = pickle.load(f)
                    cantidad = len(muestras)
                    total_dinamico += cantidad
                    print(f"   📂 {palabra}: {cantidad} muestras")
            print(f"   📈 Total dinámico: {total_dinamico} muestras")
        else:
            print("   ❌ Sin datos dinámicos")
        
        print(f"\n🧠 MODELOS:")
        print(f"   📍 Estático: {total_estatico}")
        print(f"   🌊 Dinámico: {total_dinamico}")
        
        input("\n⏎ Presiona Enter para continuar...")

#def main():
#    """Función principal"""
#    print("🚀 Iniciando Reconocedor Dual Simplificado...")
#    
#    try:
#        import cv2, mediapipe, tensorflow, sklearn, pyttsx3
#        print("✅ Dependencias cargadas")
#    except ImportError as e:
#        print(f"❌ Dependencia faltante: {e}")
#        return
#    
#    sistema = LenguaSeñasDual()
#    sistema.mostrar_menu()
#
#if __name__ == "__main__":
#    main()