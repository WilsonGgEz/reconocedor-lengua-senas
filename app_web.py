#waitress-serve --host 0.0.0.0 --port 8000 app_web:app
#http://localhost:8000

# app_web.py (Versión para Cliente Inteligente)

from flask import Flask, render_template, jsonify, request
import numpy as np
from reconocedor import LenguaSeñasDual # Asegúrate que reconocedor.py cargue los modelos originales

app = Flask(__name__)

# Instancia global del reconocedor con los modelos originales (no los _web)
sistema_señas = LenguaSeñasDual()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    """Alterna el modo y devuelve el nuevo estado."""
    sistema_señas.alternar_modo()
    print(f"🔄 Modo cambiado a: {sistema_señas.modo_actual}")
    return jsonify(modo=sistema_señas.modo_actual)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Recibe un vector de características y el MODO desde el navegador.
    """
    data = request.get_json()
    feature_vector = data.get('feature_vector')
    # ¡CAMBIO CLAVE! Leemos el modo que nos manda el cliente en esta petición específica
    mode_from_client = data.get('mode')

    if not feature_vector or not mode_from_client:
        return jsonify(error="No se recibió el vector de características o el modo"), 400

    # Usamos la variable 'mode_from_client' para la decisión, no el estado global
    if mode_from_client == 'estatico':
        modelo_a_usar = sistema_señas.modelo_estatico
        encoder_a_usar = sistema_señas.encoder_estatico
        umbral = sistema_señas.UMBRAL_CONFIANZA_ESTATICO
        modelo_entrenado = sistema_señas.modelo_estatico_entrenado
        secuencia = np.array(feature_vector) # El cliente ya manda la secuencia con el tamaño correcto
    elif mode_from_client == 'dinamico':
        modelo_a_usar = sistema_señas.modelo_dinamico
        encoder_a_usar = sistema_señas.encoder_dinamico
        umbral = sistema_señas.UMBRAL_CONFIANZA_DINAMICO
        modelo_entrenado = sistema_señas.modelo_dinamico_entrenado
        secuencia = np.array(feature_vector)
    else:
        return jsonify(error="Modo no válido"), 400

    prediccion = ""
    confianza = 0.0

    if modelo_entrenado:
        try:
            # Envolvemos la secuencia en un batch de 1
            secuencia_batch = np.expand_dims(secuencia, axis=0)
            pred = modelo_a_usar.predict(secuencia_batch, verbose=0)
            probabilidades = pred[0]
            clase_idx = np.argmax(probabilidades)
            confianza = probabilidades[clase_idx]
            if confianza >= umbral:
                prediccion = encoder_a_usar.inverse_transform([clase_idx])[0]
        except Exception as e:
            # Este print ahora será mucho más informativo
            print(f"Error en predicción (modo solicitado: {mode_from_client}): {e}")
            return jsonify(error=str(e)), 500

    return jsonify(prediccion=prediccion, confianza=float(confianza))

if __name__ == '__main__':
    # Para desarrollo local, waitress sigue siendo una buena opción
    from waitress import serve
    serve(app, host='0.0.0.0', port=8000)

    # PARA LINUX
    # app.run(debug=True)