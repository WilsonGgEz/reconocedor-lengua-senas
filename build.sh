#!/usr/bin/env bash
# exit on error
set -o errexit

echo "--- Iniciando proceso de construcción ---"

# 1. Instalar dependencias de Python
pip install -r requirements.txt

# 2. Crear carpeta para los modelos
mkdir -p modelos

# 3. Descargar los modelos desde GitHub Releases
echo "--- Descargando modelos ---"
curl -L -o modelos/modelo_estatico_web.h5 'https://github.com/WilsonGgEz/reconocedor-lengua-senas/blob/main/modelos/modelo_estatico.h5'
curl -L -o modelos/encoder_estatico_web.pkl 'https://github.com/WilsonGgEz/reconocedor-lengua-senas/blob/main/modelos/encoder_estatico.pkl'
curl -L -o modelos/modelo_dinamico_web.h5 'https://github.com/WilsonGgEz/reconocedor-lengua-senas/blob/main/modelos/modelo_dinamico.h5'
curl -L -o modelos/encoder_dinamico_web.pkl 'https://github.com/WilsonGgEz/reconocedor-lengua-senas/blob/main/modelos/encoder_dinamico.pkl'

# 4. Verificar que los archivos existen
echo "--- Verificando archivos descargados ---"
ls -lh modelos/

echo "--- Proceso de construcción finalizado ---"