#!/usr/bin/env bash
# exit on error
set -o errexit

echo "--- Iniciando proceso de construcción ---"

# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Limpiar y crear carpeta para los modelos
# (Añadimos 'rm -rf' para asegurar que la carpeta esté limpia en cada build)
rm -rf modelos/*
mkdir -p modelos

# 3. Descargar los modelos con las URLs CORRECTAS
echo "--- Descargando modelos ---"
curl -L -o modelos/modelo_estatico_web.h5 'https://github.com/WilsonGgEz/reconocedor-lengua-senas/releases/download/test2/modelo_estatico_web.h5'
curl -L -o modelos/encoder_estatico_web.pkl 'https://github.com/WilsonGgEz/reconocedor-lengua-senas/releases/download/test2/encoder_estatico_web.pkl'
curl -L -o modelos/modelo_dinamico_web.h5 'https://github.com/WilsonGgEz/reconocedor-lengua-senas/releases/download/test2/modelo_dinamico_web.h5'
curl -L -o modelos/encoder_dinamico_web.pkl 'https://github.com/WilsonGgEz/reconocedor-lengua-senas/releases/download/test2/encoder_dinamico_web.pkl'

# 4. Verificar que los archivos existen
echo "--- Verificando archivos descargados ---"
ls -lh modelos/

echo "--- Proceso de construcción finalizado ---"