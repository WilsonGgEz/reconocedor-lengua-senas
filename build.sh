#!/usr/bin/env bash
# exit on error
set -o errexit

echo "--- Iniciando proceso de construcción v2 ---"

# ** VERIFICA ESTE NÚMERO DE VERSIÓN **
# Este debe ser el nombre de la "tag" que creaste en tu Release de GitHub (ej: v1.0 o v1.1)
RELEASE_VERSION="test2"

# Construimos la URL base para no repetir
BASE_URL="https://github.com/WilsonGgEz/reconocedor-lengua-senas/releases/download/${RELEASE_VERSION}"

# 1. Instalar dependencias
echo "--- Instalando dependencias de Python ---"
pip install -r requirements.txt

# 2. Limpiar y crear carpeta para los modelos
echo "--- Preparando carpeta de modelos ---"
rm -rf modelos/*
mkdir -p modelos

# 3. Descargar los modelos con las URLs directas y correctas
echo "--- Descargando modelos desde el Release ${RELEASE_VERSION} ---"
curl -L -o modelos/modelo_estatico_web.h5 "${BASE_URL}/modelo_estatico_web.h5"
curl -L -o modelos/encoder_estatico_web.pkl "${BASE_URL}/encoder_estatico_web.pkl"
curl -L -o modelos/modelo_dinamico_web.h5 "${BASE_URL}/modelo_dinamico_web.h5"
curl -L -o modelos/encoder_dinamico_web.pkl "${BASE_URL}/encoder_dinamico_web.pkl"

# 4. Verificar que los archivos se descargaron correctamente
echo "--- Verificando archivos descargados (tamaños esperados en MB para .h5) ---"
ls -lh modelos/

echo "--- Proceso de construcción finalizado ---"