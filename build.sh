#!/usr/bin/env bash
# Este script detendrá la construcción si algún comando falla
set -o errexit

echo "--- Iniciando proceso de construcción final ---"

# --- Paso 1: Instalar dependencias de Python ---
echo "--- Instalando dependencias de requirements.txt ---"
pip install -r requirements.txt

# --- Paso 2: Limpiar y crear la carpeta de modelos ---
echo "--- Preparando carpeta de modelos (limpiando primero) ---"
rm -rf modelos
mkdir -p modelos

# --- Paso 3: Descargar los modelos con las URLs directas y correctas ---
echo "--- Descargando modelos desde el Release v1.2 ---"
curl -L -o modelos/modelo_estatico.h5 "https://github.com/WilsonGgEz/reconocedor-lengua-senas/releases/download/v1.2/modelo_estatico.h5"
curl -L -o modelos/encoder_estatico.pkl "https://github.com/WilsonGgEz/reconocedor-lengua-senas/releases/download/v1.2/encoder_estatico.pkl"
curl -L -o modelos/modelo_dinamico.h5 "https://github.com/WilsonGgEz/reconocedor-lengua-senas/releases/download/v1.2/modelo_dinamico.h5"
curl -L -o modelos/encoder_dinamico.pkl "https://github.com/WilsonGgEz/reconocedor-lengua-senas/releases/download/v1.2/encoder_dinamico.pkl"

# --- Paso 4: Verificar que los archivos se descargaron correctamente ---
echo "--- Verificando archivos descargados (los tamaños deben ser KB y MB, no bytes) ---"
ls -lh modelos/

echo "--- Proceso de construcción finalizado con éxito ---"