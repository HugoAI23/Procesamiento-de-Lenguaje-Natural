"""
API + interfaz web para clasificar el sentimiento de resenas de peliculas
usando el modelo Naive Bayes entrenado en 'entrenar_modelo_produccion.py'.

Desarrollo:   python app.py
Produccion:   gunicorn app:app
"""
import os

import joblib
from flask import Flask, jsonify, render_template, request

from explicabilidad import analizar_resena

CARPETA_MODELO = "modelo"
LONGITUD_MAXIMA_RESENA = 5000

app = Flask(__name__)

_ruta_vectorizador = os.path.join(CARPETA_MODELO, "vectorizador_tfidf.joblib")
_ruta_modelo = os.path.join(CARPETA_MODELO, "modelo_naive_bayes.joblib")

if not (os.path.exists(_ruta_vectorizador) and os.path.exists(_ruta_modelo)):
    raise FileNotFoundError(
        "No se encontraron los artefactos del modelo en 'modelo/'. "
        "Ejecuta primero: python entrenar_modelo_produccion.py"
    )

vectorizador = joblib.load(_ruta_vectorizador)
modelo = joblib.load(_ruta_modelo)


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/predecir", methods=["POST"])
def predecir():
    datos = request.get_json(silent=True) or {}
    texto = (datos.get("resena") or "").strip()

    if not texto:
        return jsonify({"error": "Escribe una resena antes de analizarla."}), 400
    if len(texto) > LONGITUD_MAXIMA_RESENA:
        return jsonify({
            "error": f"La resena es demasiado larga (maximo {LONGITUD_MAXIMA_RESENA} caracteres)."
        }), 400

    resultado = analizar_resena(texto, vectorizador, modelo)
    return jsonify(resultado)


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=puerto)
