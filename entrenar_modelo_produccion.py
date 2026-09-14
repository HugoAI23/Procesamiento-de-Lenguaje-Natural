"""
Entrena y persiste el modelo de produccion: TfidfVectorizer + MultinomialNB
para clasificar el sentimiento de resenas de peliculas.

Decision clave (a diferencia de los notebooks de exploracion): aqui se usa el
dataset COMPLETO ya sin duplicados (49,582 filas), no el submuestreo de
5,000+5,000 usado para explorar mas rapido. Ventaja: mas datos -> menor
varianza en las estimaciones y mejor generalizacion del modelo final.
Desventaja: entrenamiento un poco mas lento (aunque Naive Bayes sigue siendo
muy rapido incluso con el dataset completo).
"""
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

RUTA_DATOS = "data/IMDB Dataset.csv"
CARPETA_MODELO = "modelo"


def cargar_datos_limpios():
    df = pd.read_csv(RUTA_DATOS)
    antes = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Filas originales: {antes:,} | Tras quitar duplicados: {df.shape[0]:,}")
    return df


def evaluar_con_holdout(df):
    """
    Separamos un 15% de los datos para medir honestamente el desempeno
    ANTES de entrenar el modelo final con el 100% de los datos.

    Ventaja: da una estimacion confiable de como se comportara el modelo
    con resenas que nunca ha visto (simula el uso real en produccion).
    Desventaja: ese 15% "se pierde" para el modelo que finalmente se
    despliega -- por eso, una vez medido el desempeno, se reentrena con
    todo el dataset en `entrenar_modelo_final`.
    """
    train, test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["sentiment"]
    )

    vectorizador = TfidfVectorizer(stop_words="english")
    x_train = vectorizador.fit_transform(train["review"])
    x_test = vectorizador.transform(test["review"])

    modelo = MultinomialNB()
    modelo.fit(x_train, train["sentiment"])
    pred = modelo.predict(x_test)

    exactitud = accuracy_score(test["sentiment"], pred)
    reporte = classification_report(test["sentiment"], pred, output_dict=True)

    print(f"\nExactitud estimada en datos nunca vistos (holdout 15%): {exactitud * 100:.2f}%")
    print(classification_report(test["sentiment"], pred))
    return exactitud, reporte


def entrenar_modelo_final(df):
    """
    Reentrena con el 100% de los datos disponibles (sin dejar holdout).
    Una vez medido el desempeno real arriba, para el modelo que se
    despliega conviene aprovechar toda la informacion disponible.
    """
    vectorizador = TfidfVectorizer(stop_words="english")
    x_todo = vectorizador.fit_transform(df["review"])

    modelo = MultinomialNB()
    modelo.fit(x_todo, df["sentiment"])
    return vectorizador, modelo


def guardar_artefactos(vectorizador, modelo, exactitud, reporte, n_filas):
    os.makedirs(CARPETA_MODELO, exist_ok=True)

    joblib.dump(vectorizador, os.path.join(CARPETA_MODELO, "vectorizador_tfidf.joblib"))
    joblib.dump(modelo, os.path.join(CARPETA_MODELO, "modelo_naive_bayes.joblib"))

    metadata = {
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "filas_entrenamiento_final": n_filas,
        "exactitud_holdout_15pct": exactitud,
        "reporte_holdout": reporte,
        "modelo": "MultinomialNB",
        "vectorizador": 'TfidfVectorizer(stop_words="english")',
    }
    ruta_metadata = os.path.join(CARPETA_MODELO, "metadata.json")
    with open(ruta_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nArtefactos guardados en '{CARPETA_MODELO}/':")
    print("  - vectorizador_tfidf.joblib")
    print("  - modelo_naive_bayes.joblib")
    print("  - metadata.json")


if __name__ == "__main__":
    df = cargar_datos_limpios()
    exactitud, reporte = evaluar_con_holdout(df)
    vectorizador, modelo = entrenar_modelo_final(df)
    guardar_artefactos(vectorizador, modelo, exactitud, reporte, df.shape[0])
    print("\nListo. El modelo de produccion esta entrenado con el dataset completo.")
