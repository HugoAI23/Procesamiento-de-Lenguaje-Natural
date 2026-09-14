"""
Traduce una prediccion de Naive Bayes en una explicacion legible para un
usuario sin conocimientos de Machine Learning: que palabras (tokens)
reconocio el modelo en la resena y cuanto empujo cada una hacia "positive"
o "negative".

Matematicamente, Naive Bayes clasifica sumando, solo sobre los indices no
cero del vector TF-IDF:
    log P(y|x) = log P(y) + sum_i  x_i * log P(palabra_i | y)
Aqui reutilizamos exactamente esa idea: para cada palabra presente en la
resena, calculamos cuanto "pesa" a favor de cada clase.
"""


def analizar_resena(texto, vectorizador, modelo, top_n=12):
    vector = vectorizador.transform([texto])
    vocabulario = vectorizador.get_feature_names_out()

    clases = list(modelo.classes_)
    idx_pos = clases.index("positive")
    idx_neg = clases.index("negative")
    log_prob = modelo.feature_log_prob_

    indices_no_cero = vector.indices
    valores_no_cero = vector.data

    contribuciones = []
    for idx, valor in zip(indices_no_cero, valores_no_cero):
        # diferencia entre log P(palabra|positive) y log P(palabra|negative):
        # > 0 => la palabra empuja hacia "positive"; < 0 => empuja hacia "negative"
        diferencia = log_prob[idx_pos, idx] - log_prob[idx_neg, idx]
        contribuciones.append(
            {
                "palabra": vocabulario[idx],
                "tfidf": round(float(valor), 4),
                "contribucion": round(float(valor * diferencia), 4),
                "favorece": "positive" if diferencia > 0 else "negative",
            }
        )

    contribuciones.sort(key=lambda item: abs(item["contribucion"]), reverse=True)

    probabilidades = dict(zip(clases, modelo.predict_proba(vector)[0].round(4)))
    prediccion = modelo.predict(vector)[0]

    return {
        "prediccion": prediccion,
        "probabilidad_positive": float(probabilidades.get("positive", 0)),
        "probabilidad_negative": float(probabilidades.get("negative", 0)),
        "total_palabras_texto": len(texto.split()),
        "total_tokens_reconocidos": int(vector.nnz),
        "tokens_relevantes": contribuciones[:top_n],
    }
