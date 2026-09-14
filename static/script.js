const textarea = document.getElementById("resena");
const contador = document.getElementById("contador");
const botonAnalizar = document.getElementById("boton-analizar");
const mensajeError = document.getElementById("mensaje-error");

const seccionResultado = document.getElementById("resultado");
const badgeSentimiento = document.getElementById("badge-sentimiento");
const textoConfianza = document.getElementById("texto-confianza");
const barraPositiva = document.getElementById("barra-positiva");
const barraNegativa = document.getElementById("barra-negativa");
const valorPositiva = document.getElementById("valor-positiva");
const valorNegativa = document.getElementById("valor-negativa");
const resumenTokens = document.getElementById("resumen-tokens");
const contenedorTokens = document.getElementById("contenedor-tokens");

const LONGITUD_MAXIMA = 5000;

textarea.addEventListener("input", () => {
    contador.textContent = `${textarea.value.length} / ${LONGITUD_MAXIMA} caracteres`;
});

botonAnalizar.addEventListener("click", analizarResena);

async function analizarResena() {
    const texto = textarea.value.trim();
    ocultarError();

    if (!texto) {
        mostrarError("Escribe una reseña antes de analizarla.");
        return;
    }

    botonAnalizar.disabled = true;
    botonAnalizar.textContent = "Analizando...";

    try {
        const respuesta = await fetch("/predecir", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ resena: texto }),
        });

        const datos = await respuesta.json();

        if (!respuesta.ok) {
            mostrarError(datos.error || "Ocurrió un error al analizar la reseña.");
            return;
        }

        mostrarResultado(datos);
    } catch (error) {
        mostrarError("No se pudo conectar con el servidor. Intenta de nuevo.");
    } finally {
        botonAnalizar.disabled = false;
        botonAnalizar.textContent = "Analizar reseña";
    }
}

function mostrarError(texto) {
    mensajeError.textContent = texto;
    mensajeError.hidden = false;
}

function ocultarError() {
    mensajeError.hidden = true;
}

function mostrarResultado(datos) {
    const esPositiva = datos.prediccion === "positive";

    badgeSentimiento.textContent = esPositiva ? "😀 Positiva" : "🙁 Negativa";
    badgeSentimiento.className = "badge " + (esPositiva ? "positivo" : "negativo");

    const confianza = esPositiva ? datos.probabilidad_positive : datos.probabilidad_negative;
    textoConfianza.textContent = `Confianza del modelo: ${(confianza * 100).toFixed(1)}%`;

    const porcentajePositiva = datos.probabilidad_positive * 100;
    const porcentajeNegativa = datos.probabilidad_negative * 100;

    barraPositiva.style.width = `${porcentajePositiva}%`;
    barraNegativa.style.width = `${porcentajeNegativa}%`;
    valorPositiva.textContent = `${porcentajePositiva.toFixed(1)}%`;
    valorNegativa.textContent = `${porcentajeNegativa.toFixed(1)}%`;

    if (datos.total_tokens_reconocidos === 0) {
        resumenTokens.textContent =
            "No reconocimos ninguna palabra de esta reseña en nuestro vocabulario " +
            "(quizá está vacía, en otro idioma, o son puras palabras muy comunes). " +
            "La predicción se basa únicamente en la proporción base de reseñas " +
            "positivas y negativas del dataset de entrenamiento.";
    } else {
        resumenTokens.textContent =
            `Tu reseña tiene ${datos.total_palabras_texto} palabras en total; ` +
            `el modelo reconoció ${datos.total_tokens_reconocidos} de ellas ` +
            "(las demás son palabras muy comunes o desconocidas para el vocabulario aprendido).";
    }

    renderizarTokens(datos.tokens_relevantes);

    seccionResultado.hidden = false;
    seccionResultado.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderizarTokens(tokens) {
    contenedorTokens.innerHTML = "";

    if (!tokens || tokens.length === 0) {
        return;
    }

    const maxContribucion = Math.max(...tokens.map((t) => Math.abs(t.contribucion)), 0.0001);

    tokens.forEach((token) => {
        const intensidad = Math.abs(token.contribucion) / maxContribucion;
        const chip = document.createElement("span");
        chip.className = "token";
        chip.textContent = token.palabra;
        chip.title =
            `Peso TF-IDF: ${token.tfidf}\n` +
            `Contribución hacia "${token.favorece}": ${token.contribucion}`;

        if (token.favorece === "positive") {
            chip.style.background = `rgba(31, 157, 85, ${0.15 + intensidad * 0.5})`;
            chip.style.color = "#155c33";
        } else {
            chip.style.background = `rgba(214, 69, 69, ${0.15 + intensidad * 0.5})`;
            chip.style.color = "#8a2c2c";
        }

        contenedorTokens.appendChild(chip);
    });
}
