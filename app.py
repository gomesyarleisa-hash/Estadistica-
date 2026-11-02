# =====================================================================
#  ESTADÍSTICA PRO  -  Versión 1.0
#  Aplicación web de distribuciones muestrales (Ingeniería Industrial)
#
#  Tecnología (todo gratuito):
#    Python + Streamlit (pantalla) + NumPy/Pandas (cálculos)
#    + SciPy (probabilidades) + Matplotlib (gráficas)
#
#  Para abrirla en tu computador:   streamlit run app.py
# =====================================================================

import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="ESTADÍSTICA PRO", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------
#  CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------
POBLACION_DEFECTO = "1, 9, 12, 14, 16, 20"
MAX_MUESTRAS = 200_000
AZUL, ROJO, VERDE = "#1f4e8c", "#d9534f", "#2e8b57"

OPCIONES_EVENTO = {
    "Mayor que  (X > a)": "mayor",
    "Menor que  (X < a)": "menor",
    "Entre  (a < X < b)": "entre",
}

# ---------------------------------------------------------------------
#  FUNCIONES DE APOYO
# ---------------------------------------------------------------------
def leer_numeros(texto):
    """Convierte una cadena de números en una lista."""
    partes = texto.replace(";", ",").replace("\n", ",").split(",")
    try:
        return [float(p) for p in partes if p.strip() != ""]
    except ValueError:
        return None


def obtener_poblacion():
    """Devuelve la población guardada en el módulo 1."""
    return st.session_state.get("poblacion", leer_numeros(POBLACION_DEFECTO))


def media_varianza_desviacion(datos):
    """Calcula media, varianza y desviación de una población (Divisor N)."""
    d = np.array(datos, dtype=float)
    mu = d.mean()
    varianza = ((d - mu) ** 2).mean()
    sigma = math.sqrt(varianza)
    return mu, varianza, sigma


def media_y_desviacion(datos):
    """Mantiene la función original para los cálculos existentes."""
    mu, varianza, sigma = media_varianza_desviacion(datos)
    return mu, sigma


@st.cache_data
def calcular_muestras(poblacion, n, con_reposicion):
    """Genera todas las muestras posibles de tamaño n, sus medias, Sx y Sx²."""
    if con_reposicion:
        total = len(poblacion) ** n
        generador = itertools.product(poblacion, repeat=n)
    else:
        total = math.comb(len(poblacion), n)
        generador = itertools.combinations(poblacion, n)

    if total > MAX_MUESTRAS:
        return None, total

    filas = []
    for m in generador:
        m_arr = np.array(m, dtype=float)
        # CORRECCIÓN METODOLÓGICA: Se usa ddof=1 para las estadísticas de muestras individuales (Sx)
        desv_muestral = np.std(m_arr, ddof=1) if len(m_arr) > 1 else 0.0
        var_muestral = np.var(m_arr, ddof=1) if len(m_arr) > 1 else 0.0
        
        filas.append({
            "Muestra": "(" + ", ".join(f"{v:g}" for v in m) + ")",
            "Media (x̄)": float(np.mean(m_arr)),
            "Desviación Muestral (Sx)": float(desv_muestral),
            "Varianza Muestral (Sx²)": float(var_muestral)
        })

    tabla = pd.DataFrame(filas)
    tabla.index = range(1, len(tabla) + 1)
    return tabla, total


def grafica_barras_medias(medias, color, titulo):
    """Gráfica de frecuencias de las medias muestrales."""
    conteo = pd.Series(np.round(medias, 4)).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar([f"{v:g}" for v in conteo.index], conteo.values, color=color)
    ax.set_title(titulo)
    ax.set_xlabel("Media de la muestra")
    ax.set_ylabel("Frecuencia")

    if len(conteo) > 12:
        ax.tick_params(axis="x", rotation=90, labelsize=7)

    fig.tight_layout()
    return fig


def bloque_evento(media, desv, nombre, a0, b0, paso, formato, clave):
    """
    Calcula probabilidades con la distribución normal y muestra
    la media, varianza y desviación estándar.
    """
    if desv <= 0:
        st.error("La desviación debe ser mayor que cero.")
        return

    varianza = desv ** 2

    st.markdown("#### Evento cuya probabilidad quieres calcular")

    eleccion = st.radio(
        "Tipo de evento",
        list(OPCIONES_EVENTO),
        horizontal=True,
        key=f"{clave}_tipo"
    )

    tipo = OPCIONES_EVENTO[eleccion]

    c1, c2 = st.columns(2)

    a = c1.number_input(
        "Valor a",
        value=float(a0),
        step=paso,
        format=formato,
        key=f"{clave}_a"
    )

    b = None

    if tipo == "entre":
        b = c2.number_input(
            "Valor b",
            value=float(b0),
            step=paso,
            format=formato,
            key=f"{clave}_b"
        )

        if b <= a:
            st.warning("El valor b debe ser mayor que a.")
            return

    # Cálculo con la normal
    za = (a - media) / desv

    if tipo == "mayor":
        prob = 1 - stats.norm.cdf(za)
        evento = f"P({nombre} > {a:g})"
        pasos = f"z = ({a:g} − {media:.4f}) / {desv:.4f} = {za:.4f}"

    elif tipo == "menor":
        prob = stats.norm.cdf(za)
        evento = f"P({nombre} < {a:g})"
        pasos = f"z = ({a:g} − {media:.4f}) / {desv:.4f} = {za:.4f}"

    else:
        zb = (b - media) / desv
        prob = stats.norm.cdf(zb) - stats.norm.cdf(za)
        evento = f"P({a:g} < {nombre} < {b:g})"

        pasos = (
            f"z₁ = ({a:g} − {media:.4f}) / {desv:.4f} = {za:.4f}   |   "
            f"z₂ = ({b:g} − {media:.4f}) / {desv:.4f} = {zb:.4f}"
        )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Media esperada", f"{media:.4f}")
    m2.metric("Varianza esperada", f"{varianza:.4f}")
    m3.metric("Desviación esperada", f"{desv:.4f}")
    m4.metric(evento, f"{prob:.4f}")

    st.markdown("**Fórmulas utilizadas**")
    st.latex(r"z = \dfrac{\text{valor} - \mu}{\sigma}")

    if tipo == "mayor":
        st.latex(r"P(X > a) = 1 - \Phi(z)")
        detalle_prob = (
            f"P = 1 − Φ({za:.4f}) = 1 − {stats.norm.cdf(za):.4f} "
            f"= {prob:.4f}"
        )
    elif tipo == "menor":
        st.latex(r"P(X < a) = \Phi(z)")
        detalle_prob = f"P = Φ({za:.4f}) = {prob:.4f}"
    else:
        st.latex(r"P(a < X < b) = \Phi(z_2) - \Phi(z_1)")
        detalle_prob = (
            f"P = Φ({zb:.4f}) − Φ({za:.4f}) = "
            f"{stats.norm.cdf(zb):.4f} − {stats.norm.cdf(za):.4f} "
            f"= {prob:.4f}"
        )

    st.caption(f"Estandarización: {pasos}")
    st.caption(detalle_prob)

    # Gráfica de la normal
    x = np.linspace(media - 4 * desv, media + 4 * desv, 600)
    y = stats.norm.pdf(x, media, desv)

    if tipo == "mayor":
        zona = x >= a
    elif tipo == "menor":
        zona = x <= a
    else:
        zona = (x >= a) & (x <= b)

    fig, ax = plt.subplots(figsize=(8, 3.8))

    ax.plot(x, y, color=AZUL, linewidth=2)

    ax.fill_between(
        x[zona],
        y[zona],
        color=ROJO,
        alpha=0.55,
        label=f"{evento} = {prob:.4f}"
    )

    ax.axvline(
        media,
        color="gray",
        linestyle="--",
        label=f"Media = {media:.4f}"
    )

    ax.set_xlabel(nombre)
    ax.set_ylabel("Densidad")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"**Interpretación:** si se repitiera el muestreo muchas veces, "
        f"en aproximadamente el **{prob * 100:.2f} %** de los casos "
        f"se cumpliría el evento {evento}."
    )

# ---------------------------------------------------------------------
#  MÓDULO 1: POBLACIÓN
# ---------------------------------------------------------------------
def modulo_poblacion():
    st.header("1. Población")

    st.write(
        "Escribe los valores de la población separados por comas. "
        "Esta población se usará también en el módulo 2."
    )

    texto = st.text_area(
        "Valores de la población",
        value=st.session_state.get("pob_texto", POBLACION_DEFECTO),
    )

    datos = leer_numeros(texto)

    if datos is None or len(datos) < 2:
        st.error("Escribe al menos 2 números válidos separados por comas.")
        return

    st.session_state["poblacion"] = datos
    st.session_state["pob_texto"] = texto

    N = len(datos)
    mu, varianza, sigma = media_varianza_desviacion(datos)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Tamaño de la población (N)", N)
    c2.metric("Media poblacional (μ)", f"{mu:.4f}")
    c3.metric("Varianza poblacional (σ²)", f"{varianza:.4f}")
    c4.metric("Desviación poblacional (σ)", f"{sigma:.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([f"{v:g}" for v in datos], datos, color=AZUL)
    ax.axhline(mu, color=ROJO, linestyle="--", label=f"μ = {mu:.2f}")

    ax.set_xlabel("Valor")
    ax.set_ylabel("Magnitud")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Ver el cálculo paso a paso"):
        d = np.array(datos)

        detalle = pd.DataFrame({
            "x": d,
            "x − μ": d - mu,
            "(x − μ)²": (d - mu) ** 2
        })

        detalle.index = range(1, N + 1)

        st.dataframe(detalle)

        suma_cuadrados = ((d - mu) ** 2).sum()

        st.write(f"μ = Σx / N = {d.sum():g} / {N} = **{mu:.4f}**")
        st.write(f"σ² = Σ(x − μ)² / N = {suma_cuadrados:.4f} / {N} = **{varianza:.4f}**")
        st.write(f"σ = √σ² = √{varianza:.4f} = **{sigma:.4f}**")

# ---------------------------------------------------------------------
#  EJECUCIÓN DE LA APLICACIÓN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    st.sidebar.title("Navegación")
    seccion = st.sidebar.radio("Ir a:", ["Módulo 1: Población"])
    if seccion == "Módulo 1: Población":
        modulo_poblacion()
