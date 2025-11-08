import streamlit as st
import pandas as pd
import altair as alt
from google import genai
import os
import csv
from datetime import datetime

# =========================
# CONFIG PÁGINA
# =========================
st.set_page_config(
    page_title="Dashboard Sequías - Riohacha",
    page_icon="🌵",
    layout="wide",
)

# =========================
# CARGA Y PREPARACIÓN DE DATOS
# =========================
@st.cache_data
def load_data():
    df = pd.read_parquet("dataset_clima.parquet")

    # Asegurar columna de tiempo
    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df["date"] = df["valid_time"]
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        st.error("No se encontró columna de tiempo ('valid_time' o 'date') en el dataset.")
        st.stop()

    # Año
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year

    # Asegurar precipitacion_total
    if "precipitacion_total" not in df.columns:
        st.error("No se encontró la columna 'precipitacion_total' en el dataset.")
        st.stop()

    # Serie mensual agregada (ej: promedio espacial de precipitacion_total)
    monthly = (
        df.groupby(pd.Grouper(key="date", freq="MS"))["precipitacion_total"]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    return df, monthly


df, monthly = load_data()

if monthly.empty:
    st.error("No hay datos mensuales disponibles.")
    st.stop()

# =========================
# ENCABEZADO
# =========================

st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0.4rem;'>"
    "Dashboard de Sequías - Riohacha"
    "</h1>",
    unsafe_allow_html=True,
)

# Placeholder: valor de tu modelo
prob_sequia = 37

st.markdown(
    f"<p style='text-align: center; font-size: 1.1rem; margin-top: 0.2rem;'>"
    f"Según los datos disponibles, hay una probabilidad de "
    f"<b>{prob_sequia}%</b> de que estemos en una época de sequía en Riohacha."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================
# SECCIÓN 1:
# ÚLTIMOS 12 MESES + 1 MES FUTURO (precipitacion_total)
# =========================

last_12 = monthly.tail(12).copy()
last_date = last_12["date"].max()
next_month_date = last_date + pd.DateOffset(months=1)

# Predicción dummy: promedio últimos 12 meses
pred_prec = float(last_12["precipitacion_total"].mean())

last_12["is_pred"] = 0
pred_row = pd.DataFrame(
    {"date": [next_month_date], "precipitacion_total": [pred_prec], "is_pred": [1]}
)
plot_df = pd.concat([last_12, pred_row], ignore_index=True)

# Bandas de colores (ajusta umbrales según tu lógica)
bands = pd.DataFrame([
    {"y1": 0, "y2": 1, "color": "#2196F3"},  # azul
    {"y1": 1, "y2": 2, "color": "#4CAF50"},  # verde
    {"y1": 2, "y2": 3, "color": "#FFEB3B"},  # amarillo
    {"y1": 3, "y2": 4, "color": "#F44336"},  # rojo
])

start = plot_df["date"].min()
end = plot_df["date"].max()
bands["start"] = start
bands["end"] = end

band_chart = (
    alt.Chart(bands)
    .mark_rect(opacity=0.25)
    .encode(
        x="start:T",
        x2="end:T",
        y="y1:Q",
        y2="y2:Q",
        color=alt.Color("color:N", scale=None, legend=None),
    )
)

line_chart = (
    alt.Chart(plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "date:T",
            title="Mes",
            axis=alt.Axis(
                format="%b",
                tickCount=13,
            ),
        ),
        y=alt.Y("precipitacion_total:Q", title="precipitacion_total"),
        color=alt.condition(
            "datum.is_pred == 1",
            alt.value("#F44336"),   # predicción
            alt.value("#00d492"),   # histórico
        ),
        tooltip=["date:T", "precipitacion_total:Q"],
    )
)

pred_points = (
    alt.Chart(plot_df[plot_df["is_pred"] == 1])
    .mark_point(size=80, filled=True)
    .encode(
        x="date:T",
        y="precipitacion_total:Q",
        color=alt.value("#F44336"),
        tooltip=["date:T", "precipitacion_total:Q"],
    )
)

st.subheader("Últimos 12 meses de 'precipitacion_total' + proyección al siguiente mes")
st.altair_chart(band_chart + line_chart + pred_points, width="stretch")

st.markdown("---")

# =========================
# SECCIÓN 2:
# FILTROS GENERALES (SIDEBAR IZQUIERDA)
# =========================

st.sidebar.header("Filtros generales")

numeric_cols = [
    c for c in df.select_dtypes(include="number").columns
    if c not in ["year"]
]

if not numeric_cols:
    st.error("No se encontraron variables numéricas para visualizar.")
    st.stop()

selected_var = st.sidebar.selectbox("Variable a analizar:", numeric_cols)

year_min = int(df["year"].min())
year_max = int(df["year"].max())

start_year, end_year = st.sidebar.select_slider(
    "Rango de años:",
    options=list(range(year_min, year_max + 1)),
    value=(year_min, year_max),
)

mask = (df["year"] >= start_year) & (df["year"] <= end_year)
df_filtered = df[mask]

if df_filtered.empty:
    st.warning("No hay datos para ese rango de años.")
    st.stop()

# =========================
# SECCIÓN 3:
# LAYOUT: MAIN (EXPLORACIÓN) + PANEL DERECHO (CHATBOT)
# =========================

st.markdown("---")
main_col, chat_col = st.columns([3, 1])

# --------- MAIN: EXPLORACIÓN ----------
with main_col:
    st.subheader(f"Exploración histórica de {selected_var} ({start_year}–{end_year})")

    # Métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("Media", f"{df_filtered[selected_var].mean():.4f}")
    c2.metric("Mínimo", f"{df_filtered[selected_var].min():.4f}")
    c3.metric("Máximo", f"{df_filtered[selected_var].max():.4f}")

    # Gráficas
    left_col, right_col = st.columns((2.5, 1.5))

    with left_col:
        st.markdown("**Promedio anual**")
        yearly = (
            df_filtered
            .groupby("year")[selected_var]
            .mean()
            .reset_index()
        )

        chart_yearly = (
            alt.Chart(yearly)
            .mark_line(point=True)
            .encode(
                x=alt.X("year:O", title="Año"),
                y=alt.Y(selected_var, title=selected_var),
                tooltip=["year", selected_var],
            )
        )
        st.altair_chart(chart_yearly, width="stretch")

    with right_col:
        st.markdown("**Distribución en el rango seleccionado**")
        hist = (
            alt.Chart(df_filtered)
            .mark_bar()
            .encode(
                x=alt.X(selected_var, bin=alt.Bin(maxbins=30), title=selected_var),
                y=alt.Y("count()", title="Frecuencia"),
            )
        )
        st.altair_chart(hist, width="stretch")

    # Tabla
    st.markdown("**Muestra de datos filtrados**")
    st.dataframe(df_filtered.head(200), use_container_width=True)

# --------- PANEL DERECHO: CHATBOT CON SCROLL ----------
with chat_col:
    st.markdown("### 💬 Asistente climático")

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        st.warning("Configura GEMINI_API_KEY para habilitar el chatbot.")
    else:
        client = genai.Client(api_key=api_key)

        # Inicializar historial una sola vez
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Hola 👋 Soy tu asistente climático basado en Gemini. "
                        "Puedo ayudarte a entender este dashboard y las posibles sequías en Riohacha."
                    ),
                }
            ]

        # Capturar mensaje nuevo
        user_input = st.chat_input("Escribe tu pregunta sobre el clima o las sequías...")

        if user_input:
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_input}
            )

            contexto_basico = (
                f"Probabilidad de sequía: {prob_sequia}%. "
                f"Años visibles: {start_year}-{end_year}. "
                f"Variable seleccionada: {selected_var}."
            )

            try:
                prompt = (
                    "Eres un asistente experto en clima y sequías en Riohacha.\n"
                    "Responde en español, claro y sin inventar datos.\n\n"
                    f"Contexto del dashboard:\n{contexto_basico}\n\n"
                    f"Pregunta del usuario:\n{user_input}"
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                reply_text = (response.text or "").strip()
            except Exception as e:
                reply_text = (
                    f"No pude obtener respuesta de Gemini. "
                    f"Detalle técnico: {e}"
                )

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": reply_text}
            )

        # Contenedor scrollable para el historial
        chat_box = st.container(height=350, border=True)

        with chat_box:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f"🧑‍💻 **Tú:** {msg['content']}")
                else:
                    st.markdown(f"🤖 **Asistente:** {msg['content']}")

# =========================
# 5) BUZÓN DE REPORTES
# =========================

st.markdown("---")
st.header("Buzón de Reportes")

st.markdown(
    "Si notas signos de sequía o cambios importantes en el clima de tu zona, "
    "puedes dejar aquí tu observación. ¡Tu aporte ayuda a mejorar la información local!"
)

with st.form("form_reporte"):
    nombre = st.text_input("Tu nombre (opcional):")
    municipio = st.text_input("Municipio o zona:")
    mensaje = st.text_area("Descripción de tu observación:")

    enviado = st.form_submit_button("Enviar reporte")

    if enviado:
        if mensaje.strip() == "":
            st.warning("Por favor escribe una observación antes de enviar.")
        else:
            with open("reportes_usuarios.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), nombre, municipio, mensaje])

            st.success("¡Gracias por tu reporte! Se ha enviado correctamente.")
