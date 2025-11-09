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

import xarray as xr
import xclim as xc
import pymannkendall as mk
import plotly.graph_objects as go
import numpy as np

main_col, chat_col = st.columns([3, 1])

with main_col:
    st.header("Análisis climático - ERA5 / SPI/ SPEI")

    # Cargar el archivo .nc con los daros de ERA5
    file_path= "data_stream-moda.nc"

    if not os.path.exists(file_path):
        st.warning("No se encontró el archivo 'data_stream-moda.n'.")
    else:
        ds = xr.open_dataset(file_path)
        df_era = ds.to_dataframe().dropna().reset_index()

        df_era['valid_time'] = pd.to_datetime(df_era['valid_time'])

    # Variables importantes

    value_cols = ["t2m","swvl1","swvl2","swvl3","swvl4","ssrd","pev","e","tp"]
    df_era = df_era.groupby("valid_time", as_index= False)[value_cols].mean()
    days = 30


    # Conversiones de unidades
    df_era["t2m"] -= 273.15
    for col in ["swvl1","swvl2","swvl3","swvl4"]:
        df_era[col] *= 100
    df_era["tp"] *= days * 1000
    df_era["e"]  = -df_era["e"] * days * 1000
    df_era["pev"] = -df_era["pev"] * days * 1000
    df_era["ssrd"] /= 86400.0

    # Cálculo SPI / SPEI
    #pr  = xr.DataArray(df_era["tp"].values,  coords={"time": df_era["valid_time"]}, dims="time")
    #pet = xr.DataArray(df_era["pev"].values, coords={"time": df_era["valid_time"]}, dims="time")
    #wb = pr - pet

    # Cálculo SPI / SPEI
    pr  = xr.DataArray(
        df_era["tp"].values,
        coords={"time": df_era["valid_time"]},
        dims="time",
        attrs={"units": "mm/month"}   # 👈 aquí agregamos las unidades
    )

    pet = xr.DataArray(
        df_era["pev"].values,
        coords={"time": df_era["valid_time"]},
        dims="time",
        attrs={"units": "mm/month"}   # 👈 lo mismo para PET
    )

    wb = pr - pet
    wb.attrs["units"] = "mm/month"    # 👈 y también para el balance hídrico


    #SPI  = xc.indices.spi
    #SPEI = xc.indices.spei

    # Compatibilidad con diferentes versiones de xclim
    SPI  = getattr(xc.indices, "spi",  getattr(xc.indices, "standardized_precipitation_index"))
    SPEI = getattr(xc.indices, "spei", getattr(xc.indices, "standardized_precipitation_evapotranspiration_index"))


    spi = {k: SPI(pr, window=k).to_series().rename(f"SPI_{k}") for k in [1,3,6,12]}
    spei = {k: SPEI(wb=wb, window=k).to_series().rename(f"SPEI_{k}") for k in [1,3,6,12]}

    df_era = df_era.set_index("valid_time").join(pd.concat([*spi.values(), *spei.values()], axis=1)).dropna().reset_index()

    # ---- Gráfico 1: Precipitación y Evaporación ----
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_era["valid_time"], y=df_era["e"], mode='lines', name="Evaporación Total"))
    fig1.add_trace(go.Scatter(x=df_era["valid_time"], y=df_era["tp"], mode='lines', name="Precipitación Total"))
    fig1.update_layout(
        title="Precipitación y Evaporación Total (mm/mes)",
        xaxis_title="Año",
        yaxis_title="Valor (mm/mes)",
        hovermode="x unified"
    )

    # ---- Gráfico 2: SPI ----
    fig2 = go.Figure()
    for k in [1,3,6,12]:
        fig2.add_trace(go.Scatter(x=df_era["valid_time"], y=df_era[f"SPI_{k}"], mode='lines', name=f"SPI_{k}"))
    fig2.update_layout(title="Índice de Precipitación Estandarizado (SPI)", hovermode="x unified")

    # ---- Gráfico 3: SPEI ----
    fig3 = go.Figure()
    for k in [1,3,6,12]:
        fig3.add_trace(go.Scatter(x=df_era["valid_time"], y=df_era[f"SPEI_{k}"], mode='lines', name=f"SPEI_{k}"))
    fig3.update_layout(title="Índice de Precipitación y Evapotranspiración Estandarizado (SPEI)", hovermode="x unified")

    # Mostrar las gráficas en pestañas
    tab1, tab2, tab3 = st.tabs(["🌧️ Precipitación / Evaporación", "📈 SPI", "🔥 SPEI"])
    with tab1:
        st.plotly_chart(fig1, use_container_width=True)
    with tab2:
        st.plotly_chart(fig2, use_container_width=True)
    with tab3:
        st.plotly_chart(fig3, use_container_width=True)


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
# 4) ANALISIS DE SEQUIAS ( ERA5 / SPI/ SPEI)
# =========================

# Lo de aqui se movio para arriba junto al chat bot

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
