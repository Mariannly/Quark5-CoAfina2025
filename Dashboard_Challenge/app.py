import streamlit as st
import pandas as pd
import altair as alt

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

    # Si viene con MultiIndex (year, valid_time, lat, lon), lo pasamos a columnas
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Fecha
    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df["date"] = df["valid_time"]
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        st.error("No se encontró columna de tiempo ('valid_time' o 'date').")
        st.stop()

    # Año
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year

    # Asegurar tp
    if "tp" not in df.columns:
        st.error("No se encontró la columna 'tp' en el dataset.")
        st.stop()

    # Serie mensual agregada (ej: promedio espacial)
    monthly = (
        df.groupby(pd.Grouper(key="date", freq="MS"))["tp"]
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
prob_sequia = 37  # TODO: reemplazar con tu predicción real

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
# ÚLTIMOS 12 MESES + 1 MES FUTURO (tp)
# =========================

last_12 = monthly.tail(12).copy()
last_date = last_12["date"].max()
next_month_date = last_date + pd.DateOffset(months=1)

# Predicción dummy: promedio últimos 12 meses
pred_tp = float(last_12["tp"].mean())

last_12["is_pred"] = 0
pred_row = pd.DataFrame(
    {"date": [next_month_date], "tp": [pred_tp], "is_pred": [1]}
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
                format="%b",      # etiquetas mensuales
                tickCount=13      # 12 meses + 1 futuro
            ),
        ),
        y=alt.Y("tp:Q", title="tp"),
        color=alt.condition(
            "datum.is_pred == 1",
            alt.value("#F44336"),   # predicción
            alt.value("#00d492"),   # histórico
        ),
        tooltip=["date:T", "tp:Q"],
    )
)

pred_points = (
    alt.Chart(plot_df[plot_df["is_pred"] == 1])
    .mark_point(size=80, filled=True)
    .encode(
        x="date:T",
        y="tp:Q",
        color=alt.value("#F44336"),
        tooltip=["date:T", "tp:Q"],
    )
)

st.subheader("Últimos 12 meses de 'tp' + proyección al siguiente mes")
st.altair_chart(band_chart + line_chart + pred_points, use_container_width=True)

st.markdown("---")

# =========================
# SECCIÓN 2:
# EXPLORACIÓN GENERAL (FILTROS + 2 GRÁFICAS + TABLA)
# =========================

st.subheader("Exploración histórica de variables climáticas")

st.sidebar.header("Filtros generales")

# Variables numéricas disponibles (excluimos year)
numeric_cols = [
    c for c in df.select_dtypes(include="number").columns
    if c != "year"
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

# Métricas
col1, col2, col3 = st.columns(3)
col1.metric("Media", f"{df_filtered[selected_var].mean():.4f}")
col2.metric("Mínimo", f"{df_filtered[selected_var].min():.4f}")
col3.metric("Máximo", f"{df_filtered[selected_var].max():.4f}")

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
    st.altair_chart(chart_yearly, use_container_width=True)

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
    st.altair_chart(hist, use_container_width=True)

# Tabla
st.markdown("**Muestra de datos filtrados**")
st.dataframe(df_filtered.head(200))
