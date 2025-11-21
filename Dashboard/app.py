import streamlit as st
import pandas as pd
import altair as alt
from google import genai
import os
import csv
from datetime import datetime
import plotly.graph_objects as go
import pickle
import numpy as np
import joblib
import pymannkendall as mk
from pathlib import Path
import requests

# COLORES
PALETTE = {
    "colors": ["#EB6D1E", "#14667B", "#7C1354", "#FB9637", "#B40424"],
    "widget_bg": "#AEC3AC",
    "text": "#061D23",
}

# =========================
# CONFIG PÁGINA
# =========================
st.set_page_config(
    page_title="Dashboard Sequías - Riohacha",
    page_icon="logo.png",
    layout="wide",
)

# =========================
# ESTILOS (se mantienen globales)
# =========================
st.markdown(
    """
    <style>
    /* Expanders como tarjetas internas con fondo de widget */
    details.st-expander, div.stExpander {
        background-color: #FDEBD8 !important;
        border-radius: 0.8rem !important;
        border: 1px solid #FDEBD8 !important;
    }

    /* Formularios (Buzón de reportes + Playground IA) */
    div[data-testid="stForm"] {
        background-color: #FDEBD8 !important;
        padding: 1rem 1rem 0.75rem 1rem !important;
        border-radius: 0.8rem !important;
        border: 1px solid #FDEBD8 !important;
    }

    /* Caja scroll del chatbot */
    div[data-testid="stVerticalBlock"] > div[style*="height: 350px"][style*="border: 1px solid"] {
        background-color: #FDEBD8 !important;
        border-radius: 0.8rem !important;
    }

    /* Hover para botones principales */
    div.stButton > button,
    button[kind="primary"] {
        transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
    }

    div.stButton > button:hover,
    button[kind="primary"]:hover {
        background-color: #E36414 !important;
        color: #FFFFFF !important;
        border-color: #E36414 !important;
    }

    /* Controles + y - de los number_input: base */
    div[data-testid="stNumberInput"] button,
    button[aria-label="Increment"],
    button[aria-label="Decrement"] {
        background-color: #FFFFFF !important;
        color: #E36414 !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
    }

    /* Hover number_input */
    div[data-testid="stNumberInput"] button:hover,
    button[aria-label="Increment"]:hover,
    button[aria-label="Decrement"]:hover {
        background-color: #E36414 !important;
        color: #FFFFFF !important;
        border-color: #E36414 !important;
        box-shadow: none !important;
    }

    /* Active / focus number_input */
    div[data-testid="stNumberInput"] button:active,
    button[aria-label="Increment"]:active,
    button[aria-label="Decrement"]:active,
    div[data-testid="stNumberInput"] button:focus,
    button[aria-label="Increment"]:focus,
    button[aria-label="Decrement"]:focus {
        background-color: #9A031E !important;
        color: #FFFFFF !important;
        border-color: #9A031E !important;
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HELPERS DESCARGA
# =========================
def download_http(url: str, dest_path: str) -> bool:
    try:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Error descargando {url}: {e}")
        return False

def download_from_s3(bucket: str, key: str, dest_path: str, aws_access_key=None, aws_secret_key=None, region_name=None) -> bool:
    try:
        import boto3
        session_kwargs = {}
        if aws_access_key and aws_secret_key:
            session_kwargs = dict(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region_name,
            )
        s3 = boto3.client("s3", **session_kwargs)
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))
        return True
    except Exception as e:
        st.error(f"Error descargando s3://{bucket}/{key}: {e}")
        return False

def ensure_asset(local_path: str, secret_key_url: str=None, s3_bucket_secret: str=None, s3_key_secret: str=None) -> bool:
    if Path(local_path).exists():
        return True

    if secret_key_url:
        url = None
        try:
            url = st.secrets.get(secret_key_url) if secret_key_url in st.secrets else None
        except Exception:
            url = None
        if url:
            st.info(f"Descargando {local_path} desde URL configurada en secret {secret_key_url}...")
            return download_http(url, local_path)

    try:
        s3_bucket = st.secrets.get(s3_bucket_secret) if s3_bucket_secret and s3_bucket_secret in st.secrets else None
        s3_key = st.secrets.get(s3_key_secret) if s3_key_secret and s3_key_secret in st.secrets else None
    except Exception:
        s3_bucket = s3_key = None

    if s3_bucket and s3_key:
        st.info(f"Descargando {local_path} desde S3 {s3_bucket}/{s3_key} usando credenciales en Secrets...")
        aws_key = st.secrets.get("AWS_ACCESS_KEY_ID") if "AWS_ACCESS_KEY_ID" in st.secrets else None
        aws_secret = st.secrets.get("AWS_SECRET_ACCESS_KEY") if "AWS_SECRET_ACCESS_KEY" in st.secrets else None
        aws_region = st.secrets.get("AWS_REGION") if "AWS_REGION" in st.secrets else None
        return download_from_s3(s3_bucket, s3_key, local_path, aws_key, aws_secret, aws_region)

    st.warning(f"No se encontró '{local_path}' localmente y no se configuró una URL o S3 en Secrets para descargarlo.")
    return False

# Descarga condicional de assets
_ = ensure_asset(
    local_path="dataset_clima.parquet",
    secret_key_url="DATASET_URL",
    s3_bucket_secret="S3_BUCKET",
    s3_key_secret="DATASET_KEY",
)

_ = ensure_asset(
    local_path="modelo_sequia_hgb.pkl",
    secret_key_url="MODEL_URL",
    s3_bucket_secret="S3_BUCKET",
    s3_key_secret="MODEL_KEY",
)

# =========================
# CARGA Y PREPARACIÓN DE DATOS
# =========================
@st.cache_data
def load_data():
    df = pd.read_parquet("Dashboard/dataset_clima.parquet")

    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df["date"] = df["valid_time"]
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        st.error("No se encontró columna de tiempo ('valid_time' o 'date') en el dataset.")
        st.stop()

    df["year"] = df["date"].dt.year

    if "tp" not in df.columns:
        st.error("No se encontró la columna 'tp' (Precipitación total) en el dataset.")
        st.stop()

    monthly = (
        df.groupby(pd.Grouper(key="date", freq="MS"))["tp"]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    return df, monthly

@st.cache_data
def load_modelo_probs():
    try:
        dfm = pd.read_parquet("Dashboard/dataset_modelo.parquet")

        if "valid_time" not in dfm.columns:
            st.error("El archivo 'dataset_modelo.parquet' debe contener la columna 'valid_time'.")
            return None
        if "proba" not in dfm.columns:
            st.error("El archivo 'dataset_modelo.parquet' debe contener la columna 'proba'.")
            return None

        dfm["valid_time"] = pd.to_datetime(dfm["valid_time"])

        monthly = (
            dfm.resample("MS", on="valid_time")["proba"]
            .mean()
            .reset_index()
            .rename(columns={"valid_time": "date"})
            .sort_values("date")
        )

        return monthly

    except FileNotFoundError:
        st.warning("No se encontró 'dataset_modelo.parquet'. No se mostrará la gráfica de probabilidades de sequía.")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load("Dashboard/modelo_sequia_hgb.pkl")
        return model
    except Exception as e:
        st.error(f"No se pudo cargar el modelo de sequía desde 'modelo_sequia_hgb.pkl'. Detalle: {e}")
        return None

model = load_model()
df, monthly = load_data()

if monthly.empty:
    st.error("No hay datos mensuales disponibles.")
    st.stop()

# =========================
# PRE-CÁLCULOS COMPARTIDOS
# =========================
prob_sequia = 37  # placeholder

monthly_probs = load_modelo_probs()

numeric_cols = [
    c for c in df.select_dtypes(include="number").columns
    if c not in ["year"]
]

if not numeric_cols:
    st.error("No se encontraron variables numéricas para visualizar.")
    st.stop()

year_min = int(df["year"].min())
year_max = int(df["year"].max())

selected_var = "tp" if "tp" in numeric_cols else numeric_cols[0]
start_year, end_year = year_min, year_max

mask = (df["year"] >= start_year) & (df["year"] <= end_year)
df_filtered = df[mask]

if df_filtered.empty:
    st.warning("No hay datos para ese rango de años.")
    st.stop()

# =========================
# PESTAÑAS PRINCIPALES
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Inicio & riesgo actual",
    "📊 Análisis climático & asistente",
    "🧪 Playground IA",
    "📬 Buzón de reportes",
])

# =========================
# TAB 1: Título + logo + primera gráfica + notas
# =========================
with tab1:
    # ===== Texto inicial =====
    st.markdown(
    """
    # **Bienvenido al Sistema de Alerta y Riesgo por Incidencia de Sequías y Desastres Ambientales (S-ARIDA)**  
    """
    )

    # Descripción con tamaño personalizado
    st.markdown(
        """
        <p style="font-size: 1.3rem; line-height: 1.6;">
            Una herramienta que convierte datos climáticos en conocimiento útil para enfrentar las sequías e incendios en La Guajira.
        </p>
        <p style="font-size: 1.3rem; line-height: 1.6;">
            Aquí podrás explorar visualizaciones intuitivas, conocer proyecciones sobre el riesgo climático
            y conversar con un asistente inteligente que explica, de forma clara y sencilla, lo que muestran los datos
            para apoyar la prevención y el cuidado del territorio.
        </p>
        """,
        unsafe_allow_html=True,
    )
    # ===== Logo centrado =====
    logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 1])
    with logo_col2:
        st.image("Dashboard/logo.png", use_container_width=True)

    st.markdown("---")
    st.header("**Probabilidad de sequía estimada**")

    # Usamos el mismo loader que ya definiste
    monthly_probs = load_modelo_probs()

    if monthly_probs is not None and not monthly_probs.empty:
        df_disp = monthly_probs.copy()

        # Si 'proba' está entre 0 y 1, la pasamos a porcentaje
        df_disp["proba_pct"] = df_disp["proba"] * 100

        MESES_ES = {
            1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
            5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
            9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
        }

        df_disp["hover_fecha"] = df_disp["date"].apply(
            lambda d: f"{MESES_ES[d.month]} {d.year}"
        )

        # ===== Gráfica probabilidad (a todo el ancho) =====
        fig_model = go.Figure()
        fig_model.update_layout(colorway=PALETTE["colors"])
        fig_model.add_trace(go.Scatter(
            x=df_disp["date"],
            y=df_disp["proba_pct"],
            mode="lines+markers",
            name="Probabilidad de sequía",
            customdata=df_disp["hover_fecha"],
            hovertemplate="Fecha: %{customdata}<br>Probabilidad: %{y:.1f}%<extra></extra>",
        ))
        fig_model.add_hrect(y0=0.0, y1=33, opacity=0.2, fillcolor="#0F4C5C", line_width=0, layer="below")
        fig_model.add_hrect(y0=33, y1=50, opacity=0.2, fillcolor="#FB8B24", line_width=0, layer="below")
        fig_model.add_hrect(y0=50, y1=70, opacity=0.2, fillcolor="#E36414", line_width=0, layer="below")
        fig_model.add_hrect(y0=70, y1=90, opacity=0.2, fillcolor="#9A031E", line_width=0, layer="below")
        fig_model.add_hrect(y0=90, y1=100, opacity=0.2, fillcolor="#5F0F40", line_width=0, layer="below")

        fig_model.update_layout(
            title="Evolución mensual de la probabilidad de sequía según el modelo",
            xaxis_title="Año",
            yaxis_title="Probabilidad de sequía (%)",
            hovermode="x unified",
            xaxis=dict(
                tickformat="%Y",
                dtick="M12",
                rangeslider=dict(visible=True)
            ),
        )

        st.plotly_chart(fig_model, use_container_width=True)

        st.caption(
            "Cada punto representa la probabilidad estimada de sequía para un mes específico. "
            "El eje horizontal muestra años como referencia general, pero puedes usar el control "
            "inferior para acercarte y ver la variación mes a mes."
        )

        st.markdown("---")

        # ===== Recomendaciones + Chatbot en columnas =====
        st.subheader("Acciones recomendadas y asistencia interactiva")

        rec_col, chat_col = st.columns([2, 1])

        # ----- Columna izquierda: Recomendaciones -----
        with rec_col:
            st.header("**Recomendaciones para la prevención y manejo de sequías**")
            st.markdown("Consulta recomendaciones específicas según tu rol o nivel de responsabilidad.")

            col_inst, col_com = st.columns(2)

            # TARJETA: INSTITUCIONES
            with col_inst:
                st.markdown(
                    """
                    <div style="
                        padding: 1.2rem;
                        border: 1px solid #e7e3e4;
                        border-radius: 0.8rem;
                        background-color: #FDEBD8;
                        min-height: 130px;
                        margin-bottom: 0.6rem;
                    ">
                        <h3 style="margin-top: 0; margin-bottom: 0.4rem;">🏛️ PARA INSTITUCIONES</h3>
                        <p style="margin: 0; font-size: 0.9rem; color: #555;">
                            Lineamientos para entidades públicas, operadores de acueducto,
                            autoridades ambientales y de gestión del riesgo.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("🌤️ Antes de la sequía"):
                    st.markdown(
                        """
                        - Elaborar y socializar **planes locales de gestión del agua y sequías**.  
                        - Actualizar el **inventario de fuentes hídricas y reservas subterráneas**.  
                        - Promover campañas de **educación para el ahorro del agua**.  
                        - Implementar **sistemas de monitoreo y alerta temprana** con datos climáticos.  
                        - Coordinar con **IDEAM, UNGRD y acueductos locales** para alertas preventivas.  
                        - Reforestar **cuencas y zonas de recarga hídrica** estratégicas.
                        """
                    )

                with st.expander("☀️ Durante la sequía"):
                    st.markdown(
                        """
                        - Activar los **planes de emergencia hídrica**, priorizando agua potable.  
                        - Garantizar **distribución equitativa** (carrotanques, puntos oficiales).  
                        - Emitir **comunicados frecuentes, claros y verificables**.  
                        - Monitorear **riesgo de incendios** y restringir usos no esenciales del agua.  
                        - Apoyar la **atención en salud** por golpes de calor y enfermedades asociadas.
                        """
                    )

                with st.expander("🌧️ Después de la sequía"):
                    st.markdown(
                        """
                        - Evaluar **impactos ambientales, agrícolas y sociales**.  
                        - Promover **restauración de ecosistemas y recarga hídrica**.  
                        - Impulsar tecnologías de **captación de agua lluvia y eficiencia hídrica**.  
                        - Actualizar **POT y planes locales** considerando vulnerabilidad hídrica.  
                        - Fortalecer la **educación climática y participación ciudadana**.
                        """
                    )

            # TARJETA: COMUNIDAD
            with col_com:
                st.markdown(
                    """
                    <div style="
                        padding: 1.8rem;
                        border: 1px solid #e7e3e4;
                        border-radius: 0.8rem;
                        background-color: #FDEBD8;
                        min-height: 130px;
                        margin-bottom: 0.6rem;
                    ">
                        <h3 style="margin-top: 0; margin-bottom: 0.4rem;">👥 PARA LA COMUNIDAD</h3>
                        <p style="margin: 0; font-size: 0.9rem; color: #555;">
                            Acciones prácticas para hogares, barrios, líderes comunitarios y productores locales.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("🌤️ Antes de la sequía"):
                    st.markdown(
                        """
                        - Usar el agua **racionalmente**: reparar fugas y recolectar agua lluvia.  
                        - Evitar **quemas** o fogatas en zonas rurales y secas.  
                        - Participar en **reforestación** y cuidado de fuentes hídricas.  
                        - Identificar **fuentes de agua cercanas y rutas de abastecimiento**.  
                        - Seguir información de **canales oficiales** (IDEAM, alcaldías, Defensa Civil).
                        """
                    )

                with st.expander("☀️ Durante la sequía"):
                    st.markdown(
                        """
                        - Priorizar agua para **consumo, higiene y alimentación**.  
                        - Evitar riegos, lavado de vehículos y usos no esenciales.  
                        - No talar ni quemar vegetación; reducir riesgo de incendios.  
                        - Mantener recipientes tapados y limpios para evitar vectores.  
                        - Reportar **fugas** o uso indebido del agua pública.  
                        - Proteger animales y cultivos con **sombra y almacenamiento adecuado**.
                        """
                    )

                with st.expander("🌧️ Después de la sequía"):
                    st.markdown(
                        """
                        - Participar en **jornadas de reforestación y recuperación del suelo**.  
                        - Colaborar en **evaluaciones comunitarias** sobre la respuesta a la sequía.  
                        - Mantener **hábitos sostenibles** de uso del agua.  
                        - Promover **educación ambiental** en familia, escuelas y barrios.  
                        - Proteger fuentes naturales y **denunciar impactos negativos** sobre ellas.
                        """
                    )

        # ----- Columna derecha: Chatbot -----
        with chat_col:
            st.header("**💬 Asistente climático**")

            api_key = "AIzaSyD1CSvFFpJdKSkbpza60zqHKyKc-oO0nYQ"

            if not api_key:
                st.warning("Configura GEMINI_API_KEY para habilitar el chatbot.")
            else:
                client = genai.Client(api_key=api_key)

                if "chat_messages" not in st.session_state:
                    st.session_state.chat_messages = [
                        {
                            "role": "assistant",
                            "content": (
                                "Hola 👋 Soy tu asistente climático basado en Gemini. "
                                "Puedo ayudarte a entender todos los datos que ves aquí y darte información útil y educativa sobre las sequías y el cambio climático."
                            ),
                        }
                    ]

                user_input = st.chat_input("Escribe tu pregunta aquí")

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
                            "INFORMACIÓN CONCEPTUAL SOBRE LAS SEQUÍAS Y CAMBIO CLIMÁTICO\n\n"
                            "SEQUÍA METEOROLÓGICA: Ausencia prolongada o escasez acusada de precipitación.\n"
                            "Sequía hidrológica (déficit hídrico): Período de tiempo anormalmente seco, lo suficientemente prolongado "
                            "para ocasionar una escasez de agua, que se refleja en una disminución apreciable en el caudal de los ríos "
                            "y en el nivel de los lagos y/o en el agotamiento de la humedad del suelo y el descenso de los niveles de aguas "
                            "subterráneas por debajo de sus valores normales.\n"
                            "CAMBIO CLIMÁTICO: alteración significativa y persistente de las propiedades estadísticas del sistema climático "
                            "(principalmente su promedio y dispersión) durante periodos largos de tiempo, y puede ser causado tanto por procesos "
                            "naturales como principalmente por actividades humanas que modifican la composición de la atmósfera. Según la Convención "
                            "Marco de las Naciones Unidas sobre el Cambio Climático (CMNUCC), se trata de un cambio de clima atribuido directa o "
                            "indirectamente a la actividad humana, distinguiéndose de la mera variabilidad climática natural. Fuente: Wikipedia y "
                            "cambioclimatico.gov.co.\n"
                            "El cambio climático está intensificando los periodos de sequía y lluvia a nivel global. Las sequías actuales son más "
                            "frecuentes, extensas y prolongadas, mientras que los periodos lluviosos muestran precipitaciones más extremas e irregulares. "
                            "El aumento de temperaturas incrementa la evaporación del suelo y la evapotranspiración de las plantas, disminuyendo el agua "
                            "disponible y agravando la aridificación de los climas. En consecuencia, los años húmedos son menos húmedos y los secos son "
                            "mucho más secos.\n"
                            "Las zonas ubicadas en el ecuador y los trópicos experimentan con mayor rapidez y severidad los efectos del cambio climático. "
                            "Por ejemplo, en Ecuador y países tropicales, se observan cambios notorios en los patrones de precipitación: hay una alternancia "
                            "entre sequías intensas y lluvias torrenciales, lo que da lugar a deslizamientos de tierra, alteraciones en la agricultura y "
                            "pérdida significativa de cultivos. Además, los eventos extremos como El Niño y La Niña, influidos por el calentamiento global, "
                            "modifican las temporadas tradicionales de lluvias y sequías, volviéndolas más impredecibles y acentuando sus impactos sociales "
                            "y ecológicos. Fuente: https://www.wwfca.org/nuestrotrabajo/clima_energia/impacto_cambio_climatico_latinoamerica , "
                            "https://www.agenciasinc.es/Noticias/Las-areas-tropicales-sufriran-antes-los-efectos-del-cambio-climatico.\n"
                            "El cambio climático altera la duración, intensidad y periodicidad de las temporadas de lluvia y sequía. En muchas regiones "
                            "ecuatoriales y tropicales, las lluvias intensas pueden concentrarse en periodos más cortos y las sequías prolongarse, generando "
                            "desafíos para la gestión del agua y la seguridad alimentaria. Estas modificaciones pueden afectar de manera directa a sectores "
                            "vulnerables como la agricultura, la biodiversidad y las poblaciones rurales, incrementando los riesgos de desastres naturales y "
                            "desplazamientos humanos.\n\n"
                            "EFECTOS E IMPACTOS DE LAS SEQUÍAS\n"
                            "- Deshidratación poblacional, animal y vegetal: impacto en población, cultivos y ganado.\n"
                            "- Impacto directo en abastecimiento alimentario por afectación de cultivos.\n"
                            "- Incendios forestales por baja humedad y resequedad del suelo más radiación solar fuerte y temperaturas altas.\n"
                            "- Escasez de agua en fuentes hídricas: desabastecimiento de acueductos y pozos, afectación de higiene y saneamiento, aumento del uso de agua no potable "
                            "y aparición de enfermedades en personas y animales (gastrointestinales, dérmicas, desnutrición, especialmente en NNA).\n"
                            "- Bajísima humedad y altas temperaturas: golpes de calor, insolación, deshidratación severa, afectación a personas con condiciones de salud previas.\n"
                            "Desplazamiento por sequías: La falta de agua para consumo y agricultura lleva a la migración temporal o permanente, especialmente en áreas rurales y zonas "
                            "áridas. Las sequías, exacerbadas por el cambio climático, afectan la disponibilidad de agua, la producción agrícola y la seguridad alimentaria, lo que puede "
                            "forzar a las personas a abandonar sus hogares en busca de mejores condiciones de vida. Según el IDMC, en 2022 se registraron 31,8 millones de desplazamientos "
                            "internos por fenómenos meteorológicos extremos a nivel global. Las sequías fueron la tercera causa principal, tras inundaciones y tormentas.\n\n"
                            "SOBRE NUESTROS INDICADORES Y DATOS\n"
                            "Índice Estandarizado de Precipitación y Evapotranspiración (SPEI): propuesto por Vicente-Serrano et al. (2010) como índice de sequía mejorado. "
                            "Utiliza el balance hídrico climático (precipitación menos evapotranspiración de referencia), en distintas escalas de tiempo, proporcionando una medida "
                            "robusta de la gravedad de la sequía.\n"
                            "Cálculo SPEI: Los valores de P - ETo se ajustan a una distribución de probabilidad para transformarlos a unidades estandarizadas. Se recomienda la "
                            "distribución Loglogística (Vicente-Serrano et al., 2010), adecuada para diferentes escalas y climas. Luego se normalizan los datos.\n\n"
                            "DIFERENCIAS ENTRE INDICADORES E ÍNDICES\n"
                            "Indicadores: variables usadas para describir condiciones de sequía (precipitación, temperatura, humedad del suelo, caudal de ríos, niveles de agua subterránea, etc.).\n"
                            "Índices: representaciones numéricas de la severidad de la sequía construidas a partir de indicadores (como SPEI), que simplifican relaciones complejas y permiten "
                            "evaluar intensidad, ubicación, tiempo y duración.\n\n"
                            "IMPORTANCIA DE ESTA INFORMACIÓN\n"
                            "Comprender cómo el cambio climático altera sequías y lluvias es clave para la gestión sostenible del agua, la planificación agrícola, el diseño de infraestructuras "
                            "resilientes y la formulación de políticas públicas. La anticipación y monitoreo permiten reducir pérdidas humanas, económicas y ecológicas, especialmente en zonas "
                            "vulnerables del ecuador y el trópico.\n\n"
                            "Eres un experto en climatología y prevención de desastres naturales del Instituto de Hidrología, Meteorología y Estudios Ambientales de Colombia, pero también experto en divulgación científica y ciencia ciudadana, con mucha experiencia para compartir con funcionarios gubernamentales y población civil información que puede resultar compleja, haciéndola accesible para este público, pero que procura ceñirse a la información científica verificable y evitando a toda costa recaer en la desinformación o especulación. "
                            "Tus respuestas serán dadas en un tono educativo, confiable y claro, NO TÉCNICO. "
                            "Toma la información contextual suministrada a continuación para extraer y aprovechar el contenido, estableciendo relaciones conceptuales, contextuales y con los datos suministrados para responder de manera clara, eficiente, accesible y completa. "
                            "Busca siempre primero la respuesta a la pregunta dentro de la información ya suministrada, y como último recurso en caso de no encontrar nada relacionado, sólo entonces haz una búsqueda web muy puntual y toma la fuente más fiable de información desde una perspectiva científica para responder, complementando la información que ya se tenía y retroalimentándola para volver a la información inicial y su importancia. "
                            "Al buscar información de fuentes externas, priorizar siempre instituciones nacionales oficiales como el IDEAM, el Ministerio de Ambiente, el Ministerio de Agricultura, Corpoguajira, la Cruz Roja y la FAO, en ese orden, y secundariamente otras instituciones como ONGs especializadas en la problemática. "
                            "Al dar la respuesta, no explicites tu posición de enunciación como investigador ni como experto, limítate a dar una respuesta acorde a la pregunta planteada: algo informativo pero sucinto que responda bien a lo solicitado, sin añadir información extra innecesaria que no esté relacionada directamente con ello. "
                            "Sin embargo, puedes sugerir al final una pregunta de profundización o seguimiento en el tema. Por ejemplo, para la pregunta \"¿qué consecuencias tiene la sequía?\" la respuesta puede hablar brevemente de las consecuencias y efectos inmediatos de la sequía, mencionar que hay diferentes tipos de consecuencias (ambientales, sociales, poblacionales, de salud, etc.) y cerrar con algo como: "
                            "\"¿Quieres que te cuente más sobre alguno de estos aspectos en particular?\" "
                            "No atiborres de información: deja que las personas pregunten más por su cuenta.\n\n"
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

                # Contenedor scrollable
                chat_box = st.container()
                with chat_box:
                    messages_html = """
                    <div style="
                        background-color:#FDEBD8;
                        padding:0.75rem;
                        border-radius:0.8rem;
                        border:1px solid #FDEBD8;
                        height:350px;
                        overflow-y:auto;
                    ">
                    """
                    for msg in st.session_state.chat_messages:
                        if msg["role"] == "user":
                            messages_html += f"<p>🧑‍💻 <b>Tú:</b> {msg['content']}</p>"
                        else:
                            messages_html += f"<p>🤖 <b>Asistente:</b> {msg['content']}</p>"
                    messages_html += "</div>"

                    st.markdown(messages_html, unsafe_allow_html=True)

    else:
        st.info("No se pudo cargar información válida desde 'dataset_modelo.parquet' para esta gráfica.")


# =========================
# TAB 2: Otras gráficas + chatbot + análisis tendencias
# =========================
with tab2:
    # ===== Análisis climático ERA5-Land (Mensual) =====
    st.header("**Análisis climático basado en ERA5-Land (Mensual)**")

    # Gráfico 1: Precipitación vs Evaporación
    fig1 = go.Figure()
    fig1.update_layout(colorway=PALETTE["colors"])
    if "e" in df_filtered.columns:
        fig1.add_trace(go.Scatter(
            x=df_filtered["date"],
            y=df_filtered["e"],
            mode="lines",
            name="Evaporación total (e)"
        ))

    if "tp" in df_filtered.columns:
        fig1.add_trace(go.Scatter(
            x=df_filtered["date"],
            y=df_filtered["tp"],
            mode="lines",
            name="Precipitación total (tp)"
        ))

    fig1.update_layout(
        title="Precipitación vs Evaporación total (mm/mes)",
        xaxis_title="Año",
        yaxis_title="Precipitación y Evaporación Total (mm/mes)",
        hovermode="x unified"
    )
    # Gráfico 2: SPI
    fig2 = go.Figure()
    fig2.update_layout(colorway=PALETTE["colors"])
    spi_cols = [c for c in ["SPI_1", "SPI_3", "SPI_6", "SPI_12"] if c in df_filtered.columns]

    if spi_cols:
        for c in spi_cols:
            fig2.add_trace(go.Scatter(
                x=df_filtered["date"],
                y=df_filtered[c],
                mode="lines",
                name=c
            ))
        fig2.update_layout(
            title="Índice de Precipitación Estandarizado (SPI)",
            xaxis_title="Año",
            yaxis_title="SPI",
            hovermode="x unified"
        )
    else:
        fig2.update_layout(title="SPI no disponible en el dataset")

    # Gráfico 3: SPEI
    fig3 = go.Figure()
    fig3.update_layout(colorway=PALETTE["colors"])
    spei_cols = [c for c in ["SPEI_1", "SPEI_3", "SPEI_6", "SPEI_12"] if c in df_filtered.columns]

    if spei_cols:
        for c in spei_cols:
            fig3.add_trace(go.Scatter(
                x=df_filtered["date"],
                y=df_filtered[c],
                mode="lines",
                name=c
            ))
        fig3.update_layout(
            title="Índice SPEI (Precipitación - Evapotranspiración)",
            xaxis_title="Año",
            yaxis_title="SPEI",
            hovermode="x unified"
        )
    else:
        fig3.update_layout(title="SPEI no disponible en el dataset")

    t1, t2, t3 = st.tabs(["🌧️ Precipitación / Evaporación", "📈 SPI", "🔥 SPEI"])
    with t1:
        st.plotly_chart(fig1, use_container_width=True)
    with t2:
        st.plotly_chart(fig2, use_container_width=True)
    with t3:
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.header("**Análisis de tendencias de sequías (Mann-Kendall)**")

    if all(col in df.columns for col in [
        'SPI_1', 'SPI_3', 'SPI_6', 'SPI_12',
        'SPEI_1', 'SPEI_3', 'SPEI_6', 'SPEI_12'
    ]):
        # ==== Cálculo tendencias (se mantiene igual) ====
        trend_data = []
        for col in ['SPI_1', 'SPI_3', 'SPI_6', 'SPI_12',
                    'SPEI_1', 'SPEI_3', 'SPEI_6', 'SPEI_12']:
            series = df[col].dropna()
            if not series.empty:
                result = mk.original_test(series)
                trend_data.append({
                    'Index': col,
                    'Slope': result.slope,
                    'P_Value': result.p,
                    'Trend': result.trend,
                    'Significant': result.p < 0.05
                })

        trend_df = pd.DataFrame(trend_data)

        slope_spei12_from_df = trend_df[trend_df['Index'] == 'SPEI_12']['Slope'].iloc[0]
        numeric_index_for_slope = np.arange(len(df['SPEI_12']))
        first_spei12_value_in_series = df['SPEI_12'].iloc[0]
        trend_line_y = first_spei12_value_in_series + slope_spei12_from_df * (
            numeric_index_for_slope - numeric_index_for_slope[0]
        )

        # ==== 1) PRIMERO: Gráfica recortada + eventos + observaciones (antes del contexto histórico) ====
        n = min(150, len(df))
        x_last = df["valid_time"].iloc[-n:]
        spei_last = df["SPEI_12"].iloc[-n:]
        trend_last = trend_line_y[-n:]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=x_last,
            y=spei_last,
            mode="lines",
            name="SPEI (k=12 meses)"
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_last,
            y=trend_last,
            mode="lines",
            name="Tendencia (Mann-Kendall)",
            line=dict(color="#9A031E", dash="dash")
        ))

        # Franjas de eventos históricos
        fig_hist.add_vrect(
            x0=pd.to_datetime("2021-02-01"),
            x1=pd.to_datetime("2021-02-28"),
            line_width=0,
            fillcolor="#9A031E",
            opacity=0.18,
            layer="below",
        )
        fig_hist.add_vrect(
            x0=pd.to_datetime("2020-01-01"),
            x1=pd.to_datetime("2020-06-30"),
            line_width=0,
            fillcolor="#9A031E",
            opacity=0.18,
            layer="below",
        )
        fig_hist.add_vrect(
            x0=pd.to_datetime("2019-03-01"),
            x1=pd.to_datetime("2019-04-30"),
            line_width=0,
            fillcolor="#9A031E",
            opacity=0.18,
            layer="below",
        )
        fig_hist.add_vrect(
            x0=pd.to_datetime("2019-02-01"),
            x1=pd.to_datetime("2019-02-28"),
            line_width=0,
            fillcolor="#9A031E",
            opacity=0.18,
            layer="below",
        )
        fig_hist.add_vrect(
            x0=pd.to_datetime("2014-01-01"),
            x1=pd.to_datetime("2014-12-30"),
            line_width=0,
            fillcolor="#9A031E",
            opacity=0.18,
            layer="below",
        )

        fig_hist.update_layout(
            title="Eventos históricos recientes y tendencia de sequía (SPEI_12, últimos registros)",
            xaxis_title="Año",
            yaxis_title="SPEI (k=12 meses)",
            hovermode="x unified",
            colorway=["#3DA5D9"],
        )
        fig_hist.update_xaxes(rangeslider_visible=True)

        st.plotly_chart(fig_hist, use_container_width=True)

        # 🔍 Observaciones clave (pegadas a esta última gráfica)
        st.markdown(
            """
            <div style="background-color:#FDEBD8; padding:1rem; border-radius:0.8rem;">
            <h3>🔍 Observaciones clave:</h3>
            <ul>
                <li>Los registros confirman una <b>recurrencia de eventos de sequía severa cada 3–5 años</b>, con picos asociados al <b>Fenómeno del Niño</b>.</li>
                <li>El <b>déficit hídrico del río Tapias</b> es un indicador crítico para Riohacha y comunidades Wayúu.</li>
                <li>Los impactos sociales concuerdan con las <b>anomalías de precipitación y temperatura</b> observadas.</li>
                <li>Desde 2020 se observa <b>mayor irregularidad estacional</b>, probablemente vinculada al cambio climático global.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ==== 2) Luego: Contexto histórico (texto + tabla) ====
        st.subheader("📅 Contexto histórico y observacional de sequías en La Guajira")

        st.markdown(
            """
            <div style="background-color:#FDEBD8; padding:1rem; border-radius:0.8rem;">
            Además del análisis climático cuantitativo con datos ERA5, se recopilaron reportes de prensa y boletines institucionales
            que reflejan los impactos sociales y ambientales de las sequías recientes en La Guajira.
            Estos eventos permiten validar el comportamiento observado en los índices de sequía y comprender mejor las afectaciones locales.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            | Fecha | Evento reportado | Fuente / Observación |
            |--------|------------------|----------------------|
            | **5 de junio de 2025** | Temporada de lluvias irregular, lluvias por debajo del promedio. | *Periódicos locales* |
            | **1 de junio – 30 de noviembre (2025)** | Temporada de ciclones tropicales que incrementa la variabilidad climática. | *Servicio Meteorológico Nacional* |
            | **1er semestre de 2024** | Fenómeno del Niño afectó a más de 5.500 familias en varios municipios. | *OCHA* |
            | **Febrero de 2021** | Río Tapias presentó 1.300 L/s menos de su caudal normal. | *Prensa regional* |
            | **Enero de 2020** | Calamidad pública en Hatonuevo por escasez de agua. | *Noticias locales* |
            | **Febrero de 2019** | Disminución del nivel del río Tapia en más del 50%. | *Prensa regional* |
            | **2014** | Año de sequía extrema con afectaciones prolongadas. | *Archivo de prensa nacional* |
            """
        )

        # ==== 3) Finalmente: Gráfica completa SPEI_12 + tendencia MK (la que antes iba primero) ====
        fig_trend = go.Figure()
        fig_trend.update_layout(colorway=PALETTE["colors"])
        fig_trend.add_trace(go.Scatter(
            x=df["valid_time"],
            y=df['SPEI_12'],
            mode='lines',
            name='SPEI (k=12 meses)'
        ))
        fig_trend.add_trace(go.Scatter(
            x=df["valid_time"],
            y=trend_line_y,
            mode='lines',
            name='Tendencia Mann-Kendall',
            line=dict(color='red', dash='dash')
        ))
        fig_trend.update_layout(
            title='SPEI_12 con Línea de Tendencia Mann-Kendall (serie completa)',
            xaxis_title='Año',
            yaxis_title='Valor de SPEI (k=12 meses)',
            hovermode='x unified'
        )
        fig_trend.update_xaxes(rangeslider_visible=True)

        st.plotly_chart(fig_trend, use_container_width=True)

    else:
        st.info("⚠️ Aún no se han calculado los índices SPI/SPEI necesarios para el análisis de tendencias.")

# =========================
# TAB 3: Playground / simulador IA
# =========================
with tab3:
    st.header("**Playground para expertos**")

    if model is None:
        st.info("El modelo de IA no está disponible en este momento.")
    else:
        last_row = df.sort_values("date").iloc[-1]

        left_col, right_col = st.columns([1, 2])

        with left_col:
            st.markdown(
                """
                <div style="background-color:#FDEBD8; padding:1rem; border-radius:0.8rem;">
                <p><b>¡Bienvenido a nuestro simulador predictivo!</b> En él, podrás ingresar diferentes valores para cada uno de los índices e indicadores que alimentan nuestro modelo, para así tener una idea del riesgo de sequía según cómo se comportan las diferentes variables climatológicas.</p>

                <p><b>¿Cómo usar esta herramienta?</b><br>
                Ingresa los valores climáticos mensuales observados o estimados para tu zona.
                Con estas variables, el modelo de IA calcula la probabilidad de que se presenten
                condiciones compatibles con sequía.</p>

                <p><b>Interpretación del resultado</b><br>
                - Se muestra una probabilidad estimada de sequía.<br>
                - Además, se indica si, según el modelo, las condiciones corresponden o no
                a un posible episodio de sequía.<br>
                Esta sección está pensada para apoyar la toma de decisiones,
                comunicación de riesgos y análisis exploratorio.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right_col:
            with st.form("form_prediccion_sequia"):
                c1, c2 = st.columns(2)

                with c1:
                    t2m_input = st.number_input(
                        "t2m - Temperatura 2 m (°C)",
                        value=float(last_row.get("t2m", 25.0)),
                        format="%.8f"
                    )
                    st.caption("Más calor = más sed del aire.")

                    swvl1_input = st.number_input(
                        "swvl1 - Humedad del suelo capa 1 (mm3)",
                        value=float(last_row.get("swvl1", 0.0)),
                        format="%.8f"
                    )
                    st.caption("Reserva muy superficial; responde rápido a falta de lluvia, la primera en evaporarse.")

                    swvl2_input = st.number_input(
                        "swvl2 - Humedad del suelo capa 2 (mm3)",
                        value=float(last_row.get("swvl2", 0.0)),
                        format="%.8f"
                    )
                    st.caption("Reserva poco profunda; sostiene los cultivos durante algunos días/semanas.")

                    swvl3_input = st.number_input(
                        "swvl3 - Humedad del suelo capa 3 (mm3)",
                        value=float(last_row.get("swvl3", 0.0)),
                        format="%.8f"
                    )
                    st.caption("Reserva profunda; de verse afectada negativamente, refleja una sequía más persistente.")

                    swvl4_input = st.number_input(
                        "swvl4 - Humedad del suelo capa 4 (mm3)",
                        value=float(last_row.get("swvl4", 0.0)),
                        format="%.8f"
                    )
                    st.caption("Reserva muy profunda; cuando baja, también sufren ríos, embalses y cuencas hídricas.")

                with c2:
                    ssrd_input = st.number_input(
                        "ssrd - Radiación solar hacia abajo (MJ/m²/día)",
                        value=float(last_row.get("ssrd", 0.0)),
                        format="%.8f"
                    )
                    st.caption("A mayor radiación, más energía para evaporar agua.")

                    pev_input = st.number_input(
                        "pev - Evaporación potencial (mm/mes)",
                        value=float(last_row.get("pev", 0.0)),
                        format="%.8f"
                    )
                    st.caption("La 'sed' del aire según calor, sol y viento.")

                    e_input = st.number_input(
                        "e - Evaporación total (mm/mes)",
                        value=float(last_row.get("e", 0.0)),
                        format="%.8f"
                    )
                    st.caption("Lo que realmente se evapora y transpiran las plantas.")

                    tp_input = st.number_input(
                        "tp - Precipitación total (mm/mes)",
                        value=float(last_row.get("tp", 0.0)),
                        format="%.8f"
                    )
                    st.caption("Cantidad de agua que cae con la lluvia en el mes.")

                submitted = st.form_submit_button("Calcular probabilidad de sequía")

                if submitted:
                    X_input = np.array([[
                        t2m_input,
                        swvl1_input,
                        swvl2_input,
                        swvl3_input,
                        swvl4_input,
                        ssrd_input,
                        pev_input,
                        e_input,
                        tp_input,
                    ]])

                    try:
                        if hasattr(model, "predict_proba"):
                            prob = float(model.predict_proba(X_input)[0][1])
                            pred_class = int(model.predict(X_input)[0])
                        else:
                            pred_class = int(model.predict(X_input)[0])
                            prob = None

                        col_res1, col_res2 = st.columns(2)

                        with col_res1:
                            if prob is not None:
                                st.metric(
                                    "Probabilidad estimada de sequía",
                                    f"{prob * 100 :.8f}%"
                                )
                            else:
                                st.write("El modelo no expone `predict_proba`, solo la clase predicha.")

                        with col_res2:
                            if pred_class == 1:
                                st.markdown(
                                    "<div style='padding:0.6rem; border-radius:0.5rem; background-color:#ffe5e5;'>"
                                    "<b>Resultado del modelo:</b> Condiciones compatibles con <b>sequía</b>."
                                    "</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    "<div style='padding:0.6rem; border-radius:0.5rem; background-color:#e6ffed;'>"
                                    "<b>Resultado del modelo:</b> Sin indicios fuertes de sequía."
                                    "</div>",
                                    unsafe_allow_html=True,
                                )

                        st.caption(
                            "Esta herramienta es de apoyo. La interpretación final debe considerar el contexto local, "
                            "los índices de sequía y la información de entidades oficiales."
                        )

                    except Exception as e:
                        st.error(f"Ocurrió un error al generar la predicción: {e}")

# =========================
# TAB 4: Buzón de reportes
# =========================
with tab4:
    st.header("**Buzón de Reportes**")

    st.markdown(
        "Si notas signos de sequía o cambios importantes en el clima de tu zona, "
        "puedes dejar aquí tu observación. Tu aporte ayuda a mejorar la información local."
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

    st.markdown(
        """
        <hr style="margin-top: 2rem;">
        <div style="
            text-align: center;
            font-size: 0.8rem;
            color: #666666;
            padding: 0.5rem 0 1rem 0;
        ">
            Desarrollado para el monitoreo de sequías en Riohacha • 
            <a href="https://github.com/Mariannly/Quark5-CoAfina2025.git" target="_blank" style="color: #00d492; text-decoration: none;">
                Más información del proyecto
            </a>
            <br>
            <a href="https://github.com/Mariannly/Quark5-CoAfina2025.git">S-ARIDA</a> © 2025 by <a href="https://github.com/Mariannly/Quark5-CoAfina2025.git">Quark5</a> is licensed under <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International</a>
            <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;">
            <img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;">
        </div>
        """,
        unsafe_allow_html=True,
    )
