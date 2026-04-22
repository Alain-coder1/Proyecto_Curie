import streamlit as st
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from src.penguins_pipeline_guia import load_data, apply_filters, compute_kpis

# fondo dashboard

import base64

def set_background(image_path: str):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center bottom;  /* muestra los pingüinos */
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def set_sidebar_background(image_path: str):  # barra lateral
    import base64
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>

    /* Fondo sidebar */
    [data-testid="stSidebar"] {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center bottom;
        background-repeat: no-repeat;
    }}

    /* ❌ EVITA ESTO GLOBAL (rompe estilos) */
    /* [data-testid="stSidebar"] * {{
        color: white !important;
    }} */

    /* 🟦 CHIPS seleccionados */
    [data-testid="stSidebar"] span[data-baseweb="tag"] {{
        background-color: #87CEEB !important;
        color: black !important;
        border-radius: 6px;
    }}

    /* 🔤 LABELS (Especie, Isla, Sexo) */
    [data-testid="stSidebar"] label {{
        color: black !important;
        font-weight:500;
    }}

    /* Caja multiselect */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
    }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

st.markdown("""
<style>

/* Tabs base */
.stTabs [role="tab"] {
    background-color: rgba(30, 40, 60, 0.85);
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 1rem;
    color: #E8EAF6 !important;
    border: 1px solid rgba(135, 206, 250, 0.2);
}

/* Hover */
.stTabs [role="tab"]:hover {
    background-color: rgba(135, 206, 250, 0.2);
}

/* 🔥 TAB ACTIVA (sobrescribe rojo) */
.stTabs [role="tab"][aria-selected="true"] {
    background-color: rgba(135, 206, 250, 0.35) !important;
    color: black !important;
    border: 1px solid #87CEFA !important;
    border-bottom: none !important;   /* 👈 quita rojo */
}

/* 🔥 ESTA ES LA CLAVE REAL */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
}

/* eliminar línea roja completamente */
.stTabs [role="tab"][aria-selected="true"]::after {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

# ── Configuración ──────────────────────────────────────────

st.set_page_config(page_title="Penguins Dashboard", layout="wide")
set_background("assets/dark_pingu.png")
set_sidebar_background("assets/sidebarpingu.jpeg")  # barra lateral


# ── Carga y limpieza (igual que en tu notebook) ─────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv('data/penguins_raw.csv')
    columnas = ['Species', 'Island', 'Individual ID',
                'Culmen Length (mm)', 'Culmen Depth (mm)',
                'Flipper Length (mm)', 'Body Mass (g)', 'Sex']
    df_limpio = df[columnas].copy()
    df_limpio['Species'] = df_limpio['Species'].str.split().str[0]

    # Imputación por mediana por especie
    cols_num = ['Culmen Length (mm)', 'Culmen Depth (mm)',
                'Flipper Length (mm)', 'Body Mass (g)']
    for especie in df_limpio['Species'].unique():
        mask = df_limpio['Species'] == especie
        df_limpio.loc[mask, cols_num] = df_limpio.loc[mask, cols_num].fillna(
            df_limpio.loc[mask, cols_num].median()
        )
    df_limpio['Sex'] = df_limpio['Sex'].fillna('UNKNOWN')
    return df_limpio

df = cargar_datos()

# ── Título ──────────────────────────────────────────────────
# Primero carga la imagen FUERA del markdown
ruta_foto = os.path.join("assets", "minipingui.png")
with open(ruta_foto, "rb") as f:
    foto_titulo = base64.b64encode(f.read()).decode()

# Luego construye todo el HTML junto en un solo markdown
st.markdown(f"""
<div style='
    background: rgba(255, 255, 255, 0.06);
    padding: 6px 12px;
    border-radius: 10px;
    border: 1px solid rgba(135, 206, 250, 0.25);
    backdrop-filter: blur(6px);
    margin: 0 auto 10px auto;
    width: fit-content;
    text-align: center;
'>
    <h1 style='margin:0; font-size:3rem; color:#E8EAF6; display:flex; align-items:center; gap:15px;'>
        <img src='data:image/png;base64,{foto_titulo}' style='height:90px; border-radius:8px;'>
        Dashboard - Palmer Penguins - Equipo 1
    </h1>
    <p style='margin: 2px 0 0 0; font-size: 1rem; color: #aaaaaa;'>
        Análisis exploratorio de las tres especies del archipiélago Palmer.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Filtros en la barra lateral ───────────────────────────────────
st.sidebar.header("Filtros")

especies = st.sidebar.multiselect(
    "Especie", df['Species'].unique(), default=df['Species'].unique()
)
islas = st.sidebar.multiselect(
    "Isla", df['Island'].unique(), default=df['Island'].unique()
)
sexos = st.sidebar.multiselect(
    "Sexo", df['Sex'].unique(), default=df['Sex'].unique()
)

df_filtrado = df[
    df['Species'].isin(especies) &
    df['Island'].isin(islas) &
    df['Sex'].isin(sexos)
]

st.sidebar.markdown(f"📊 {len(df_filtrado)} registros")


# kpis

kpis = compute_kpis(df_filtrado)

st.markdown("### 📊 Resumen de la selección")
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.markdown(f"""
    <div data-testid="stMetric">
        <p style="font-size:14px; margin:0; color:white;">🐧 Registros</p>
        <p style="font-size:28px; font-weight:bold; margin:0; color:white;">
            {kpis["total_registros"]} 
            <span style="font-size:14px; margin-left:16px; color:white;">
                {kpis['total_registros'] - len(df)} vs total
            </span>
        </p>
    </div>
""", unsafe_allow_html=True)
col2.metric(
    label="🔬 Especies",
    value=kpis["num_especies"],
)
col3.metric(
    label="🏝️ Islas",
    value=kpis["num_islas"],
)
col4.metric(
    label="⚖️ Masa media",
    value=f"{kpis['masa_media']} g",
)
col5.metric(
    label="🪶 Aleta media",
    value=f"{kpis['aleta_media']} mm",
)
col6.metric(
    label="♂️ % Machos",
    value=f"{kpis['pct_machos']} %",
)

st.markdown("---")

st.markdown("""
<style>

/* KPI CARD */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.08);  /* fondo suave */
    border-radius: 14px;
    padding: 12px;
    border: 1px solid rgba(135, 206, 250, 0.3);  /* azul cielo */
    backdrop-filter: blur(8px);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
}

/* VALOR KPI */
[data-testid="stMetricValue"] {
    color: white !important;  /* blanco */
    font-size: 1.8rem;
    font-weight: bold;
}

/* LABEL KPI */
[data-testid="stMetricLabel"] {
    color: #cccccc !important;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}

/* DELTA */
[data-testid="stMetricDelta"] {
    color: #4ECDC4 !important;
}

</style>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Vista Datos", "Univariado", "Bivariado", "Panel Final"])

# TAB 1 - Vista Datos
with tab1:
    st.subheader("Vista de datos")
    st.dataframe(df_filtrado, use_container_width=True)


# TAB 2 — Univariado
with tab2:
    variable = st.selectbox("Variable numérica", [
        'Culmen Length (mm)', 'Culmen Depth (mm)',
        'Flipper Length (mm)', 'Body Mass (g)'
    ])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    sns.boxplot(data=df_filtrado[variable], ax=ax[0], orient='h', width=0.4)
    sns.histplot(data=df_filtrado[variable], ax=ax[1], kde=False)
    ax[1].set_ylabel('Frecuencia')
    sns.kdeplot(data=df_filtrado[variable], ax=ax[2], fill=True)
    ax[2].set_ylabel('Densidad')
    for a in ax:
        a.set_xlabel(variable)
    fig.suptitle(f'Análisis Univariado — {variable}')
    plt.tight_layout()
    st.pyplot(fig)

# TAB 3 — Bivariado
with tab3:
    opcion = st.select_slider(
        "Selecciona el gráfico",
        options=["Longitud vs Profundidad del pico", "Masa corporal por especie y sexo"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    if opcion == "Longitud vs Profundidad del pico":
        sns.scatterplot(data=df_filtrado,
                        x='Culmen Length (mm)', y='Culmen Depth (mm)',
                        hue='Species', ax=ax)
        ax.set_title('Longitud vs Profundidad del pico')

    else:
        sns.boxplot(data=df_filtrado, x='Species', y='Body Mass (g)',
                    hue='Sex', palette='Set2', ax=ax)
        ax.set_title('Masa corporal por especie y sexo')
        plt.tight_layout()

    st.pyplot(fig)

# TAB 4 — Panel Final (tus 4 visualizaciones)
with tab4:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(data=df_filtrado, x="Flipper Length (mm)",
                 hue="Species", kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribución de tamaño de aleta por especie')

    sns.boxplot(data=df_filtrado, x='Species', y='Body Mass (g)',
                hue='Sex', palette='Set2', ax=axes[0, 1])
    axes[0, 1].set_title('Masa corporal por especie y sexo')

    sns.scatterplot(data=df_filtrado, x='Culmen Length (mm)',
                    y='Culmen Depth (mm)', hue='Species', ax=axes[1, 0])
    axes[1, 0].set_title('Longitud vs profundidad del pico')

    counts = df_filtrado['Species'].value_counts()
    axes[1, 1].pie(counts.values, labels=counts.index,
                   autopct='%1.1f%%',
                   colors=sns.color_palette("pastel", len(counts)))
    axes[1, 1].set_title('Distribución de ejemplares por especie')

    plt.tight_layout()
    st.pyplot(fig)


