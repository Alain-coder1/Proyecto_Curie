import streamlit as st
import pandas as pd
import seaborn as sns
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

def set_sidebar_background(image_path: str):          # imagen de la barra lateral
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    [data-testid="stSidebar"] {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center bottom;
        background-repeat: no-repeat;
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stMultiSelect > div {{
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

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
st.title("🐧 Dashboard - Palmer Penguins")
st.markdown("Análisis exploratorio de las tres especies del archipiélago Palmer.")

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

st.markdown(f"**{len(df_filtrado)} registros** seleccionados")


# kpis

kpis = compute_kpis(df_filtrado)

st.markdown("### 📊 Resumen de la selección")
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(
    label="🐧 Registros",
    value=kpis["total_registros"],
    delta=f"{kpis['total_registros'] - len(df)} vs total",
)
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
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


