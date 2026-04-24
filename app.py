import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import os
import matplotlib.pyplot as plt
from src.penguins_pipeline_guia import load_data, apply_filters, compute_kpis
import folium
from streamlit_folium import st_folium
from PIL import Image
import numpy as np



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

# ___ Configuracuón estilos gráficos ________________________

def style_matplotlib(fig, axes):
    BG = (30/255, 41/255, 59/255, 0.55)

    fig.patch.set_facecolor(BG)

    if not isinstance(axes, (list, tuple)):
        axes = [axes]

    for ax in axes:
        ax.set_facecolor(BG)

        ax.grid(True, linestyle='--', alpha=0.15)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(colors="#E5E7EB")
        ax.xaxis.label.set_color("#E8EAF6")
        ax.yaxis.label.set_color("#F1F5F9")
        ax.title.set_color("#F1F5F9")

import plotly.express as px

PLOTLY_BG = 'rgba(30, 41, 59, 0.55)'

def style_plotly(fig):
    fig.update_layout(
        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor=PLOTLY_BG,
        font=dict(color="#E8EAF6"),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

PALETA = {
    "Adelie": "#60A5FA",
    "Chinstrap": "#F59E0B",
    "Gentoo": "#34D399"
}


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

#--- NUEVA SECCIÓN: LÓGICA DE MACHINE LEARNING ---
@st.cache_resource # Usamos cache_resource para objetos complejos como modelos
def entrenar_modelo(data):
    # Seleccionamos las columnas numéricas para el entrenamiento
    X = data[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']]
    y = data['Species']

#Creamos y entrenamos el Bosque Aleatorio
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

#Entrenamos el modelo con los datos base
modelo_ia = entrenar_modelo(df)

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

st.markdown("### 💻 Resumen de la selección")
col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1.5])

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
    label="👥 Sexo",
    value=f"{kpis['pct_male']}% ♂ | {kpis['pct_female']}% ♀"
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
    color: white !important;
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
# ── Estilos botones ────────────────────────────────────────
st.markdown("""
<style>
button[kind="secondary"] {
    background: rgba(15, 23, 42, 0.55) !important;
    color: #E8EAF6 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(6px);
    padding: 10px 18px !important;
    transition: all 0.2s ease;
}
button[kind="secondary"]:hover {
    background: rgba(30, 41, 59, 0.7) !important;
}
button[kind="primary"] {
    background: rgba(96, 165, 250, 0.35) !important;
    border: 1px solid #60A5FA !important;
    color: white !important;
}
div[data-testid="column"] {
    padding: 0px 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Estado inicial ─────────────────────────────────────────
if "view" not in st.session_state:
    st.session_state.view = None

def toggle_view(v):
    if st.session_state.view == v:
        st.session_state.view = None
    else:
        st.session_state.view = v

# ── Botones ────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

with c1:
    st.button("🔎 Vista Datos", on_click=toggle_view, args=("Vista Datos",),
        type="primary" if st.session_state.view == "Vista Datos" else "secondary", use_container_width=True)
with c2:
    st.button("📈 Univariado", on_click=toggle_view, args=("Univariado",),
        type="primary" if st.session_state.view == "Univariado" else "secondary", use_container_width=True)
with c3:
    st.button("📊 Bivariado", on_click=toggle_view, args=("Bivariado",),
        type="primary" if st.session_state.view == "Bivariado" else "secondary", use_container_width=True)
with c4:
    st.button("🏁 Panel Final", on_click=toggle_view, args=("Panel Final",),
        type="primary" if st.session_state.view == "Panel Final" else "secondary", use_container_width=True)
with c5:
    st.button("🔮 Predicción IA", on_click=toggle_view, args=("Predicción IA",),
        type="primary" if st.session_state.view == "Predicción IA" else "secondary", use_container_width=True)
with c6:
    st.button("🗺️ Archipiélago", on_click=toggle_view, args=("Archipiélago",),
        type="primary" if st.session_state.view == "Archipiélago" else "secondary", use_container_width=True)
with c7:
    st.button("📝 Hallazgos", on_click=toggle_view, args=("Hallazgos",),
        type="primary" if st.session_state.view == "Hallazgos" else "secondary", use_container_width=True)

view = st.session_state.view

# ── Vistas ─────────────────────────────────────────────────

if view == "Vista Datos":
    st.subheader("Vista de datos")
    st.dataframe(df_filtrado, use_container_width=True)

# TAB 2 — Univariado
elif view == "Univariado":

    variable = st.selectbox("Variable numérica", [
        'Culmen Length (mm)', 'Culmen Depth (mm)',
        'Flipper Length (mm)', 'Body Mass (g)'
    ])

    col1, col2 = st.columns(2)

    # ─────────────────────────────
    # 📦 BOXPLOT (PLOTLY)
    # ─────────────────────────────
    with col1:
        fig = px.box(
            df_filtrado,
            y=variable,
            color='Species',
            color_discrete_map=PALETA
        )

        fig.update_layout(
            title="Boxplot",
            height=640,
            paper_bgcolor='rgba(30, 41, 59, 0.55)',
            plot_bgcolor='rgba(30, 41, 59, 0.55)',
            font=dict(color="#E8EAF6")
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────
    # 📊 HISTOGRAMA (SEABORN → COMO TU IMAGEN)
    # ─────────────────────────────
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.histplot(
            data=df_filtrado,
            x=variable,
            hue="Species",
            bins=20,
            kde=True,
            palette=PALETA,
            edgecolor="black",
            alpha=0.4,
            ax=ax
        )

        ax.set_title("Distribución")

        style_matplotlib(fig, ax)

        st.pyplot(fig)

    
# TAB 3 — Bivariado
elif view == "Bivariado":

    col1, col2 = st.columns(2)

    # 🔵 SCATTER
    with col1:
        fig = px.scatter(
            df_filtrado,
            x='Culmen Length (mm)',
            y='Culmen Depth (mm)',
            color='Species',
            color_discrete_map=PALETA
        )
        fig.update_layout(title="Longitud vs Profundidad")
        st.plotly_chart(style_plotly(fig), use_container_width=True)

    # 📦 BOXPLOT
    with col2:
        fig = px.box(
            df_filtrado,
            x='Species',
            y='Body Mass (g)',
            color='Sex',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(title="Masa corporal por especie y sexo")
        st.plotly_chart(style_plotly(fig), use_container_width=True)

# TAB 4 — Panel Final (tus 4 visualizaciones)
elif view == "Panel Final":

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # ─────────────────────────────
    # 📊 HISTOGRAMA (SEABORN PRO)
    # ─────────────────────────────
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))

        sns.histplot(
            data=df_filtrado,
            x="Flipper Length (mm)",
            hue="Species",
            bins=20,
            kde=True,
            palette=PALETA,
            edgecolor="black",
            alpha=0.4,
            ax=ax
        )

        ax.set_title("Distribución de aletas")

        style_matplotlib(fig, ax)

        st.pyplot(fig)

    # ─────────────────────────────
    # 📦 BOXPLOT (PLOTLY)
    # ─────────────────────────────
    with col2:
        fig = px.box(
            df_filtrado,
            x='Species',
            y='Body Mass (g)',
            color='Sex',
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_layout(
            title="Masa corporal",
            height=640,
            paper_bgcolor='rgba(30, 41, 59, 0.55)',
            plot_bgcolor='rgba(30, 41, 59, 0.55)',
            font=dict(color="#E8EAF6")
        )

        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────
    # 🔵 SCATTER (PLOTLY)
    # ─────────────────────────────
    with col3:
        fig = px.scatter(
            df_filtrado,
            x='Culmen Length (mm)',
            y='Culmen Depth (mm)',
            color='Species',
            color_discrete_map=PALETA
        )

        fig.update_layout(
            title="Relación del pico",
            paper_bgcolor='rgba(30, 41, 59, 0.55)',
            plot_bgcolor='rgba(30, 41, 59, 0.55)',
            font=dict(color="#E8EAF6")
        )

        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────
    # 🥧 PIE (PLOTLY)
    # ─────────────────────────────
    with col4:
        counts = df_filtrado['Species'].value_counts().reset_index()
        counts.columns = ['Species', 'count']

        fig = px.pie(
            counts,
            names='Species',
            values='count',
            color='Species',
            color_discrete_map=PALETA
        )

        fig.update_layout(
            title="Distribución de especies",
            paper_bgcolor='rgba(30, 41, 59, 0.55)',
            font=dict(color="#E8EAF6")
        )

        st.plotly_chart(fig, use_container_width=True)

elif view == "Predicción IA":
    st.header("🔮 Identificador de Especies Inteligente")
    st.write("Ajusta las medidas para ver la predicción en tiempo real:")
    c1, c2 = st.columns(2)
    with c1:
        val_culmen_l = st.slider("Longitud del pico (mm)", 30.0, 60.0, 44.0)
        val_culmen_d = st.slider("Profundidad del pico (mm)", 13.0, 22.0, 17.0)
    with c2:
        val_flipper = st.slider("Longitud de la aleta (mm)", 170.0, 240.0, 200.0)
        val_mass = st.slider("Masa corporal (g)", 2500.0, 6500.0, 4200.0)

    input_data = pd.DataFrame([[val_culmen_l, val_culmen_d, val_flipper, val_mass]],
                              columns=['Culmen Length (mm)', 'Culmen Depth (mm)',
                                       'Flipper Length (mm)', 'Body Mass (g)'])
    pred = modelo_ia.predict(input_data)[0]
    prob = modelo_ia.predict_proba(input_data)
    confianza = max(prob[0]) * 100

    if confianza <= 25:
        color_gradiente = "linear-gradient(90deg, #8b0000, #ff4b2b)"
        color_borde = "#ff4b2b"
    elif confianza <= 50:
        color_gradiente = "linear-gradient(90deg, #f37335, #fdc830)"
        color_borde = "#f37335"
    elif confianza <= 75:
        color_gradiente = "linear-gradient(90deg, #add100, #7b920a)"
        color_borde = "#add100"
    else:
        color_gradiente = "linear-gradient(90deg, #11998e, #38ef7d)"
        color_borde = "#38ef7d"

    st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(2px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 25px;
            text-align: center;
        ">
            <h3 style="color: white; margin-bottom: 5px;">Especie Identificada</h3>
            <h1 style="color: #87CEFA; margin-top: 0; font-size: 3rem;">🐧 {pred}</h1>
            <div style="
                background: {color_gradiente};
                height: 30px; width: 100%;
                border-radius: 15px; margin-top: 15px;
                display: flex; align-items: center; justify-content: center;
                border: 1px solid {color_borde};
            ">
                <span style="color: white; font-weight: bold;">
                    Confianza: {confianza:.2f}%
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif view == "Archipiélago":
    st.header("🗺️ Islas del Archipiélago Palmer")
    islas_coords = {
        "Torgersen": {"lat": -64.7667, "lon": -64.0833,
                      "especies": "Adelie", "color": "red"},
        "Biscoe":    {"lat": -64.8038, "lon": -63.8326,
                      "especies": "Adelie, Gentoo", "color": "blue"},
        "Dream":     {"lat": -64.7333, "lon": -64.2333,
                      "especies": "Adelie, Chinstrap", "color": "orange"},
    }
    m = folium.Map(location=[-64.77, -64.10], zoom_start=10)
    for isla, datos in islas_coords.items():
        folium.Marker(
            location=[datos["lat"], datos["lon"]],
            popup=f"<b>{isla}</b><br>Especies: {datos['especies']}",
            tooltip=f"🏝️ {isla} — Especies: {datos['especies']}",
            icon=folium.Icon(color=datos["color"], icon="info-sign")
        ).add_to(m)
    st_folium(m, use_container_width=True, height=500)

elif view == "Hallazgos":
    st.header("🏆 Hallazgos Principales del Análisis")

    if "hallazgo_abierto" not in st.session_state:
        st.session_state.hallazgo_abierto = None

    def toggle_hallazgo(h):
        if st.session_state.hallazgo_abierto == h:
            st.session_state.hallazgo_abierto = None
        else:
            st.session_state.hallazgo_abierto = h

    hallazgos = [
        {
            "key": "distribucion_isla",
            "titulo": "1. Patrón de Distribución por Isla",
            "texto": """
                <b>Evidencia observada:</b> La especie Adelie aparece en las 3 islas, pero Gentoo y Chinstrap solamente aparecen en una de ellas, Biscoe y Dream, respectivamente.<br><br>
                <b>Interpretación:</b> Hay un desequilibrio de distribución en los datos recolectados. No podemos determinar si este desequilibrio se debe a la metodología del muestreo o a la distribución real de especies en cada isla.<br><br>
                <b>Implicación para el cliente:</b> No se pueden realizar comparaciones bivariadas entre especies dentro de la Isla Torgersen (solo hay Adelie).<br><br>
                <b>Recomendación:</b> En la próxima campaña, documentar cómo se realizó el muestreo, si se realiza de forma aleatoria o si se hace una preselección por especie.
            """
        },
        {
            "key": "dimorfismo",
            "titulo": "2. Dimorfismo Sexual en la Masa Corporal",
            "texto": """
                <b>Evidencia observada:</b> En todas las categorías cruzadas, la media de masa de los machos es superior a la de las hembras.<br><br>
                <b>Interpretación:</b> El sexo es un factor de variabilidad constante en el peso, independientemente de la especie.<br><br>
                <b>Implicación para el cliente:</b> Un análisis de peso que no segregue por sexo dará una visión distorsionada (bimodal) de la realidad.<br><br>
                <b>Recomendación:</b> Asegurar la identificación del sexo en el 100% de los individuos capturados para no invalidar las comparaciones de masa.
            """
        },
        {
            "key": "pico",
            "titulo": "3. Patrón de Forma del Pico",
            "texto": """
                <b>Evidencia observada:</b> Los Adelie tienen picos cortos y anchos, los Chinstrap tienen picos largos pero también anchos (más equilibrados), y los Gentoo tienen picos largos pero más estrechos.<br><br>
                <b>Interpretación:</b> La relación largo/ancho del pico es el único factor que permite distinguir a los Adelie de los Chinstrap, que por lo demás son muy similares en peso y tamaño de aletas.<br><br>
                <b>Implicación para el cliente:</b> Para distinguir estas dos especies "pequeñas", se requiere obligatoriamente la medida del culmen.<br><br>
                <b>Recomendación:</b> En futuras campañas, tratar de reducir el número de datos nulos en Culmen Depth, ya que es el único diferenciador para el 50% de las especies del dataset.
            """
        },
        {
            "key": "adelie",
            "titulo": "4. Distribución de la Especie Adelie",
            "texto": """
                <b>Evidencia observada:</b> La especie Adelie es la única con registros en las tres islas, representando aproximadamente el 44% del total de observaciones.<br><br>
                <b>Interpretación:</b> Es la especie con mayor presencia y distribución geográfica en los datos recolectados.<br><br>
                <b>Implicación para el cliente:</b> Los datos de Adelie son los más robustos para realizar comparaciones transversales entre islas.<br><br>
                <b>Recomendación:</b> Utilizar a la población de Adelie como grupo de control o referencia para comparar las condiciones de las tres islas, ya que es la única constante en el dataset.
            """
        },
        {
            "key": "calidad",
            "titulo": "5. Calidad del Muestreo (Valores Faltantes)",
            "texto": """
                <b>Evidencia observada:</b> Las variables morfológicas tienen muy pocos nulos (2 casos por cada variable), pero el sexo tiene más (11 casos).<br><br>
                <b>Interpretación:</b> Las medidas físicas son más fáciles de obtener o registrar que la determinación del sexo.<br><br>
                <b>Implicación para el cliente:</b> La base de datos es muy confiable para análisis morfológico, pero menos para análisis demográficos por sexo.<br><br>
                <b>Recomendación:</b> Revisar el protocolo de sexado en campo o incluir pruebas genéticas si los datos de sexo son críticos para el futuro del proyecto.
            """
        },
    ]

    for h in hallazgos:
        abierto = st.session_state.hallazgo_abierto == h["key"]

        st.button(
            f"{'▼' if abierto else '▶'}  {h['titulo']}",
            key=f"btn_{h['key']}",
            on_click=toggle_hallazgo,
            args=(h["key"],),
            use_container_width=True,
        )

        if abierto:
            st.markdown(
                f"""<div style="
                    background: rgba(30, 41, 59, 0.40);
                    padding: 16px 20px;
                    border-radius: 10px;
                    border-left: 3px solid #87CEEB;
                    backdrop-filter: blur(8px);
                    border-top: 1px solid rgba(255,255,255,0.08);
                    box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
                    color: #E8EAF6;
                    margin-bottom: 8px;
                ">{h['texto']}</div>""",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.image(
        "assets/resupingui.png",
        caption="Resumen visual: Hallazgos y Conclusiones finales",
        use_container_width=True
    )
#── Footer ─────────────────────────────────────────────────
st.markdown("""
<style>

/* Footer fijo */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: rgba(255, 255, 255, 0.05);
    color: #cccccc;
    text-align: center;
    padding: 8px;
    font-size: 0.8rem;
    border-top: 1px solid rgba(135, 206, 250, 0.2);
    backdrop-filter: blur(6px);
}

</style>

<div class="footer">
🐧 © 2026 Alain · Lucia P. · Cheyenne · Agata · Fran D. · Carolina
</div>
""", unsafe_allow_html=True)