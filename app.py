import plotly.express as px
import streamlit as st

from src.penguins_pipeline_guia import apply_filters, compute_kpis, load_data

# ------------------------------------------------------------
# 1) CONFIGURACION DE LA APP
# ------------------------------------------------------------
st.set_page_config(page_title="Proyecto Penguins - Dashboard", layout="wide")
st.title("Dashboard Palmer Penguins")
st.caption("Plantilla base: completadla con vuestro criterio de equipo.")

# Esta app NO es una solucion final. Es una estructura guia para que:
# - La logica de datos viva en src/ (reutilizable en notebook y app)
# - La app se centre en interfaz, filtros y visualizacion
DATA_PATH = "data/penguins.csv"

# ------------------------------------------------------------
# 2) CARGA DE DATOS (desde la capa de pipeline en src/)
# ------------------------------------------------------------
df = load_data(DATA_PATH)

# ------------------------------------------------------------
# 3) FILTROS DE INTERFAZ (sidebar)
# ------------------------------------------------------------
st.sidebar.header("Filtros")
species_options = sorted(df["species"].dropna().unique().tolist()) if "species" in df.columns else []
island_options = sorted(df["island"].dropna().unique().tolist()) if "island" in df.columns else []

selected_species = st.sidebar.multiselect("Especie", options=species_options, default=species_options)
selected_islands = st.sidebar.multiselect("Isla", options=island_options, default=island_options)

# Reutilizamos la funcion de filtros de src para evitar logica duplicada.
filtered_df = apply_filters(df, species=selected_species, islands=selected_islands)

# ------------------------------------------------------------
# 4) KPIs (resumen rapido del filtrado actual)
# ------------------------------------------------------------
kpis = compute_kpis(filtered_df)

col1, col2, col3 = st.columns(3)
col1.metric("Filas", kpis["rows"])
col2.metric("Especies", kpis["species"])
col3.metric("Islas", kpis["islands"])

# ------------------------------------------------------------
# 5) TABLA Y GRAFICOS (capa de presentacion)
# ------------------------------------------------------------
st.subheader("Vista de datos")
st.dataframe(filtered_df, use_container_width=True)

if "body_mass_g" in filtered_df.columns and "species" in filtered_df.columns:
    st.subheader("Masa corporal por especie")
    fig = px.box(filtered_df, x="species", y="body_mass_g", points="outliers")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Completad esta seccion con los graficos acordados por el equipo.")

# ------------------------------------------------------------
# 6) EXPORTACION (obligatoria para entrega ampliada)
# ------------------------------------------------------------
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar CSV filtrado",
    data=csv,
    file_name="penguins_filtrado.csv",
    mime="text/csv",
)

st.markdown("---")
st.write("Checklist rapido antes de deploy:")
st.write("- Coherencia entre notebook y dashboard")
st.write("- Filtros funcionales")
st.write("- CSV correcto")
st.write("- URL publica operativa")