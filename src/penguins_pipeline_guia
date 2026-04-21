"""
Guia de pipeline de datos para el proyecto Penguins.

Objetivo: centralizar en un solo sitio la logica de datos que usan
- el notebook (analisis)
- la app Streamlit (presentacion)

Asi evitamos resultados distintos por duplicar codigo.
"""

from __future__ import annotations

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Carga el dataset base.

    Recomendacion para alumnado:
    - Si aplicais limpieza fija para todo el proyecto, hacedla aqui.
    - Si una limpieza es solo exploratoria, mantenedla en notebook y justificadla.
    """
    return pd.read_csv(path)


def apply_filters(df: pd.DataFrame, species: list[str], islands: list[str]) -> pd.DataFrame:
    """
    Aplica filtros seleccionados por la interfaz.

    Nota didactica:
    - Esta funcion debe ser determinista.
    - El mismo input debe devolver el mismo output siempre.
    """
    filtered_df = df.copy()

    if "species" in filtered_df.columns and species:
        filtered_df = filtered_df[filtered_df["species"].isin(species)]

    if "island" in filtered_df.columns and islands:
        filtered_df = filtered_df[filtered_df["island"].isin(islands)]

    return filtered_df


def compute_kpis(df: pd.DataFrame) -> dict[str, int]:
    """
    Calcula KPIs minimos comunes entre notebook y app.

    Puedes ampliar con:
    - masa media
    - longitud de aleta media
    - ratio por especie
    """
    return {
        "rows": int(len(df)),
        "species": int(df["species"].nunique()) if "species" in df.columns else 0,
        "islands": int(df["island"].nunique()) if "island" in df.columns else 0,
    }