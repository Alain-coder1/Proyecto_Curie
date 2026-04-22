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
    Carga y limpia el dataset base.

    Recomendacion para alumnado:
    - Si aplicais limpieza fija para todo el proyecto, hacedla aqui.
    - Si una limpieza es solo exploratoria, mantenedla en notebook y justificadla.
    """
    df = pd.read_csv(path)

    # Seleccion de columnas relevantes (igual que en el notebook)
    columnas = [
        'Species', 'Island', 'Individual ID',
        'Culmen Length (mm)', 'Culmen Depth (mm)',
        'Flipper Length (mm)', 'Body Mass (g)', 'Sex'
    ]
    df = df[columnas].copy()

    # Simplificar nombre de especie a la primera palabra (ej: "Adelie Penguin" -> "Adelie")
    df['Species'] = df['Species'].str.split().str[0]

    # Imputacion de nulos numericos con la mediana de cada especie
    cols_num = [
        'Culmen Length (mm)', 'Culmen Depth (mm)',
        'Flipper Length (mm)', 'Body Mass (g)'
    ]
    for especie in df['Species'].unique():
        mask = df['Species'] == especie
        df.loc[mask, cols_num] = df.loc[mask, cols_num].fillna(
            df.loc[mask, cols_num].median()
        )

    # Imputacion de nulos en Sex con categoria UNKNOWN
    df['Sex'] = df['Sex'].fillna('UNKNOWN')

    return df


def apply_filters(
    df: pd.DataFrame,
    species: list[str],
    islands: list[str],
    sexes: list[str]
) -> pd.DataFrame:
    """
    Aplica los filtros seleccionados por la interfaz del dashboard.

    Nota didactica:
    - Esta funcion debe ser determinista.
    - El mismo input debe devolver el mismo output siempre.
    """
    filtered_df = df.copy()

    if 'Species' in filtered_df.columns and species:
        filtered_df = filtered_df[filtered_df['Species'].isin(species)]

    if 'Island' in filtered_df.columns and islands:
        filtered_df = filtered_df[filtered_df['Island'].isin(islands)]

    if 'Sex' in filtered_df.columns and sexes:
        filtered_df = filtered_df[filtered_df['Sex'].isin(sexes)]

    return filtered_df



def compute_kpis(df: pd.DataFrame) -> dict:
    return {
        "total_registros": int(len(df)),
        "num_especies":    int(df["Species"].nunique()) if "Species" in df.columns else 0,
        "num_islas":       int(df["Island"].nunique())  if "Island"  in df.columns else 0,
        "masa_media":      round(df["Body Mass (g)"].mean(), 1) if "Body Mass (g)" in df.columns else 0,
        "aleta_media":     round(df["Flipper Length (mm)"].mean(), 1) if "Flipper Length (mm)" in df.columns else 0,
        "pct_machos":      round((df["Sex"] == "MALE").sum() / len(df) * 100, 1) if "Sex" in df.columns else 0,
    }