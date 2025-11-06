import pandas as pd
import geopandas as gpd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv('data/50140NED_TypedDataSet_22092025_192544.csv', sep=';')
    df_geo = pd.read_csv("data/georef-netherlands-gemeente.csv", sep=";", low_memory=False, on_bad_lines="skip")
    df_shp = gpd.read_file('data/georef-netherlands-gemeente-millesime.shp')

    df = df.drop(columns=df.columns[df.isna().sum() > 75])

    gemeente_to_prov = df_geo.set_index("Gemeente code (with prefix)")["Provincie name"].to_dict()
    df_geo["Provincie"] = df_geo["Gemeente code (with prefix)"].apply(lambda x: gemeente_to_prov.get(x, "Onbekend"))

    df = df.merge(
        df_geo[["Gemeente code (with prefix)", "Provincie"]],
        left_on="RegioS", right_on="Gemeente code (with prefix)", how="left"
    )

    df['RegioS'] = df['RegioS'].astype(str).str.strip()
    df_shp['gem_code'] = df_shp['gem_code'].astype(str).str.strip()

    gdf_merged = df_shp.merge(df, left_on='gem_code', right_on='RegioS', how='left')

    df_map = df[['RegioS', 'MoeiteMetRondkomen_1']].copy()
    df_map = df_map.rename(columns={'MoeiteMetRondkomen_1':'val'})
    gdf = gpd.read_file("data/gemeente_gegeneraliseerd.geojson")[['statcode','statnaam','geometry']]
    gdf = gdf.merge(df_map, left_on='statcode', right_on='RegioS', how='left')

    gdf_map = gdf[['statcode','statnaam','geometry']].copy()
    df_map_val = df[['RegioS', 'MoeiteMetRondkomen_1']].rename(columns={'MoeiteMetRondkomen_1': 'val'})
    gdf_map = gdf_map.merge(df_map_val, left_on='statcode', right_on='RegioS', how='left')
    
    return df, df_geo, df_shp, gdf_merged, gdf, gdf_map
