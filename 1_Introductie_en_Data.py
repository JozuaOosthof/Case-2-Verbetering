import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import load_data

df, df_geo, df_shp, gdf_merged, gdf, gdf_map = load_data.load_data()

df = df.drop(columns=df.columns[df.isna().sum() > 75])

st.set_page_config(page_title = "Introductie & Data", layout='wide', page_icon='📄')
sidebar = st.sidebar.header('')

st.title('Introductie & Data')

with st.container(border = True):
    st.markdown("""
        ### Introductie

        Dit dashboard is ontwikkeld voor Case 2. Het doel is om met behulp van data-analyse en visualisatie inzicht te krijgen in de 
        **Gezondheidsmonitor Jongvolwassenen (2024)** van het RIVM.  

        **Projectgroep 6:**  
        - Jozua Oosthof 
        - Joris Kroon
        - Jelle van Wees
        - Niek Tensen

        ### Context
        De Gezondheidsmonitor Jongvolwassenen is een grootschalig onderzoek naar de gezondheid, leefstijl en het welzijn van jongeren en jongvolwassenen in Nederland. 
        De dataset bevat informatie over onder andere **mentale gezondheid, leefgewoonten, middelengebruik en ervaren gezondheid**. 

        ### Doel van dit dashboard
        - Een eerste **dataverkenning** (introductie & kwaliteit van de data).  
        - **Interactieve visualisaties** om verbanden tussen variabelen te ontdekken.  
        - Een **geografisch overzicht** van verschillen tussen regio’s.  
        - Een **statistische analyse** van de relatie tussen financiële situatie en mentale gezondheid.
                
        ### Hoofdvraag
        **Hoe hangt de financiële situatie van jongvolwassenen af met hun gezondheid en leefstijl?**
        """
                
        )

with st.container(border = True):
    st.subheader('Grondige Dataverkenning')
    st.write('Overzicht van de dataset:')
    st.dataframe(df.head(), hide_index=True)
    st.write('Beschrijvende statistieken van numerieke kolommen:')

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.dataframe(df[numerical_cols].describe(), hide_index=True)

    st.write("Aantal missende waarden per kolom:")
    st.dataframe(df.isnull().sum(), hide_index=True)

    st.write('Gemeente Dataset:')
    st.dataframe(gdf.head(), hide_index=True)

    st.write('Databronnen:')
    st.page_link("https://www.rivm.nl/gezondheidsmonitors/jongvolwassenen", label="Gezondheidsmonitor Jongvolwassene (2024)", icon="🌎")
    st.page_link("https://data.opendatasoft.com/explore/dataset/georef-netherlands-gemeente%40public/export/?disjunctive.prov_code&disjunctive.prov_name&disjunctive.gem_code&disjunctive.gem_name", label="Nederland Gemeente Dataset (GeoJSON)", icon="🌎")
