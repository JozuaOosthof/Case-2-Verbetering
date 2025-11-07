import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import re
from collections import Counter
import load_data

df, df_geo, df_shp, gdf_merged, gdf, gdf_map = load_data.load_data()

df = df.drop(columns=df.columns[df.isna().sum() > 75])

def _clean_column_label(name: str) -> str:
    return re.sub(r'_\d+$', '', name)

_raw_labels = {col: _clean_column_label(col) for col in df.columns}
_label_counts = Counter(_raw_labels.values())
DISPLAY_LABELS = {
    col: label if _label_counts[label] == 1 else col
    for col, label in _raw_labels.items()
}

def _format_column(col: str) -> str:
    return DISPLAY_LABELS.get(col, col)

st.set_page_config(page_title = "Introductie & Data", layout='wide', page_icon='📄')
sidebar = st.sidebar.header('')

st.title('Introductie & Data')
st.caption("De Gezondheidsmonitor Jongvolwassenen 2024 brengt de ervaringen van ruim 135 duizend jongeren in kaart. Dit dashboard vertaalt die cijfers naar inzichten voor beleid en interventies.")

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
        **Hoe hangt de financiële situatie van jongvolwassenen af met hun ervaren gezondheid?**
        """
                
        )
    st.markdown(
        """
        > 💡 **Kerninzicht**: Financiële stress en mentale gezondheid blijken sterk samen te hangen. De volgende pagina's laten zien hoe dat verschil zich uit in tijd, ruimte en doelgroepen.
        """,
        unsafe_allow_html=True
    )

with st.container(border=True):
    st.subheader('Datakaarten in één oogopslag')
    total_records = len(df)
    gemeenten = df['RegioS'].nunique(dropna=True)
    provincies = df['Provincie'].nunique(dropna=True)
    avg_missing = df.isnull().mean().mean() * 100 if not df.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric('Aantal records', f"{total_records:,}".replace(',', '.'))
    with c2:
        st.metric('Gemeenten in dataset', gemeenten)
    with c3:
        st.metric('Provincies', provincies)
    with c4:
        st.metric('Gem. ontbrekende waarden', f"{avg_missing:.1f}%")

with st.container(border = True):
    st.subheader('Grondige Dataverkenning')
    st.write('Gebruik de tabs hieronder voor een compacte data-audit.')

    overview_tab, stats_tab, missing_tab, id_tab = st.tabs([
        '📄 Voorbeeldgegevens',
        '📊 Statistieken',
        '⚠️ Missende waarden',
        '🧭 Gemeente-ID'
    ])

    with overview_tab:
        st.dataframe(df.head().rename(columns=DISPLAY_LABELS), hide_index=True)
        st.caption("Eerste vijf rijen van de dataset. Elke rij representeert een regio (CBS-statcode) met bijbehorende indicatoren.")

    with stats_tab:
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != 'ID']
        stats_df = df[numerical_cols].describe().reset_index().rename(columns={'index': 'Statistiek'})
        stats_df = stats_df.rename(columns=DISPLAY_LABELS)
        st.dataframe(stats_df, hide_index=True)
        st.caption("Statistieken zijn inclusief gemiddelde (mean), standaarddeviatie (std) en percentielen. Handig om extreme waarden te signaleren.")

    with missing_tab:
        missing_values = df.isnull().sum().reset_index()
        missing_pct = (df.isnull().mean() * 100).reset_index(name='Percentage missend')
        missing_values.columns = ['Variabele', 'Aantal missend']
        missing_pct = missing_pct.rename(columns={'index': 'Variabele'})
        missing_overview = missing_values.merge(missing_pct, on='Variabele')
        missing_overview['Variabele'] = missing_overview['Variabele'].map(_format_column)
        st.dataframe(missing_overview.sort_values('Percentage missend', ascending=False), hide_index=True)
        st.caption("Kolommen met >20% missende waarden worden op de analysepagina expliciet gemarkeerd.")

    with id_tab:
        gemeente_lookup = (
            df[['ID', 'RegioS', 'Provincie']]
            .drop_duplicates()
            .assign(
                Gemeente=lambda d: d['RegioS'].map(
                    gdf[['statcode', 'statnaam']]
                    .drop_duplicates()
                    .set_index('statcode')['statnaam']
                )
            )
            [['ID', 'Gemeente', 'Provincie']]
            .sort_values('Gemeente')
        )
        gemeente_lookup['Gemeente'] = gemeente_lookup['Gemeente'].fillna('Onbekend')
        st.dataframe(gemeente_lookup, hide_index=True)
        st.caption("Deze koppeltabel helpt om ID’s te vertalen naar gemeenten wanneer je verdiepende analyses doet.")

    st.divider()
    st.markdown(
        f"""
        **Belangrijkste indicatoren in dit dashboard**

        - `{_format_column('MoeiteMetRondkomen_1')}` – percentage jongeren dat moeite heeft met rondkomen (☂️ financiële druk).  
        - `{_format_column('GoedErvarenMentaleGezondheid_12')}` – aandeel jongeren dat zijn mentale gezondheid als goed ervaart.  
        - `{_format_column('SlaaptMeestalSlecht_7')}` – percentage dat de slaapkwaliteit als onvoldoende beoordeelt.  
        - `{_format_column('VaakGelukkigAfg4Weken_15')}` – aandeel dat zich vaak gelukkig voelt.  
        - `{_format_column('SportWekelijks_66')}` – aandeel dat wekelijks sport (balans tussen leefstijl en welzijn).
        """
    )

    st.info(
        "Datakwaliteit: de meeste kernvariabelen hebben minder dan 10% missende waarden. "
        "Kolommen met >75% ontbrekende data zijn vooraf verwijderd om de analyses robuust te houden."
    )

    st.write('Databronnen:')
    st.page_link("https://www.rivm.nl/gezondheidsmonitors/jongvolwassenen", label="Gezondheidsmonitor Jongvolwassene (2024)", icon="🌎")
    st.page_link("https://data.opendatasoft.com/explore/dataset/georef-netherlands-gemeente%40public/export/?disjunctive.prov_code&disjunctive.prov_name&disjunctive.gem_code&disjunctive.gem_name", label="Nederland Gemeente Dataset (GeoJSON)", icon="🌎")
