import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
from streamlit_folium import st_folium
import numpy as np
import json
from branca.colormap import linear
import load_data

df, df_geo, df_shp, gdf_merged, gdf, gdf_map = load_data.load_data()

st.set_page_config('Gezondheidsmonitor 2024 Dashboard', layout='wide', page_icon='📊')
st.title('Gezondheidsmonitor 2024 Dashboard')

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(label='Aantal Gemeenten', value=len(df))
with c2:
    st.metric(label='Aantal Variabelen', value=len(df.columns))
with c3:
    st.metric(label='Aantal Respondenten', value=135 * 1000)
with c4:
    st.metric(label='Aantal Waarnemingen', value=df.count().sum())

st.divider()

st.subheader('Visualisaties')

# --- Scatterplot ---
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        scatter_options = [col for col in df.columns if col not in ['ID', 'RegioS', 'Persoonskenmerken', 'Marges', 'Provincie']]
        sb_scatter = st.selectbox('Kleur op', ['Geen', 'Provincie', 'Gemeente'], index=0)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            x_axis = st.selectbox('X-as', scatter_options, index=scatter_options.index('MoeiteMetRondkomen_1'))
        with col_s2:
            y_axis = st.selectbox('Y-as', scatter_options, index=scatter_options.index('HeeftSchulden_3'))
        add_regression = st.checkbox('Regressielijn', value=True)

        scatter = alt.Chart(df).mark_circle().encode(
            x=alt.X(f'{x_axis}:Q', title=x_axis, scale=alt.Scale(zero=False)),
            y=alt.Y(f'{y_axis}:Q', title=y_axis, scale=alt.Scale(zero=False)),
            color=alt.Color(f'{sb_scatter}:N') if sb_scatter != 'Geen' else alt.value('steelblue'),
            tooltip=[x_axis, y_axis]
        ).properties(width=800, height=800, title = f'{x_axis} vs. {y_axis}').interactive()

        if add_regression:
            scatter += scatter.transform_regression(x_axis, y_axis).mark_line(color='red')

        st.altair_chart(scatter, use_container_width=True)

# --- Bar chart ---
with col2:
    with st.container(border=True):
        bar_options = [col for col in df.columns if col not in ['ID', 'RegioS', 'Persoonskenmerken', 'Marges', 'Provincie']]
        selected_bar = st.selectbox('Selecteer Variabele', bar_options, index=0)

        if pd.api.types.is_numeric_dtype(df[selected_bar]):
            bar_data = df.groupby("Provincie", as_index=False)[selected_bar].mean()
        else:
            bar_data = df.groupby("Provincie", as_index=False)[selected_bar].count()

        bar_chart = alt.Chart(bar_data).mark_bar().encode(
            x=alt.X("Provincie:N", sort='-y', title="Provincie"),
            y=alt.Y(f"{selected_bar}:Q", title=selected_bar),
            tooltip=["Provincie:N", f"{selected_bar}:Q"]
        ).properties(width=700, height=400, title=f"{selected_bar} per provincie")

        st.altair_chart(bar_chart, use_container_width=True)

    # --- Boxplot ---
    with st.container(border=True):
        provinces = df['Provincie'].dropna().unique().tolist()
        box_ms = st.multiselect('Selecteer Provincies', provinces, default=provinces[:3])
        box_sb = st.selectbox('X-as Boxplot', bar_options, index=0)
        df_filtered = df[df['Provincie'].isin(box_ms)]
        n_provs = max(len(box_ms), 1)
        box_size = max(10, 200 // n_provs)
        if box_ms:
            box = alt.Chart(df_filtered).mark_boxplot(size=box_size).encode(
                x=alt.X(f'{box_sb}:Q', title=box_sb, scale=alt.Scale(zero=False)),
                y=alt.Y('Provincie:N', title='Provincie'),
                color=alt.Color('Provincie:N', legend=None)
            ).properties(height=392, title = f'Boxplot van {box_sb} per provincie')
            st.altair_chart(box, use_container_width=True)
        else:
            st.info("Selecteer minstens één provincie om de boxplot te tonen.")

# --- Histogram ---
with st.container(border=True):
    hist_ms = st.multiselect('Selecteer Provincies voor Histogram', provinces, default=provinces[:3], key='hist_ms')
    selected_hist = st.selectbox('X-as Histogram', bar_options, index=0, key='hist_select')
    range_val = st.slider('Selecteer bereik', float(df[selected_hist].min()), float(df[selected_hist].max()), (float(df[selected_hist].min()), float(df[selected_hist].max())), key='hist_range', step = 0.1)
    kleur = st.checkbox('Kleur op Provincie', value=False)
    df_hist_filtered = df[(df['Provincie'].isin(hist_ms)) & (df[selected_hist] >= range_val[0]) & (df[selected_hist] <= range_val[1])]

    if hist_ms:
        hist = alt.Chart(df_hist_filtered).mark_bar().encode(
            x=alt.X(f"{selected_hist}:Q", bin=alt.Bin(maxbins=30), title=f'{selected_hist}'),
            y=alt.Y('count()', title='Aantal'),
            color=alt.Color('Provincie:N')
        ).properties(title=f'Verdeling van {selected_hist} per provincie')
        st.altair_chart(hist, use_container_width=True)
    else:
        st.info("Selecteer minstens één provincie om het histogram te tonen.")

# --- Grouped Bar ---
with st.container(border=True):
    stack_vars = st.multiselect(
        'Selecteer Variabelen voor Grouped Bar',
        [col for col in bar_options if pd.api.types.is_numeric_dtype(df[col])],
        default=[bar_options[0]]
    )

    if stack_vars:
        min_val = df[stack_vars].min().min()
        max_val = df[stack_vars].max().max()
        range_val = st.slider(
            'Filter waarden voor de grouped bar',
            min_value=float(min_val),
            max_value=float(max_val),
            value=(float(min_val), float(max_val)),
            step = 0.1
        )

        stack_data = df.groupby("Provincie")[stack_vars].mean().reset_index()
        stack_data = stack_data[(stack_data[stack_vars] >= range_val[0]).all(axis=1)]

        grouped_bar = (
            alt.Chart(stack_data)
            .transform_fold(stack_vars, as_=['Variable', 'Value'])
            .mark_bar()
            .encode(
                x=alt.X('Provincie:N', title="Provincie"),
                xOffset='Variable:N',
                y=alt.Y('Value:Q', title="Gemiddelde waarde"),
                color=alt.Color('Variable:N', legend=alt.Legend(orient='bottom')),
                tooltip=['Provincie:N', 'Variable:N', 'Value:Q']
            )
            .properties(width=700, height=600, title = 'Grouped Bar van geselecteerde variabelen per provincie')
        )

        st.altair_chart(grouped_bar, use_container_width=True)
    else:
        st.info("Selecteer minimaal één numerieke variabele om de grouped bar te tonen.")


# --- Correlatie Heatmap ---
with st.container(border=True):
    financieel_vars = [
        'MoeiteMetRondkomen_1',
        'WeinigControleOverGeldzaken_2',
        'HeeftSchulden_3',
        'HeeftStudieschuld_4',
        'ZorgenOverStudieschuld_5',
        'StressGeldzakenSchulden_33',
        'GeldproblemenOnlineGokkenAfg12Mnd_139',
        'DoetAanBeleggen_140'
    ]

    gezondheid_leefstijl_vars = [
        'GoedErvarenGezondheid_6',
        'SlaaptMeestalSlecht_7',
        'OverdagSlaperigOfMoe_8',
        'TevredenMetZichzelf_9',
        'TevredenMetEigenLichaam_10',
        'TevredenMetEigenLeven_11',
        'GoedErvarenMentaleGezondheid_12',
        'AngstDepressiegevoelensAfg4Weken_13',
        'BeperktDoorPsychischeKlachten_14',
        'VaakGelukkigAfg4Weken_15',
        'VoldoendeWeerbaar_16',
        'Overgewicht_59',
        'Obesitas_60',
        'MatigOvergewicht_61',
        'GezondGewicht_62',
        'Ondergewicht_63',
        'BeweegtDagelijksMinHalfUur_64',
        'BeweegtMin5DagenPWMinHalfUur_65',
        'SportWekelijks_66',
        'SportMin2DagenPW_67',
        'LidSportclubSportschoolOfSportabo_68',
        'AlcoholAfg12Maanden_69',
        'AlcoholAfg4Weken_70',
        'VoldoetAanAlcoholrichtlijn_71',
        'ZwareDrinker_72',
        'RooktTabak_75',
        'ExTabakroker_76',
        'RooktDagelijksTabak_77',
        'RooktWekelijksTabak_78',
        'VapetESigaret_79',
        'ExVaper_80',
        'VapetDagelijks_81',
        'VapetWekelijks_82'
    ]

    fin_vars = st.multiselect('Selecteer financiële variabelen (X-as)', financieel_vars, default=financieel_vars[:3])
    health_vars = st.multiselect('Selecteer gezondheid/leefstijl variabelen (Y-as)', gezondheid_leefstijl_vars, default=gezondheid_leefstijl_vars[:4])

    if fin_vars and health_vars:
        corr_df = df[fin_vars + health_vars].corr().loc[fin_vars, health_vars]
        corr_long = corr_df.reset_index().melt(id_vars='index', var_name='HealthVar', value_name='Correlation')

        corr_chart = alt.Chart(corr_long).mark_rect().encode(
            x=alt.X('HealthVar:O', title='Gezondheid/Leefstijl'),
            y=alt.Y('index:O', title='Financieel'),
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1))),
            tooltip=['index', 'HealthVar', 'Correlation:Q']
        ).properties(width=600, height=650, title='Correlatie Financieel ↔ Gezondheid/Leefstijl')

        st.altair_chart(corr_chart, use_container_width=True)
    else:
        st.info("Selecteer minstens één financiële en één gezondheids-/leefstijlvariabele om de correlatie te tonen.")

st.divider()

df_map = df[['RegioS', 'MoeiteMetRondkomen_1']].copy()
df_map = df_map.rename(columns={'MoeiteMetRondkomen_1':'val'})

map_sb = st.selectbox('Kies variabele voor de kaart', bar_options, index=0, key='map_sb')

gdf_map = gdf[['statcode','statnaam','geometry']].copy()
df_map = df[['RegioS', map_sb]].rename(columns={map_sb: 'val'})
gdf_map = gdf_map.merge(df_map, left_on='statcode', right_on='RegioS', how='left')
gdf_map = gdf_map.drop(columns=['RegioS'])

def gem_opvullen(row, gdf):
    if pd.notna(row['val']):
        return row['val']
    neighbors = gdf[gdf.geometry.touches(row['geometry'])]
    if len(neighbors) > 0:
        return neighbors['val'].mean()
    return np.nan

gdf_map['val'] = gdf_map.apply(lambda row: gem_opvullen(row, gdf_map), axis=1)
gdf_map['val'] = gdf_map['val'].fillna(gdf_map['val'].mean())

def maak_kaart(_gdf, _variable):
    m = folium.Map(location=[52.1, 5.3], zoom_start=7, tiles='CartoDB positron')
    colormap = linear.Blues_09.scale(_gdf['val'].min(), _gdf['val'].max())
    colormap.caption = _variable
    colormap.add_to(m)

    tooltip_fields = []
    tooltip_aliases = []
    if 'statnaam' in _gdf.columns:
        tooltip_fields.append('statnaam')
        tooltip_aliases.append('Naam:')
    if 'statcode' in _gdf.columns:
        tooltip_fields.append('statcode')
        tooltip_aliases.append('Code:')
    if 'Provincie' in _gdf.columns:
        tooltip_fields.append('Provincie')
        tooltip_aliases.append('Provincie:')

    tooltip_fields.append('val')
    tooltip_aliases.append(f'{_variable}:')

    folium.GeoJson(
        _gdf,
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['val']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True
        )
    ).add_to(m)
    return m


col_m1, col_m2 = st.columns(2)

gdf_prov = gdf_map.merge(df[['RegioS','Provincie']], left_on='statcode', right_on='RegioS', how='left')

gdf_prov = gdf_prov.dissolve(by='Provincie', aggfunc={'val':'mean'}).reset_index()
gdf_prov['statnaam'] = gdf_prov['Provincie']  # tooltip

df['RegioS'] = df['RegioS'].astype(str).str.strip()
df_geo['Gemeente code (with prefix)'] = df_geo['Gemeente code (with prefix)'].astype(str).str.strip()



with col_m2:
    with st.container(border=True, height=900):
        st.subheader('Statistieken per gebied')
        map_sb2 = st.selectbox('Selecteer', ['Gemeenten', 'Provincies'], index=0)

        if map_sb2 == 'Gemeenten':
            gdf_to_show = gdf_map
            map_title = f"Kaart van {map_sb} per Gemeente"
            df_stats = df.copy()
            df_stats['Naam'] = df_stats['RegioS'].map(
                gdf[['statcode','statnaam']].drop_duplicates().set_index('statcode')['statnaam']
            ).fillna(df_stats['Provincie'])
            value_col = map_sb
        else:
            gdf_to_show = gdf_prov
            map_title = f"Kaart van {map_sb} per Provincie"
            df_stats = gdf_prov.copy()
            df_stats['Naam'] = df_stats['Provincie']
            value_col = 'val'

        st.write('Samenvattende statistieken:')
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric('Max.', f"{round(df_stats[value_col].max(), 1)}%")
            st.metric('SD.', round(df_stats[value_col].std(), 1))
        with col_s2:
            st.metric('Gem.', f"{round(df_stats[value_col].mean(), 1)}%")
            st.metric('Med.', f"{round(df_stats[value_col].median(), 1)}%")
        with col_s3:
            st.metric('Min.', f"{round(df_stats[value_col].min(), 1)}%")
        st.divider()

        st.write("Top 3:")
        st.dataframe(df_stats.nlargest(3, value_col)[['Naam', value_col]], hide_index=True)

        st.write("Laagste 3:")
        st.dataframe(df_stats.nsmallest(3, value_col)[['Naam', value_col]], hide_index=True)



with col_m1:
    with st.container(border=True):
        st.subheader(f"Kaart van {map_sb} per Gemeente")
        m = maak_kaart(gdf_to_show, map_sb)
        st_folium(m, width=700, height=800)

