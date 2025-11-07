import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
from streamlit_folium import st_folium
import numpy as np
import re
from collections import Counter
from branca.colormap import linear
import load_data

FINANCIAL_COLUMNS = [
    'MoeiteMetRondkomen_1',
    'WeinigControleOverGeldzaken_2',
    'HeeftSchulden_3',
    'HeeftStudieschuld_4',
    'ZorgenOverStudieschuld_5',
    'StressGeldzakenSchulden_33',
    'GeldproblemenOnlineGokkenAfg12Mnd_139',
    'DoetAanBeleggen_140'
]

HEALTH_COLUMNS = [
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

df, df_geo, df_shp, gdf_merged, gdf, gdf_map = load_data.load_data()

RELEVANT_COLUMNS = [
    col for col in dict.fromkeys(FINANCIAL_COLUMNS + HEALTH_COLUMNS)
    if col in df.columns
]

EXCLUDED_COLUMNS = ['ID', 'RegioS', 'Persoonskenmerken', 'Marges', 'Provincie']
AVAILABLE_RELEVANT = [col for col in RELEVANT_COLUMNS if col not in EXCLUDED_COLUMNS]
BAR_OPTIONS = AVAILABLE_RELEVANT
bar_options = BAR_OPTIONS
provinces = df['Provincie'].dropna().unique().tolist()

DEFAULTS = {
    'scatter_color': 'Geen',
    'scatter_x': 'MoeiteMetRondkomen_1' if 'MoeiteMetRondkomen_1' in AVAILABLE_RELEVANT else (AVAILABLE_RELEVANT[0] if AVAILABLE_RELEVANT else None),
    'scatter_y': 'GoedErvarenMentaleGezondheid_12' if 'GoedErvarenMentaleGezondheid_12' in AVAILABLE_RELEVANT else ('HeeftSchulden_3' if 'HeeftSchulden_3' in AVAILABLE_RELEVANT else (AVAILABLE_RELEVANT[0] if AVAILABLE_RELEVANT else None)),
    'bar_variable': 'MoeiteMetRondkomen_1' if 'MoeiteMetRondkomen_1' in AVAILABLE_RELEVANT else (AVAILABLE_RELEVANT[0] if AVAILABLE_RELEVANT else None),
    'box_variable': 'MoeiteMetRondkomen_1' if 'MoeiteMetRondkomen_1' in AVAILABLE_RELEVANT else (AVAILABLE_RELEVANT[0] if AVAILABLE_RELEVANT else None),
    'hist_variable': 'MoeiteMetRondkomen_1' if 'MoeiteMetRondkomen_1' in AVAILABLE_RELEVANT else (AVAILABLE_RELEVANT[0] if AVAILABLE_RELEVANT else None),
    'grouped_variables': ['MoeiteMetRondkomen_1'] if 'MoeiteMetRondkomen_1' in AVAILABLE_RELEVANT else (AVAILABLE_RELEVANT[:1]),
    'map_variable': 'MoeiteMetRondkomen_1' if 'MoeiteMetRondkomen_1' in AVAILABLE_RELEVANT else (AVAILABLE_RELEVANT[0] if AVAILABLE_RELEVANT else None)
}
def _clean_column_label(name: str) -> str:
    """Strip trailing _<number> to create friendlier labels for the UI."""
    return re.sub(r'_\d+$', '', name)

_column_labels_raw = {col: _clean_column_label(col) for col in df.columns}
_label_counts = Counter(_column_labels_raw.values())
COLUMN_LABELS = {
    col: label if _label_counts[label] == 1 else col
    for col, label in _column_labels_raw.items()
}

def _format_column(col: str) -> str:
    return COLUMN_LABELS.get(col, col)

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
st.markdown(
    "Gebruik de interactieve grafieken om financiële stress en ervaren gezondheid vanuit meerdere invalshoeken te bekijken. "
    "Iedere sectie bevat een korte toelichting zodat je snel de juiste visualisatie kiest."
)

if st.button('🔄 Reset filters naar standaard', type='secondary'):
    for key, value in [
        ('scatter_color', DEFAULTS['scatter_color']),
        ('scatter_x', DEFAULTS['scatter_x']),
        ('scatter_y', DEFAULTS['scatter_y']),
        ('bar_variable', DEFAULTS['bar_variable']),
        ('box_variable', DEFAULTS['box_variable']),
        ('hist_select', DEFAULTS['hist_variable']),
        ('grouped_vars', DEFAULTS['grouped_variables']),
        ('map_sb', DEFAULTS['map_variable']),
        ('corr_fin', [col for col in FINANCIAL_COLUMNS if col in df.columns][:3]),
        ('corr_health', [col for col in HEALTH_COLUMNS if col in df.columns][:4])
    ]:
        if value is not None:
            st.session_state[key] = value
    for key in ['box_ms', 'hist_ms', 'hist_range', 'hist_color', 'scatter_regression']:
        if key in st.session_state:
            del st.session_state[key]
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.rerun()

# --- Scatterplot ---
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("#### Relatie tussen financiële stress en ervaren gezondheid")
        st.caption("Kies twee indicatoren om hun onderlinge verband te zien. Een regressielijn helpt de trend te duiden.")
        scatter_options = AVAILABLE_RELEVANT
        if not scatter_options:
            st.info("Geen relevante variabelen beschikbaar voor de scatterplot.")
        else:
            sb_scatter = st.selectbox('Kleur op', ['Geen', 'Provincie', 'Gemeente'], index=0, key='scatter_color')
            col_s1, col_s2 = st.columns(2)
            default_x = scatter_options.index(DEFAULTS['scatter_x']) if DEFAULTS['scatter_x'] in scatter_options else 0
            default_y = scatter_options.index(DEFAULTS['scatter_y']) if DEFAULTS['scatter_y'] in scatter_options else 0
            with col_s1:
                x_axis = st.selectbox('X-as', scatter_options, index=default_x, format_func=_format_column, key='scatter_x')
            with col_s2:
                y_axis = st.selectbox('Y-as', scatter_options, index=default_y, format_func=_format_column, key='scatter_y')
            add_regression = st.checkbox('Regressielijn', value=True, key='scatter_regression')

            scatter = alt.Chart(df).mark_circle().encode(
                x=alt.X(f'{x_axis}:Q', title=_format_column(x_axis), scale=alt.Scale(zero=False)),
                y=alt.Y(f'{y_axis}:Q', title=_format_column(y_axis), scale=alt.Scale(zero=False)),
                color=alt.Color(f'{sb_scatter}:N') if sb_scatter != 'Geen' else alt.value('steelblue'),
                tooltip=[
                    alt.Tooltip(x_axis, title=_format_column(x_axis)),
                    alt.Tooltip(y_axis, title=_format_column(y_axis))
                ]
            ).properties(
                width=800,
                height=800,
                title=f'{_format_column(x_axis)} vs. {_format_column(y_axis)}'
            ).interactive()

            if add_regression:
                scatter += scatter.transform_regression(x_axis, y_axis).mark_line(color='red')

            st.altair_chart(scatter, use_container_width=True)

# --- Bar chart ---
with col2:
    with st.container(border=True):
        st.markdown("#### Financiële indicatoren per provincie")
        st.caption("Selecteer een indicator om te vergelijken hoe provincies scoren. Numerieke variabelen tonen gemiddelden, categorische kolommen tonen aantallen.")
        bar_options = AVAILABLE_RELEVANT
        if not bar_options:
            st.info("Geen relevante variabelen beschikbaar voor de staafdiagram.")
        else:
            default_bar = bar_options.index(DEFAULTS['bar_variable']) if DEFAULTS['bar_variable'] in bar_options else 0
            selected_bar = st.selectbox('Selecteer Variabele', bar_options, index=default_bar, format_func=_format_column, key='bar_variable')

            if pd.api.types.is_numeric_dtype(df[selected_bar]):
                bar_data = df.groupby("Provincie", as_index=False)[selected_bar].mean()
            else:
                bar_data = df.groupby("Provincie", as_index=False)[selected_bar].count()

            bar_chart = alt.Chart(bar_data).mark_bar().encode(
                x=alt.X("Provincie:N", sort='-y', title="Provincie"),
                y=alt.Y(f"{selected_bar}:Q", title=_format_column(selected_bar)),
                tooltip=[
                    alt.Tooltip("Provincie:N", title="Provincie"),
                    alt.Tooltip(f"{selected_bar}:Q", title=_format_column(selected_bar))
                ]
            ).properties(width=700, height=400, title=f"{_format_column(selected_bar)} per provincie")

            st.altair_chart(bar_chart, use_container_width=True)

    # --- Boxplot ---
    with st.container(border=True):
        st.markdown("#### Spreiding binnen provincies")
        st.caption("Toon de verdeling van een indicator binnen geselecteerde provincies. Handig om uitschieters en verschillen in spreiding te bespreken.")
        if not bar_options:
            st.info("Geen relevante variabelen beschikbaar voor de boxplot.")
        else:
            box_ms = st.multiselect('Selecteer Provincies', provinces, default=provinces[:3], key='box_ms')
            default_box = bar_options.index(DEFAULTS['box_variable']) if DEFAULTS['box_variable'] in bar_options else 0
            box_sb = st.selectbox('X-as Boxplot', bar_options, index=default_box, format_func=_format_column, key='box_variable')
            df_filtered = df[df['Provincie'].isin(box_ms)]
            n_provs = max(len(box_ms), 1)
            box_size = max(10, 200 // n_provs)
            if box_ms:
                box = alt.Chart(df_filtered).mark_boxplot(size=box_size).encode(
                    x=alt.X(f'{box_sb}:Q', title=_format_column(box_sb), scale=alt.Scale(zero=False)),
                    y=alt.Y('Provincie:N', title='Provincie'),
                    color=alt.Color('Provincie:N', legend=None)
                ).properties(height=392, title = f'Boxplot van {_format_column(box_sb)} per provincie')
                st.altair_chart(box, use_container_width=True)
            else:
                st.info("Selecteer minstens één provincie om de boxplot te tonen.")

# --- Histogram ---
with st.container(border=True):
    if not bar_options:
        st.info("Geen relevante variabelen beschikbaar voor het histogram.")
    else:
        st.markdown("#### Verdeling binnen geselecteerde provincies")
        st.caption("Bekijk hoe een indicator binnen een bepaald bereik verdeeld is. Handig voor het spotten van clusters of uitschieters.")
        hist_ms = st.multiselect('Selecteer Provincies voor Histogram', provinces, default=provinces[:3], key='hist_ms')
        default_hist = bar_options.index(DEFAULTS['hist_variable']) if DEFAULTS['hist_variable'] in bar_options else 0
        selected_hist = st.selectbox('X-as Histogram', bar_options, index=default_hist, key='hist_select', format_func=_format_column)
        range_val = st.slider('Selecteer bereik', float(df[selected_hist].min()), float(df[selected_hist].max()), (float(df[selected_hist].min()), float(df[selected_hist].max())), key='hist_range', step = 0.1)
        kleur = st.checkbox('Kleur op Provincie', value=False, key='hist_color')
        df_hist_filtered = df[(df['Provincie'].isin(hist_ms)) & (df[selected_hist] >= range_val[0]) & (df[selected_hist] <= range_val[1])]

        if hist_ms:
            hist = alt.Chart(df_hist_filtered).mark_bar().encode(
                x=alt.X(f"{selected_hist}:Q", bin=alt.Bin(maxbins=30), title=_format_column(selected_hist)),
                y=alt.Y('count()', title='Aantal'),
                color=alt.Color('Provincie:N')
            ).properties(title=f'Verdeling van {_format_column(selected_hist)} per provincie')
            st.altair_chart(hist, use_container_width=True)
        else:
            st.info("Selecteer minstens één provincie om het histogram te tonen.")

# --- Grouped Bar ---
with st.container(border=True):
    if not bar_options:
        st.info("Geen relevante variabelen beschikbaar voor de grouped bar.")
    else:
        st.markdown("#### Vergelijk meerdere indicatoren tegelijk")
        st.caption("Selecteer twee of meer numerieke variabelen om hun gemiddelde per provincie naast elkaar te zetten.")
        numeric_options = [col for col in bar_options if pd.api.types.is_numeric_dtype(df[col])]
        stack_vars = st.multiselect(
            'Selecteer Variabelen voor Grouped Bar',
            numeric_options,
            default=DEFAULTS['grouped_variables'] if DEFAULTS['grouped_variables'] else numeric_options[:1],
            format_func=_format_column,
            key='grouped_vars'
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
            stack_long = stack_data.melt(id_vars='Provincie', value_vars=stack_vars, var_name='Variable', value_name='Value')
            stack_long['VariableLabel'] = stack_long['Variable'].map(_format_column)

            grouped_bar = (
                alt.Chart(stack_long)
                .mark_bar()
                .encode(
                    x=alt.X('Provincie:N', title="Provincie"),
                    xOffset='VariableLabel:N',
                    y=alt.Y('Value:Q', title="Gemiddelde waarde"),
                    color=alt.Color('VariableLabel:N', legend=alt.Legend(orient='bottom', title='Variabele')),
                    tooltip=[
                        alt.Tooltip('Provincie:N', title='Provincie'),
                        alt.Tooltip('VariableLabel:N', title='Variabele'),
                        alt.Tooltip('Value:Q', title='Gemiddelde waarde')
                    ]
                )
                .properties(width=700, height=600, title = 'Grouped Bar van geselecteerde variabelen per provincie')
            )

            st.altair_chart(grouped_bar, use_container_width=True)
        else:
            st.info("Selecteer minimaal één numerieke variabele om de grouped bar te tonen.")


# --- Correlatie Heatmap ---
with st.container(border=True):
    st.markdown("#### Correlatie tussen financiën en gezondheid")
    st.caption("Blauwe waarden staan voor positieve verbanden, rode cellen voor negatieve relaties. Gebruik dit overzicht om opvallende combinaties te detecteren.")
    financieel_vars = [col for col in FINANCIAL_COLUMNS if col in df.columns]
    gezondheid_leefstijl_vars = [col for col in HEALTH_COLUMNS if col in df.columns]

    fin_vars = st.multiselect('Selecteer financiële variabelen (X-as)', financieel_vars, default=financieel_vars[:3], format_func=_format_column, key='corr_fin')
    health_vars = st.multiselect('Selecteer gezondheid/leefstijl variabelen (Y-as)', gezondheid_leefstijl_vars, default=gezondheid_leefstijl_vars[:4], format_func=_format_column, key='corr_health')

    if fin_vars and health_vars:
        corr_df = df[fin_vars + health_vars].corr().loc[fin_vars, health_vars]
        corr_long = corr_df.reset_index().melt(id_vars='index', var_name='HealthVar', value_name='Correlation')
        corr_long['FinLabel'] = corr_long['index'].map(_format_column)
        corr_long['HealthLabel'] = corr_long['HealthVar'].map(_format_column)

        corr_chart = alt.Chart(corr_long).mark_rect().encode(
            x=alt.X('HealthLabel:O', title='Gezondheid/Leefstijl'),
            y=alt.Y('FinLabel:O', title='Financieel'),
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1))),
            tooltip=[
                alt.Tooltip('FinLabel:N', title='Financieel'),
                alt.Tooltip('HealthLabel:N', title='Gezondheid/Leefstijl'),
                alt.Tooltip('Correlation:Q', title='Correlatie')
            ]
        ).properties(width=600, height=650, title='Correlatie Financieel ↔ Gezondheid/Leefstijl')

        st.altair_chart(corr_chart, use_container_width=True)
    else:
        st.info("Selecteer minstens één financiële en één gezondheids-/leefstijlvariabele om de correlatie te tonen.")

st.divider()

if not bar_options:
    st.info("Geen relevante variabelen beschikbaar voor de kaartvisualisaties.")
else:
    st.markdown("#### Regionale verschillen zichtbaar maken")
    st.caption("Selecteer een indicator en kies vervolgens of je de kaart per gemeente of per provincie wilt bekijken.")
    baseline_col = 'MoeiteMetRondkomen_1' if 'MoeiteMetRondkomen_1' in df.columns else bar_options[0]
    df_map = df[['RegioS', baseline_col]].copy()
    df_map = df_map.rename(columns={baseline_col: 'val'})

    default_map_index = bar_options.index(DEFAULTS['map_variable']) if DEFAULTS['map_variable'] in bar_options else 0
    map_sb = st.selectbox('Kies variabele voor de kaart', bar_options, index=default_map_index, key='map_sb', format_func=_format_column)

    gdf_map = gdf[['statcode', 'statnaam', 'geometry']].copy()
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
        variable_label = _format_column(_variable)
        m = folium.Map(location=[52.1, 5.3], zoom_start=7, tiles='CartoDB positron')
        colormap = linear.Blues_09.scale(_gdf['val'].min(), _gdf['val'].max())
        colormap.caption = variable_label
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
        tooltip_aliases.append(f'{variable_label}:')

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

    gdf_prov = gdf_map.merge(df[['RegioS', 'Provincie']], left_on='statcode', right_on='RegioS', how='left')

    gdf_prov = gdf_prov.dissolve(by='Provincie', aggfunc={'val': 'mean'}).reset_index()
    gdf_prov['statnaam'] = gdf_prov['Provincie']  # tooltip

    df['RegioS'] = df['RegioS'].astype(str).str.strip()
    df_geo['Gemeente code (with prefix)'] = df_geo['Gemeente code (with prefix)'].astype(str).str.strip()

    with col_m2:
        with st.container(border=True, height=900):
            st.subheader('Statistieken per gebied')
            st.caption("Gebruik deze samenvatting om de uitersten en spreiding per regio te benoemen in je verhaal.")
            map_sb2 = st.selectbox('Selecteer', ['Gemeenten', 'Provincies'], index=0)

            if map_sb2 == 'Gemeenten':
                gdf_to_show = gdf_map
                df_stats = df.copy()
                df_stats['Naam'] = df_stats['RegioS'].map(
                    gdf[['statcode', 'statnaam']].drop_duplicates().set_index('statcode')['statnaam']
                ).fillna(df_stats['Provincie'])
                value_col = map_sb
                area_label = 'Gemeente'
            else:
                gdf_to_show = gdf_prov
                df_stats = gdf_prov.copy()
                df_stats['Naam'] = df_stats['Provincie']
                value_col = 'val'
                area_label = 'Provincie'

            st.write(f"Samenvattende statistieken voor {_format_column(map_sb)}:")
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
            st.dataframe(
                df_stats.nlargest(3, value_col)[['Naam', value_col]].rename(columns={value_col: _format_column(map_sb)}),
                hide_index=True
            )

            st.write("Laagste 3:")
            st.dataframe(
                df_stats.nsmallest(3, value_col)[['Naam', value_col]].rename(columns={value_col: _format_column(map_sb)}),
                hide_index=True
            )

    with col_m1:
        with st.container(border=True):
            st.subheader(f"Kaart van {_format_column(map_sb)} per {area_label}")
            st.caption("Donkerdere kleuren wijzen op hogere waarden voor de geselecteerde indicator.")
            m = maak_kaart(gdf_to_show, map_sb)
            st_folium(m, width=700, height=800)

