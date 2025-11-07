import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
from streamlit_folium import st_folium
import numpy as np
from branca.colormap import linear
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings
import load_data

warnings.filterwarnings('ignore')

st.set_page_config('Statistische Analyse', layout='wide', page_icon='🔍')

df, df_geo, df_shp, gdf_merged, gdf, gdf_map = load_data.load_data()

df = df.drop(columns=df.columns[df.isna().sum() > 75])

# --- Nieuwe variabelen creëren ---
df['FinancieelRisicoScore'] = df[['MoeiteMetRondkomen_1', 'WeinigControleOverGeldzaken_2', 'HeeftSchulden_3']].mean(axis=1)
df['MentaleGezondheidsScore'] = df[['GoedErvarenMentaleGezondheid_12', 'AngstDepressiegevoelensAfg4Weken_13', 'BeperktDoorPsychischeKlachten_14']].mean(axis=1)
df['ervarengezondheid'] = df[['GoedErvarenGezondheid_6', 'GoedErvarenMentaleGezondheid_12']].mean(axis=1)

bins = [0, 10, 30, 100]
labels = ['Laag', 'Gemiddeld', 'Hoog']
df['MoeiteMetRondkomenCat'] = pd.cut(df['MoeiteMetRondkomen_1'], bins=bins, labels=labels, right=False)


st.subheader('Statistische Analyse: Correlatie en Regressie')
with st.container(border=True):
    st.write("#### Correlatie tussen Financiën, Leefstijl en Gezondheid")
    corr_vars = [
        'ervarengezondheid',
        'FinancieelRisicoScore', 
        'MentaleGezondheidsScore', 
        'GoedErvarenMentaleGezondheid_12',
        'GoedErvarenGezondheid_6',
        'MoeiteMetRondkomen_1', 
        'WeinigControleOverGeldzaken_2', 
        'HeeftSchulden_3', 
        'ZorgenOverStudieschuld_5',
        'RooktTabak_75',
        'Overgewicht_59',
        'SportWekelijks_66',
        'ZwareDrinker_72',
        'CannabisInAfg12Maanden_89'
    ]
    
    corr_matrix = df[corr_vars].corr()
    st.dataframe(corr_matrix)
    
    st.write("#### Meervoudige Lineaire Regressie")
    st.write("Dit model voorspelt 'ervarengezondheid' op basis van financiële en leefstijl variabelen.")

    X_vars = [
        'MoeiteMetRondkomen_1', 
        'WeinigControleOverGeldzaken_2', 
        'HeeftSchulden_3', 
        'ZorgenOverStudieschuld_5',
        'RooktTabak_75',
        'Overgewicht_59',
        'SportWekelijks_66',
        'ZwareDrinker_72',
        'CannabisInAfg12Maanden_89'
    ]
    
    # Gebruik een subset van de data zonder ontbrekende waarden
    df_reg = df.dropna(subset=X_vars + ['ervarengezondheid'])
    X = df_reg[X_vars]
    y = df_reg['ervarengezondheid']

    if not X.empty and len(X) > 1:
        # Splits de data voor validatie van de voorspellende kracht
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Gebruik sklearn voor de prestatie metrics
        model_sk = LinearRegression()
        model_sk.fit(X_train, y_train)
        y_pred = model_sk.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        n = y_test.shape[0]  # Aantal observaties
        p = X_test.shape[1]  # Aantal predictoren

        # Bereken de adjusted R-squared
        r2_adj = 1 - (1-r2) * (n-1) / (n-p-1)

        st.write(f"R-kwadraat (R²) score: {r2:.3f}")
        st.write(f"Adjusted R-kwadraat (R² adj) score: {r2_adj:.3f}")
        st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.3f}")

        st.write("#### Coëfficiënten van de variabelen")
        coefficients = pd.DataFrame(model_sk.coef_, X.columns, columns=['Coëfficiënt'])
        st.dataframe(coefficients)
        st.write(f"Intercept: {model_sk.intercept_:.3f}")
        
        # --- Residuen plot toevoegen ---
        st.write("#### Residuen Plot")
        st.write("Deze plot toont de residuen (de fouten van het model) ten opzichte van de voorspelde waarden. Een goed model laat een willekeurige spreiding van de punten rond de horizontale lijn op y=0 zien.")

        residuals = y_test - y_pred
        residuals_df = pd.DataFrame({'Voorspelde Waarden': y_pred, 'Residuen': residuals})
        
        # Bepaal het bereik voor de x-as
        x_min = 50
        x_max = residuals_df['Voorspelde Waarden'].max()

        chart = alt.Chart(residuals_df).mark_circle(size=60).encode(
            x=alt.X('Voorspelde Waarden', title='Voorspelde ' + 'ervarengezondheid', scale=alt.Scale(domain=[x_min, x_max])),
            y=alt.Y('Residuen', title='Residuen'),
            tooltip=['Voorspelde Waarden', 'Residuen']
        ).properties(
            title='Residuen Plot'
        ).interactive()

        rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red').encode(y='y')
        
        final_chart = chart + rule
        st.altair_chart(final_chart, use_container_width=True)

        # --- Histogram van Residuen ---
        st.write("#### Histogram van Residuen")
        st.write("Dit histogram toont de verdeling van de voorspellingsfouten. Voor een goed lineair model verwacht je een verdeling die lijkt op een normale verdeling (een klokvormige curve), gecentreerd rond nul.")
        
        hist_chart = alt.Chart(residuals_df).mark_bar().encode(
            x=alt.X('Residuen:Q', bin=alt.Bin(maxbins=30), title='Residuen'),
            y=alt.Y('count()', title='Aantal')
        ).properties(
            title='Verdeling van de Residuen'
        ).interactive()
        
        st.altair_chart(hist_chart, use_container_width=True)
        
        # --- Cook's Distance Plot toevoegen ---
        st.write("#### Cook's Distance Plot")
        st.write("Deze plot identificeert invloedrijke datapunten. Punten met een hoge Cook's afstand hebben een grote invloed op de regressielijn en kunnen de resultaten vertekenen.")

        # Gebruik statsmodels voor regressiediagnostiek
        # OLS: Ordinary Least Squares
        X_sm = sm.add_constant(X)
        model_sm = sm.OLS(y, X_sm).fit()
        
        # Bereken Cook's afstand
        cooks_distance = model_sm.get_influence().cooks_distance[0]
        cooks_df = pd.DataFrame({'Index': range(len(cooks_distance)), 'Cooks_Distance': cooks_distance})
        
        # Altair plot
        cooks_chart = alt.Chart(cooks_df).mark_circle(size=60).encode(
            x=alt.X('Index', title='Index van Datapunt'),
            y=alt.Y('Cooks_Distance', title='Cooks Afstand', scale=alt.Scale(domain=[0, 0.4])),
            tooltip=['Cooks_Distance']
        ).properties(
            title='Cooks Afstand per Datapunt'
        ).interactive()
        
        # Voeg een rode drempellijn toe
        cooks_threshold = 0.05
        rule_cooks = alt.Chart(pd.DataFrame({'y': [cooks_threshold]})).mark_rule(color='red').encode(y='y')
        
        final_cooks_chart = cooks_chart + rule_cooks
        st.altair_chart(final_cooks_chart, use_container_width=True)

        st.write("#### Onderzoek van Invloedrijke Punten")
        st.write("Hieronder zie je de data voor de twee datapunten met de grootste Cook's distance. Dit kan je helpen te bepalen of er datakwaliteitsproblemen zijn of dat het om uitzonderlijke, maar correcte, observaties gaat.")
        
        # Zoek de indices van de top 2 invloedrijke punten
        cooks_df['Original_Index'] = y.index.values
        influential_indices = cooks_df.sort_values(by='Cooks_Distance', ascending=False).head(4)['Original_Index'].tolist()

        # Haal de data op van de invloedrijke punten, inclusief alle relevante variabelen en geografische info
        # Let op: we gebruiken de originele df voor alle data
        influential_points_data = df.loc[influential_indices, X_vars + ['ervarengezondheid', 'Gemeente code (with prefix)', 'Provincie']]

        # Maak een lijst van de gewenste kolomvolgorde
        cols = ['Gemeente code (with prefix)', 'Provincie'] + X_vars + ['ervarengezondheid']
        
        # Sorteer de kolommen
        influential_points_data = influential_points_data[cols]

        # Toon de tabel met invloedrijke punten
        st.dataframe(influential_points_data)

        st.write("#### Vergelijking met het Gemiddelde")
        st.write("Om te zien of de invloedrijke datapunten uitschieters zijn, vergelijken we hun waarden met het gemiddelde van de dataset die gebruikt is voor het model.")
        
        # Bereken het gemiddelde van de X-variabelen voor de regressie dataset
        average_values = X.mean().to_frame().T
        average_values.index = ['Gemiddelde']
        
        # Combineer de invloedrijke punten en het gemiddelde in één dataframe
        comparison_df = pd.concat([influential_points_data[X_vars], average_values])
        st.dataframe(comparison_df)
    else:
        st.warning("De geselecteerde variabelen bevatten te veel missende waarden voor regressie-analyse.")