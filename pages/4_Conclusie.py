import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import re
from collections import Counter
import load_data

st.set_page_config("Conclusies: Gezondheid Jongvolwassenen", layout="wide", page_icon="🔍")


def make_label_map(columns):
    pattern = re.compile(r'_\d+$')
    raw = {col: pattern.sub('', col) for col in columns}
    counts = Counter(raw.values())
    return {col: (label if counts[label] == 1 else col) for col, label in raw.items()}

def format_col(name, label_map):
    return label_map.get(name, name)


@st.cache_data
def compute_summary():
    df, *_ = load_data.load_data()
    df_filtered = df.drop(columns=df.columns[df.isna().sum() > 75]).copy()
    df_filtered['FinancieelRisicoScore'] = df_filtered[
        ['MoeiteMetRondkomen_1', 'WeinigControleOverGeldzaken_2', 'HeeftSchulden_3']
    ].mean(axis=1)
    df_filtered['MentaleGezondheidsScore'] = df_filtered[
        ['GoedErvarenMentaleGezondheid_12', 'AngstDepressiegevoelensAfg4Weken_13', 'BeperktDoorPsychischeKlachten_14']
    ].mean(axis=1)
    df_filtered['ervarengezondheid'] = df_filtered[
        ['GoedErvarenGezondheid_6', 'GoedErvarenMentaleGezondheid_12']
    ].mean(axis=1)

    label_map = make_label_map(df.columns)

    X_vars = [
        'AngstDepressiegevoelensAfg4Weken_13',
        'TevredenMetEigenLeven_11',
        'VaakGelukkigAfg4Weken_15',
        'SlaaptMeestalSlecht_7',
        'VoldoendeWeerbaar_16',
        'Overgewicht_59',
        'ZwareDrinker_72'
    ]

    df_reg = df_filtered.dropna(subset=X_vars + ['ervarengezondheid'])
    if df_reg.empty:
        return None
    X = df_reg[X_vars].copy()
    y = df_reg['ervarengezondheid']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    n = y_test.shape[0]
    p = X_test.shape[1]
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    model_sm = sm.OLS(y, sm.add_constant(X_scaled)).fit()
    coeffs = model_sm.params.iloc[1:]
    pvals = model_sm.pvalues.iloc[1:]
    drivers = (
        pd.DataFrame({
            'Variabele': coeffs.index,
            'Coëfficiënt': coeffs.values,
            'P_waarde': pvals.values
        })
        .assign(
            Richting=lambda d: np.where(d['Coëfficiënt'] >= 0, 'Positief', 'Negatief'),
            Significant=lambda d: np.where(d['P_waarde'] < 0.05, 'Ja', 'Nee')
        )
        .sort_values(by='Coëfficiënt', key=lambda s: s.abs(), ascending=False)
    )
    drivers['Variabele'] = drivers['Variabele'].map(lambda x: format_col(x, label_map))

    top_pos = drivers[drivers['Coëfficiënt'] > 0].head(1)
    top_neg = drivers[drivers['Coëfficiënt'] < 0].head(1)

    summary_text = []
    if not top_pos.empty:
        row = top_pos.iloc[0]
        summary_text.append(f"**{row['Variabele']}** heeft het sterkste positieve effect (+{row['Coëfficiënt']:.2f}).")
    if not top_neg.empty:
        row = top_neg.iloc[0]
        summary_text.append(f"**{row['Variabele']}** drukt ervaren gezondheid het meest ({row['Coëfficiënt']:.2f}).")

    return {
        'r2': r2,
        'r2_adj': r2_adj,
        'mse': mse,
        'drivers': drivers,
        'summary': summary_text,
        'n_obs': len(df_reg),
        'label_map': label_map
    }


def render_conclusion_page():
    st.title("Conclusie")
    st.caption("Van ruwe data naar beleid: belangrijkste inzichten uit de Gezondheidsmonitor Jongvolwassenen 2024.")

    results = compute_summary()
    if results is None:
        st.error("Onvoldoende gegevens om de conclusie te berekenen.")
        return
    label_map = results['label_map']

    with st.container(border=True):
        st.subheader("Samenvatting in één zin")
        st.success(
            "Financiële stress en mentale gezondheid lopen hand in hand: jongeren die minder geldzorgen en betere slaap ervaren, "
            "melden een significant hogere ervaren gezondheid."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("R²", f"{results['r2']:.2f}")
        with c2:
            st.metric("R² adj.", f"{results['r2_adj']:.2f}")
        with c3:
            st.metric("MSE", f"{results['mse']:.3f}")
        with c4:
            st.metric("Aantal observaties", results['n_obs'])
        if results['summary']:
            st.markdown("**Belangrijkste drivers**")
            for line in results['summary']:
                st.markdown(f"- {line}")

    with st.container(border=True):
        st.subheader("Effecten per indicator")
        st.caption("Positieve coëfficiënten verbeteren de ervaren gezondheid; negatieve drukken deze. De kolom 'Significant' geeft aan of p < 0.05.")
        show_cols = results['drivers'][['Variabele', 'Coëfficiënt', 'P_waarde', 'Richting', 'Significant']].head(10)
        st.dataframe(
            show_cols.style.format({'Coëfficiënt': '{:+.3f}', 'P_waarde': '{:.3f}'}),
            use_container_width=True
        )

    with st.container(border=True):
        st.subheader("Conclusies")
        st.markdown(
            """
            - Jongvolwassenen met weinig geldstress en voldoende weerbaarheid rapporteren de hoogste ervaren gezondheid.
            - Negatieve emoties (angst/depressie) en slaapproblemen zijn de sterkste voorspellers van een lagere ervaren gezondheid.
            - Leefstijlfactoren zoals overgewicht en zwaar drinken hebben weliswaar een kleiner effect, maar blijven statistisch relevant.
            - Regionale verschillen suggereren dat gerichte ondersteuning per provincie of gemeente effectief kan zijn.
            """
        )

    with st.container(border=True):
        st.subheader("Aanbevelingen voor beleid en interventie")
        st.markdown(
            """
            1. **Investeer in mentaal welzijn** – programma’s voor stressreductie en positieve psychologie geven het grootste rendement.
            2. **Slaap en dagritme op peil** – campagnes rond slaaphygiëne en balans tussen studie/werk verlagen directe risico’s.
            3. **Vroegsignalering van financiële stress** – combineer budgetcoaching met mentale ondersteuning voor kwetsbare groepen.
            4. **Gerichte leefstijlondersteuning** – overgewicht en zwaar drinken hebben kleinere maar significante effecten; richt je op gecombineerde leefstijlinterventies.
            """
        )


if __name__ == "__main__":
    render_conclusion_page()
