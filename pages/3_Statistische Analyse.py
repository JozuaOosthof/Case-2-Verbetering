import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import warnings
import load_data

warnings.filterwarnings('ignore')

st.set_page_config('Statistische Analyse', layout='wide', page_icon='🔍')


def prepare_dataframe(dataframe):
    df_filtered = dataframe.drop(columns=dataframe.columns[dataframe.isna().sum() > 75]).copy()
    df_filtered['FinancieelRisicoScore'] = df_filtered[['MoeiteMetRondkomen_1', 'WeinigControleOverGeldzaken_2', 'HeeftSchulden_3']].mean(axis=1)
    df_filtered['MentaleGezondheidsScore'] = df_filtered[['GoedErvarenMentaleGezondheid_12', 'AngstDepressiegevoelensAfg4Weken_13', 'BeperktDoorPsychischeKlachten_14']].mean(axis=1)
    df_filtered['ervarengezondheid'] = df_filtered[['GoedErvarenGezondheid_6', 'GoedErvarenMentaleGezondheid_12']].mean(axis=1)
    bins = [0, 10, 30, 100]
    labels = ['Laag', 'Gemiddeld', 'Hoog']
    df_filtered['MoeiteMetRondkomenCat'] = pd.cut(df_filtered['MoeiteMetRondkomen_1'], bins=bins, labels=labels, right=False)
    return df_filtered


def render_correlation_section(dataframe):
    st.write("#### Correlatie tussen Financiën, Leefstijl en Gezondheid")
    st.write("""
    Een correlatiematrix laat zien hoe sterk variabelen met elkaar samenhangen:
    - Een waarde van 1 betekent een perfecte positieve correlatie
    - Een waarde van -1 betekent een perfecte negatieve correlatie
    - Een waarde van 0 betekent geen correlatie
    - Algemeen: |r| > 0.7 is sterk, 0.3 < |r| < 0.7 is matig, |r| < 0.3 is zwak
    """)
    all_vars = {
        'Gezondheid': ['ervarengezondheid', 'GoedErvarenGezondheid_6', 'GoedErvarenMentaleGezondheid_12'],
        'Financieel': ['FinancieelRisicoScore', 'MoeiteMetRondkomen_1', 'WeinigControleOverGeldzaken_2', 'HeeftSchulden_3', 'ZorgenOverStudieschuld_5'],
        'Leefstijl': ['RooktTabak_75', 'Overgewicht_59', 'SportWekelijks_66', 'ZwareDrinker_72', 'CannabisInAfg12Maanden_89'],
        'Samengestelde Scores': ['FinancieelRisicoScore', 'MentaleGezondheidsScore']
    }
    selected_categories = st.multiselect(
        "Selecteer categorieën voor de correlatiematrix",
        list(all_vars.keys()),
        default=['Gezondheid', 'Financieel']
    )
    corr_vars = []
    for category in selected_categories:
        corr_vars.extend(all_vars[category])
    corr_vars = list(dict.fromkeys(corr_vars))
    if corr_vars:
        corr_matrix = dataframe[corr_vars].corr()
        corr_data = corr_matrix.reset_index().melt(
            id_vars='index',
            var_name='variable',
            value_name='correlation'
        )
        heatmap = alt.Chart(corr_data).mark_rect().encode(
            x=alt.X('variable:N', title=None),
            y=alt.Y('index:N', title=None),
            color=alt.Color(
                'correlation:Q',
                scale=alt.Scale(domain=[-1, 1], scheme='redblue'),
                legend=alt.Legend(title="Correlatie")
            ),
            tooltip=[
                alt.Tooltip('index:N', title='Variabele 1'),
                alt.Tooltip('variable:N', title='Variabele 2'),
                alt.Tooltip('correlation:Q', title='Correlatie', format='.2f')
            ]
        ).properties(
            width=600,
            height=400,
            title='Correlatiematrix'
        )
        text = alt.Chart(corr_data).mark_text(baseline='middle').encode(
            x=alt.X('variable:N'),
            y=alt.Y('index:N'),
            text=alt.Text('correlation:Q', format='.2f'),
            color=alt.condition(
                'datum.correlation > 0.5 || datum.correlation < -0.5',
                alt.value('white'),
                alt.value('black')
            )
        )
        st.altair_chart(heatmap + text, use_container_width=True)
        with st.expander("Toon correlatiematrix als tabel"):
            st.dataframe(corr_matrix)


def render_transformation_options(y, histogram_data):
    st.write("#### Transformatie van de Doelvariabele (ervarengezondheid)")
    st.write("""
    Transformaties kunnen helpen om:
    - De normaliteit van residuen te verbeteren
    - Heteroscedasticiteit te verminderen
    - De lineaire relatie tussen variabelen te versterken
    """)
    option = st.selectbox(
        "Kies een transformatie voor de target:",
        ["Geen", "Log", "Sqrt"],
        index=2
    )
    hist_orig = alt.Chart(pd.DataFrame({'Waarde': histogram_data})).mark_bar().encode(
        x=alt.X('Waarde:Q', bin=True, title='Originele waarde'),
        y=alt.Y('count():Q', title='Frequentie')
    ).properties(title='Originele verdeling', height=200)
    if option == "Log":
        if (y <= 0).any():
            st.warning("⚠️ Log-transformatie niet mogelijk: data bevat negatieve of nulwaarden.")
            st.altair_chart(hist_orig, use_container_width=True)
            return y
        transformed = np.log(y)
        st.success("✅ Log-transformatie succesvol toegepast op de target.")
        hist_trans = alt.Chart(pd.DataFrame({'Waarde': transformed})).mark_bar().encode(
            x=alt.X('Waarde:Q', bin=True, title='Log-getransformeerde waarde'),
            y=alt.Y('count():Q', title='Frequentie')
        ).properties(title='Verdeling na log-transformatie', height=200)
        st.write("#### Effect van de Transformatie")
        st.altair_chart(alt.hconcat(hist_orig, hist_trans), use_container_width=True)
        return transformed
    if option == "Sqrt":
        if (y < 0).any():
            st.warning("⚠️ Vierkantswortel-transformatie niet mogelijk: data bevat negatieve waarden.")
            st.altair_chart(hist_orig, use_container_width=True)
            return y
        transformed = np.sqrt(y)
        st.success("✅ Vierkantswortel-transformatie succesvol toegepast op de target.")
        hist_trans = alt.Chart(pd.DataFrame({'Waarde': transformed})).mark_bar().encode(
            x=alt.X('Waarde:Q', bin=True, title='Vierkantswortel-getransformeerde waarde'),
            y=alt.Y('count():Q', title='Frequentie')
        ).properties(title='Verdeling na vierkantswortel-transformatie', height=200)
        st.write("#### Effect van de Transformatie")
        st.altair_chart(alt.hconcat(hist_orig, hist_trans), use_container_width=True)
        return transformed
    st.info("ℹ️ Geen transformatie toegepast op de target.")
    st.altair_chart(hist_orig, use_container_width=True)
    return y


def render_vif_table(X):
    st.write("#### Analyse van Multicollineariteit (VIF)")
    st.write("""
    De Variance Inflation Factor (VIF) helpt bij het identificeren van multicollineariteit tussen voorspellende variabelen:
    - VIF = 1: Geen correlatie met andere variabelen
    - 1 < VIF < 5: Matige correlatie, meestal acceptabel
    - VIF > 5: Hoge correlatie, mogelijk problematisch
    - VIF > 10: Zeer hoge correlatie, moet worden aangepakt
    """)
    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variabele"] = X_with_const.columns[1:]
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(1, X_with_const.shape[1])]
    def vif_color(value):
        if value < 1.5:
            return 'lightgreen'
        if value < 5:
            return 'yellow'
        return 'lightcoral'
    def style_row(row):
        return ['background-color: ' + vif_color(val) if idx == 1 else '' for idx, val in enumerate(row)]
    st.dataframe(vif_data.style.apply(style_row, axis=1))


def render_feature_importance(final_model, X, y):
    y_pred_full = final_model.predict(X)
    mse_full = mean_squared_error(y, y_pred_full)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    dof = n_samples - n_features - 1
    X_with_const = sm.add_constant(X)
    variance_matrix = mse_full * np.linalg.inv(np.dot(X_with_const.T, X_with_const))
    se = np.sqrt(np.diag(variance_matrix)[1:])
    t_stats = final_model.coef_ / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    feature_importance = pd.DataFrame({
        'Variabele': X.columns,
        'Coëfficiënt': final_model.coef_,
        'Std_Error': se,
        'T_Stat': t_stats,
        'P_waarde': p_values
    })
    feature_importance = feature_importance.iloc[np.abs(feature_importance['Coëfficiënt']).argsort()[::-1]].reset_index(drop=True)
    importance_chart = alt.Chart(feature_importance).mark_bar().encode(
        x=alt.X('Coëfficiënt:Q', title='Coëfficiënt'),
        y=alt.Y('Variabele:N', sort=alt.EncodingSortField(field='Coëfficiënt', order='descending'), title=''),
        color=alt.condition(
            'datum.P_waarde < 0.05',
            alt.value('steelblue'),
            alt.value('lightgrey')
        ),
        tooltip=[
            alt.Tooltip('Variabele:N', title='Variabele'),
            alt.Tooltip('Coëfficiënt:Q', format='.3f'),
            alt.Tooltip('Std_Error:Q', format='.3f', title='Std. Error'),
            alt.Tooltip('T_Stat:Q', format='.3f', title='T-statistiek'),
            alt.Tooltip('P_waarde:Q', format='.3f', title='P-waarde')
        ]
    ).properties(
        title='Feature Importance (donkerblauw = statistisch significant, p < 0.05)',
        width=600,
        height=400
    )
    st.altair_chart(importance_chart, use_container_width=True)
    with st.expander("Toon gedetailleerde statistieken"):
        styled_stats = feature_importance.style.format({
            'Coëfficiënt': '{:.3f}',
            'Std_Error': '{:.3f}',
            'T_Stat': '{:.3f}',
            'P_waarde': '{:.3f}'
        }).background_gradient(
            subset=['P_waarde'],
            cmap='RdYlGn_r',
            vmin=0,
            vmax=0.1
        )
        st.dataframe(styled_stats)
    significant_features = feature_importance[feature_importance['P_waarde'] < 0.05]
    if significant_features.empty:
        return
    st.write("#### Significante Features (p < 0.05)")
    st.write("De volgende variabelen hebben een statistisch significant effect op de ervaren gezondheid:")
    for _, row in significant_features.iterrows():
        effect = "positief" if row['Coëfficiënt'] > 0 else "negatief"
        st.write(f"- **{row['Variabele']}**: {effect} effect (coëf. = {row['Coëfficiënt']:.3f}, p = {row['P_waarde']:.3f})")


def render_residual_plots(y_test, y_pred, model_sk, X, y, df_reg, X_vars, residuals_model):
    st.write(f"Intercept: {model_sk.intercept_:.3f}")
    st.write("#### Residuen Plot")
    st.write("Deze plot toont de residuen (de fouten van het model) ten opzichte van de voorspelde waarden. Een goed model laat een willekeurige spreiding van de punten rond de horizontale lijn op y=0 zien.")
    residuals = y_test - y_pred
    residuals_df = pd.DataFrame({'Voorspelde Waarden': y_pred, 'Residuen': residuals})
    chart = alt.Chart(residuals_df).mark_circle(size=60).encode(
        x=alt.X('Voorspelde Waarden', title='Voorspelde ervarengezondheid', scale=alt.Scale(zero=False)),
        y=alt.Y('Residuen', title='Residuen'),
        tooltip=['Voorspelde Waarden', 'Residuen']
    ).properties().interactive()
    rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red').encode(y='y')
    st.altair_chart(chart + rule, use_container_width=True)
    st.write("#### Residuen vs Voorspelde Waarden (Heteroscedasticiteit)")
    st.write("Deze plot laat zien of de spreiding van de residuen verandert bij verschillende voorspelde waarden. Een trechtervorm duidt op heteroscedasticiteit.")
    resid_fit_df = pd.DataFrame({'Fitted': model_sk.predict(X), 'Residuen': y - model_sk.predict(X)})
    resid_fit_chart = alt.Chart(resid_fit_df).mark_circle(size=60).encode(
        x=alt.X('Fitted', title='Voorspelde Waarde', scale=alt.Scale(zero=False)),
        y=alt.Y('Residuen', title='Residuen'),
        tooltip=['Fitted', 'Residuen']
    ).properties(title='Residuen vs Voorspelde Waarden').interactive()
    st.altair_chart(resid_fit_chart, use_container_width=True)
    st.write("#### Breusch-Pagan Test op Heteroscedasticiteit")
    X_bp = sm.add_constant(X)
    bp_test = het_breuschpagan(y - model_sk.predict(X), X_bp)
    bp_labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    bp_results = dict(zip(bp_labels, bp_test))
    st.write(f"Lagrange multiplier statistic: {bp_results['Lagrange multiplier statistic']:.3f}")
    st.write(f"p-value: {bp_results['p-value']:.3f}")
    st.write(f"F-statistiek: {bp_results['f-value']:.3f}")
    st.write(f"F p-value: {bp_results['f p-value']:.3f}")
    if bp_results['p-value'] < 0.05:
        st.error("Er is statistisch significante heteroscedasticiteit aanwezig (p < 0.05). Resultaten hieronder zijn met robuuste standaardfouten (HC3).")
        robust_model = sm.OLS(y, sm.add_constant(X)).fit(cov_type='HC3')
        robust_summary = robust_model.summary2().tables[1].reset_index()
        colmap = {'index': 'Variabele', 'Coef.': 'Coëfficiënt', 'Std.Err.': 'Robuuste Std.Error'}
        t_col = next((c for c in robust_summary.columns if c.lower().startswith('t')), None)
        p_col = next((c for c in robust_summary.columns if 'p' in c.lower() and '|' in c), None)
        if t_col:
            colmap[t_col] = 'T_waarde'
        if p_col:
            colmap[p_col] = 'P_waarde'
        robust_summary = robust_summary.rename(columns=colmap)
        st.write("#### Coëfficiënten met Robuuste Standaardfouten (HC3)")
        show_cols = ['Variabele', 'Coëfficiënt', 'Robuuste Std.Error']
        if 'T_waarde' in robust_summary.columns:
            show_cols.append('T_waarde')
        if 'P_waarde' in robust_summary.columns:
            show_cols.append('P_waarde')
        st.dataframe(robust_summary[show_cols].style.format({c: '{:.3f}' for c in show_cols if c != 'Variabele'}))
    else:
        st.success("Geen statistisch significante heteroscedasticiteit gevonden (p >= 0.05).")
    st.write("#### QQ-plot van Residuen")
    st.write("Deze plot vergelijkt de verdeling van de residuen met een normale verdeling. Punten moeten ongeveer op de rode lijn liggen voor een goed model.")
    std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    qq = stats.probplot(std_residuals, dist="norm")
    qq_df = pd.DataFrame({'Theoretisch': qq[0][0], 'Geobserveerd': qq[0][1]})
    min_q = min(qq_df['Theoretisch'].min(), qq_df['Geobserveerd'].min()) - 0.1
    max_q = max(qq_df['Theoretisch'].max(), qq_df['Geobserveerd'].max()) + 0.1
    line_df = pd.DataFrame({'x': np.linspace(min_q, max_q, 100), 'y': np.linspace(min_q, max_q, 100)})
    qq_chart = alt.Chart(qq_df).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X('Theoretisch', title='Theoretische Kwantielen', scale=alt.Scale(domain=[min_q, max_q])),
        y=alt.Y('Geobserveerd', title='Gestandaardiseerde Residuen', scale=alt.Scale(domain=[min_q, max_q])),
        tooltip=[
            alt.Tooltip('Theoretisch', format='.3f'),
            alt.Tooltip('Geobserveerd', format='.3f')
        ]
    ).properties(
        title='QQ-plot van Residuen',
        width=600,
        height=400
    )
    diagonal = alt.Chart(line_df).mark_line(color='red', strokeWidth=2).encode(x='x', y='y')
    st.altair_chart(alt.layer(qq_chart, diagonal).interactive(), use_container_width=True)
    st.write("**Statistieken van de residuen:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gemiddelde", f"{np.mean(residuals):.3f}")
    with col2:
        st.metric("Standaarddeviatie", f"{np.std(residuals):.3f}")
    with col3:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        st.metric("Shapiro-Wilk p-waarde", f"{shapiro_p:.3f}")
 
    st.write("#### Histogram van Residuen")
    st.write("Dit histogram toont de verdeling van de voorspellingsfouten. Voor een goed lineair model verwacht je een verdeling die lijkt op een normale verdeling (een klokvormige curve), gecentreerd rond nul.")
    hist_chart = alt.Chart(residuals_df).mark_bar().encode(
        x=alt.X('Residuen:Q', bin=alt.Bin(maxbins=30), title='Residuen'),
        y=alt.Y('count()', title='Aantal')
    ).properties(title='Verdeling van de Residuen').interactive()
    st.altair_chart(hist_chart, use_container_width=True)
    st.write("#### Cook's Distance Plot")
    st.write("Deze plot identificeert invloedrijke datapunten. Punten met een hoge Cook's afstand hebben een grote invloed op de regressielijn en kunnen de resultaten vertekenen.")
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    cooks_distance = model_sm.get_influence().cooks_distance[0]
    cooks_df = pd.DataFrame({'Index': range(len(cooks_distance)), 'Cooks_Distance': cooks_distance})
    cooks_chart = alt.Chart(cooks_df).mark_circle(size=60).encode(
        x=alt.X('Index', title='Index van Datapunt'),
        y=alt.Y('Cooks_Distance', title='Cooks Afstand', scale=alt.Scale(domain=[0, 0.4])),
        tooltip=['Cooks_Distance']
    ).properties(title='Cooks Afstand per Datapunt').interactive()
    cooks_threshold = 0.05
    rule_cooks = alt.Chart(pd.DataFrame({'y': [cooks_threshold]})).mark_rule(color='red').encode(y='y')
    st.altair_chart(cooks_chart + rule_cooks, use_container_width=True)
    st.write("#### Onderzoek van Invloedrijke Punten")
    st.write("Hieronder zie je de data voor de twee datapunten met de grootste Cook's distance. Dit kan je helpen te bepalen of er datakwaliteitsproblemen zijn of dat het om uitzonderlijke, maar correcte, observaties gaat.")
    cooks_df['Original_Index'] = y.index.values
    influential_indices = cooks_df.sort_values(by='Cooks_Distance', ascending=False).head(4)['Original_Index'].tolist()
    influential_points_data = df_reg.loc[influential_indices, X_vars + ['ervarengezondheid', 'Gemeente code (with prefix)', 'Provincie']]
    cols = ['Gemeente code (with prefix)', 'Provincie'] + X_vars + ['ervarengezondheid']
    influential_points_data = influential_points_data[cols]
    st.dataframe(influential_points_data)
    # Bereken gemiddelde van alle punten voor de gekozen variabelen
    average_values = df_reg[X_vars].mean().round(1).to_frame().T
    average_values.index = ['Gemiddelde']

    # Bereik (min tot max) van de originele waarden
    range_values = df_reg[X_vars].agg(lambda x: f"{x.min():.1f} - {x.max():.1f}").to_frame().T
    range_values.index = ['Bereik']

    comparison_df = pd.concat([influential_points_data[X_vars], range_values, average_values])

    st.write("#### Vergelijking met het Gemiddelde")
    st.write("Om te zien of de invloedrijke datapunten uitschieters zijn, vergelijken we hun waarden met het gemiddelde van de dataset.")
    st.dataframe(comparison_df)


def render_regression_section(dataframe):
    st.write("#### Meervoudige Lineaire Regressie")
    st.write("Dit model voorspelt 'ervarengezondheid' op basis van financiële en leefstijl variabelen.")
    X_vars = [
        # Financiële situatie

        # Mentale gezondheid / stress
        'AngstDepressiegevoelensAfg4Weken_13',
        'TevredenMetEigenLeven_11',
        'VaakGelukkigAfg4Weken_15',
        'SlaaptMeestalSlecht_7',
        'VoldoendeWeerbaar_16',

        # Leefstijl
        'Overgewicht_59',
        'ZwareDrinker_72'
    ]

    df_reg = dataframe.dropna(subset=X_vars + ['ervarengezondheid'])
    X_full = df_reg[X_vars].copy()
    y = df_reg['ervarengezondheid']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_full), columns=X_full.columns, index=X_full.index)
    corr_with_target = abs(X_scaled.corrwith(y))
    min_corr_threshold = 0.0
    selected_features = corr_with_target[corr_with_target >= min_corr_threshold].index.tolist()
    X_selected = X_scaled[selected_features]
    if X_selected.empty or X_selected.shape[1] <= 1:
        st.warning("Er zijn onvoldoende variabelen geselecteerd voor regressie-analyse. Controleer de invoerdata.")
        return
    y_orig = y.copy()
    y = render_transformation_options(y, y_orig)
    df_reg = df_reg.loc[X_selected.index]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model_sk = LinearRegression()
    model_sk.fit(X_train, y_train)
    y_pred = model_sk.predict(X_test)
    full_model = LinearRegression()
    full_model.fit(X_selected, y)
    full_pred = full_model.predict(X_selected)
    residuals = y - full_pred
    r2 = r2_score(y_test, y_pred)
    n = y_test.shape[0]
    p = X_test.shape[1]
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    st.write(f"R-kwadraat (R²) score: {r2:.3f}")
    st.write(f"Adjusted R-kwadraat (R² adj) score: {r2_adj:.3f}")
    st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.3f}")
    st.write("#### Model Performance Samenvatting")
    model_sm = sm.OLS(y, sm.add_constant(X_selected)).fit()
    summary_data = {
        'R²': [model_sm.rsquared],
        'Adjusted R²': [model_sm.rsquared_adj],
        'MSE': [mean_squared_error(y, model_sm.fittedvalues)],
        'F-statistiek': [model_sm.fvalue],
        'F p-waarde': np.array([model_sm.f_pvalue]).round(3)

    }
    st.dataframe(pd.DataFrame(summary_data).T.rename(columns={0: 'Waarde'}), use_container_width=True)
    render_vif_table(X_selected)
    final_model = LinearRegression()
    final_model.fit(X_selected, y)
    render_feature_importance(final_model, X_selected, y)
    render_residual_plots(y_test, y_pred, model_sk, X_selected, y, df_reg, X_vars, model_sm)


def main():
    df_loaded, *_ = load_data.load_data()
    dataframe = prepare_dataframe(df_loaded)
    st.subheader('Statistische Analyse: Correlatie en Regressie')
    with st.container(border=True):
        render_correlation_section(dataframe)
        render_regression_section(dataframe)
main()
