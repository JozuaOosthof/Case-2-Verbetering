import streamlit as st

def render_conclusion_page():
    st.set_page_config("Conclusies: Gezondheid Jongvolwassenen", layout="wide", page_icon="🔒")
    
    st.title("Conclusie")

    with st.container(border=True):
        st.write("""
        Deze analyse onderzoekt hoe de financiële situatie, mentale gezondheid en leefstijl van jongvolwassenen samenhangen met hun ervaren gezondheid.
        """)

        st.header("Belangrijkste bevindingen")
        st.write("""
        - Jongvolwassenen die vaker gelukkig zijn en tevreden met hun leven, ervaren een **betere gezondheid**.
        - Angst, depressie en slechte slaap hebben een **negatief effect** op de ervaren gezondheid.
        - Overgewicht en zwaar alcoholgebruik beïnvloeden de gezondheid ook negatief, maar in mindere mate.
        - Weerbaarheid heeft een licht **positief effect**.
        """)

        st.header("Modelkwaliteit")
        st.write("""
        - Het model verklaart een groot deel van de variatie in ervaren gezondheid.
        - Residuen zijn normaal verdeeld en er is geen aanwijzing voor heteroscedasticiteit.
        - De resultaten zijn statistisch betrouwbaar.
        """)

        st.header("Conclusie")
        st.write("""
        Zowel mentale gezondheid als leefstijlfactoren zijn belangrijk voor het ervaren welzijn van jongvolwassenen.  
        Interventies die welzijn, slaap en gezonde leefstijl bevorderen, kunnen de ervaren gezondheid positief beïnvloeden.
        """)

# Main functie
if __name__ == "__main__":
    render_conclusion_page()
