import streamlit as st

def apply_custom_css():
    """
    Fungsi untuk menyuntikkan CSS kustom agar tampilan Streamlit
    sedikit bergaya tegas (Neobrutalism lite).
    """
    custom_css = """
    <style>
    /* Styling tombol utama */
    div.stButton > button:first-child {
        background-color: #facc15;
        color: black;
        font-weight: bold;
        border: 3px solid black;
        border-radius: 0px;
        box-shadow: 4px 4px 0px 0px #000000;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0px 0px #000000;
        background-color: #eab308;
        color: black;
    }
    
    /* Metrik Card */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 3px solid black;
        padding: 1rem;
        box-shadow: 4px 4px 0px 0px #000000;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
