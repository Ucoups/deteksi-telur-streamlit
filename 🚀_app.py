import streamlit as st
import time
from utils.ui_styling import apply_custom_css
from utils.config import USERS

# Setup Halaman
st.set_page_config(
    page_title="EggQuality AI - Beranda",
    page_icon="🥚",
    layout="wide"
)

# Terapkan styling kustom (mirip Neobrutalism)
apply_custom_css()

# Inisialisasi Session State Login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None

st.title("🥚 EggQuality AI System")
st.markdown("### Sistem Deteksi Kualitas Telur Cerdas berbasis Streamlit")

if not st.session_state['logged_in']:
    st.subheader("🔒 Silakan Login Terlebih Dahulu")
    st.write("Masukkan kredensial Anda untuk mengakses sistem deteksi kualitas telur.")
    
    # Form Login bergaya Neobrutalism
    with st.container():
        st.markdown(
            """
            <style>
            div[data-testid="stForm"] {
                border: 3px solid black !important;
                border-radius: 0px !important;
                box-shadow: 6px 6px 0px 0px #000000 !important;
                background-color: white !important;
                padding: 2rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        with st.form("login_form"):
            username_input = st.text_input("Username:", placeholder="Masukkan username")
            password_input = st.text_input("Password:", type="password", placeholder="Masukkan password")
            
            submit_button = st.form_submit_button("MASUK SISTEM 🚀", use_container_width=True)
            
            if submit_button:
                if username_input in USERS and USERS[username_input]["password"] == password_input:
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = USERS[username_input]["role"]
                    st.session_state['name'] = USERS[username_input]["name"]
                    st.session_state['username'] = username_input
                    st.success(f"Selamat datang, {USERS[username_input]['name']}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Kombinasi Username dan Password salah!")
else:
    # Tampilan Selamat Datang jika sudah login
    st.success(f"🔓 Anda masuk sebagai **{st.session_state['name']}** ({st.session_state['role'].upper()})")
    
    # Kotak Informasi Hak Akses
    with st.container():
        st.markdown(
            f"""
            <div style="border: 3px solid black; padding: 20px; background-color: #facc15; box-shadow: 4px 4px 0px 0px black; color: black; font-weight: bold; margin-bottom: 20px;">
              <h4>Halo, {st.session_state['name']}!</h4>
              <p>Hak Akses Anda: {st.session_state['role'].upper()}</p>
              {"<ul><li>Akses penuh ke semua modul deteksi & pelatihan model</li><li>Dapat melakukan input data training baru di halaman Deteksi Live</li></ul>" if st.session_state['role'] == 'admin' else "<ul><li>Dapat melakukan deteksi kualitas telur secara live</li><li>Tidak memiliki akses untuk melatih ulang model atau memanipulasi dataset</li></ul>"}
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.write("Silakan gunakan menu navigasi di sebelah kiri untuk mengakses modul sistem.")
    st.info("Pilih **📷 Deteksi Live** untuk mulai mengecek telur segar/busuk.")
    
    if st.button("KELUAR SISTEM (LOGOUT) 🚪", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state['role'] = None
        st.session_state['name'] = None
        st.session_state['username'] = None
        st.rerun()

