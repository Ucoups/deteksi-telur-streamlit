import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_styling import apply_custom_css
from utils.database import get_dashboard_stats

# Cek Login
if not st.session_state.get('logged_in', False):
    st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
    apply_custom_css()
    st.title("📊 Dashboard Statistik")
    st.warning("🔒 Silakan login terlebih dahulu di halaman Beranda!")
    st.stop()

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
apply_custom_css()


st.title("📊 Dashboard Statistik")
st.write("Ringkasan kualitas produksi telur harian yang tercatat di sistem.")

# Ambil data statistik dari database
stats = get_dashboard_stats()

# Tampilkan metrik utama dengan delta
col1, col2, col3 = st.columns(3)
col1.metric("Total Telur Diuji", f"{stats['total_uji']:,}", delta=f"{stats.get('delta_uji', 0):+} hari ini")
col2.metric("Total Segar", f"{stats['total_segar']:,}", delta=f"{stats.get('delta_segar', 0):+} hari ini")
col3.metric("Total Busuk", f"{stats['total_busuk']:,}", delta=f"{stats.get('delta_busuk', 0):+} hari ini", delta_color="inverse")

st.divider()

col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
    st.subheader("📈 Tren Produksi Harian")
    if not stats['grafik']:
        st.info("Belum ada data deteksi yang cukup untuk menampilkan grafik.")
    else:
        # Ubah dict stats['grafik'] menjadi DataFrame
        chart_data = pd.DataFrame.from_dict(stats['grafik'], orient='index').reset_index()
        chart_data.rename(columns={'index': 'Tanggal'}, inplace=True)
        
        # Plotly Bar Chart Interaktif
        fig_bar = px.bar(chart_data, x='Tanggal', y=['Segar', 'Busuk'], 
                         barmode='group',
                         color_discrete_map={"Segar": "#28a745", "Busuk": "#dc3545"},
                         labels={'value': 'Jumlah Telur', 'variable': 'Kategori'})
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

with col_chart2:
    st.subheader("🍩 Rasio Kualitas")
    if stats['total_uji'] == 0:
        st.info("Belum ada data uji.")
    else:
        fig_pie = px.pie(
            names=['Segar', 'Busuk'], 
            values=[stats['total_segar'], stats['total_busuk']],
            color=['Segar', 'Busuk'],
            color_discrete_map={"Segar": "#28a745", "Busuk": "#dc3545"},
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
