import streamlit as st
from stock_dashboard import render_stock_dashboard
from luchuuthanh_chart import render_luchuuthanh_chart

st.set_page_config(
    page_title="StockLab Việt Nam",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1600px;}
[data-testid="stMetric"] {background: #101a2a; padding: 12px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

main_tab, luchuuthanh_tab = st.tabs(["📈 STOCKLAB – Bộ lọc và dữ liệu", "📊 LUCHUUTHANH – Biểu đồ"])
with main_tab:
    render_stock_dashboard()
with luchuuthanh_tab:
    render_luchuuthanh_chart()
