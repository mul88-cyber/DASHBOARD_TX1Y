import streamlit as st
import pandas as pd
import gdown

# --- CONFIGURASI DASHBOARD ---
st.set_page_config(
    page_title="Stock Volume Spike Dashboard",
    layout="wide",
    page_icon="📈"
)

st.title("📊 Stock Volume Spike Dashboard")
st.caption("Data Source: Google Drive - Folder 'Data Storage' (Public Access)")

# --- LINK CSV GOOGLE DRIVE ---
# Ganti ID di bawah ini dengan ID file CSV kompilasi terbaru kamu
# (contoh ID: 1AbCDeFGhijklMNOPqrSTuvwXYz)
file_id = "PASTE_ID_CSV_DISINI"
url = f"https://drive.google.com/uc?id={file_id}"

# --- LOAD DATA ---
@st.cache_data(ttl=3600)
def load_data():
    try:
        gdown.download(url, "data.csv", quiet=True)
        df = pd.read_csv("data.csv")
        df.columns = df.columns.str.strip()  # bersihkan nama kolom
        return df
    except Exception as e:
        st.error(f"Gagal membaca data dari Google Drive: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Data belum berhasil dimuat. Periksa kembali ID file atau izin Google Drive.")
    st.stop()

# --- FILTER ---
st.sidebar.header("Filter Data")
sector = st.sidebar.multiselect("Pilih Sektor", sorted(df["Sector"].dropna().unique()))
signal = st.sidebar.multiselect("Filter Final Signal", sorted(df["Final Signal"].dropna().unique()))

filtered = df.copy()
if sector:
    filtered = filtered[filtered["Sector"].isin(sector)]
if signal:
    filtered = filtered[filtered["Final Signal"].isin(signal)]

st.write(f"📈 Menampilkan {len(filtered)} baris data")

# --- TABEL ---
st.dataframe(
    filtered[[
        "Stock Code", "Company Name", "Sector", "Last Trading Date",
        "Close", "Change %", "Volume", "Volume Spike (x)",
        "Unusual Volume", "Final Signal"
    ]].head(100),
    use_container_width=True
)

# --- STATISTIK ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Saham", len(df["Stock Code"].unique()))
col2.metric("Spike Signifikan", (df["Unusual Volume"] == "Spike Volume Signifikan").sum())
col3.metric("Total Sektor", len(df["Sector"].dropna().unique()))

# --- CHART: Distribusi Spike ---
st.subheader("📊 Distribusi Volume Spike")
spike_count = df["Unusual Volume"].value_counts()
st.bar_chart(spike_count)
