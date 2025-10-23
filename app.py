import streamlit as st
import pandas as pd
import gdown

# =====================================================================
# ⚙️ KONFIGURASI DASHBOARD
# =====================================================================
st.set_page_config(
    page_title="📊 Stock Volume Spike Dashboard",
    layout="wide",
    page_icon="📈"
)

st.title("📈 Stock Volume Spike Dashboard")
st.caption("Data Source: Google Drive → Folder 'Data Storage' (Public Access)")

# =====================================================================
# 📦 AMBIL DATA DARI GOOGLE DRIVE
# =====================================================================
# ID file CSV publik (dari link Google Drive)
FILE_ID = "1A3eqXBUhzOTOQ1QR72ArEbLhGCTtYQ3L"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_data(ttl=3600)
def load_data():
    try:
        gdown.download(URL, "data.csv", quiet=True)
        df = pd.read_csv("data.csv")
        df.columns = df.columns.str.strip()  # hapus spasi di header
        return df
    except Exception as e:
        st.error(f"❌ Gagal membaca data dari Google Drive: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("⚠️ Data belum berhasil dimuat. Periksa kembali link Google Drive atau izin akses file.")
    st.stop()

# =====================================================================
# 🧭 FILTER DATA
# =====================================================================
st.sidebar.header("🎛️ Filter Data")

sector = st.sidebar.multiselect(
    "Pilih Sektor",
    sorted(df["Sector"].dropna().unique()),
    placeholder="Pilih sektor"
)

signal = st.sidebar.multiselect(
    "Filter Berdasarkan Final Signal",
    sorted(df["Final Signal"].dropna().unique()),
    placeholder="Pilih signal"
)

filtered_df = df.copy()
if sector:
    filtered_df = filtered_df[filtered_df["Sector"].isin(sector)]
if signal:
    filtered_df = filtered_df[filtered_df["Final Signal"].isin(signal)]

st.markdown(f"**Menampilkan {len(filtered_df)} baris data**")

# =====================================================================
# 📋 TABEL DATA
# =====================================================================
st.dataframe(
    filtered_df[[
        "Stock Code", "Company Name", "Sector",
        "Last Trading Date", "Close", " Change % ",
        "Volume", " Volume Spike (x) ",
        "Unusual Volume", "Final Signal"
    ]].head(100),
    use_container_width=True
)

# =====================================================================
# 📈 STATISTIK RINGKAS
# =====================================================================
col1, col2, col3 = st.columns(3)
col1.metric("Total Saham", len(df["Stock Code"].unique()))
col2.metric("Spike Signifikan", df["Unusual Volume"].eq("Spike Volume Signifikan").sum())
col3.metric("Total Sektor", len(df["Sector"].dropna().unique()))

# =====================================================================
# 📊 DISTRIBUSI SIGNAL
# =====================================================================
st.subheader("📊 Distribusi Final Signal")
signal_counts = df["Final Signal"].value_counts()
st.bar_chart(signal_counts)

# =====================================================================
# 📊 DISTRIBUSI SEKTOR (HANYA SAHAM SPIKE)
# =====================================================================
st.subheader("🏭 Distribusi Sektor dengan Volume Spike Signifikan")
spike_df = df[df["Unusual Volume"] == "Spike Volume Signifikan"]
sector_counts = spike_df["Sector"].value_counts()
st.bar_chart(sector_counts)

# =====================================================================
# 🧾 INFO DATA
# =====================================================================
st.markdown("---")
st.caption("Data diperbarui otomatis setiap kali file CSV di folder Google Drive diperbarui.")
