# ==============================================================================
# 📦 1) IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ 2) KONFIGURASI DASHBOARD & GOOGLE DRIVE
# ==============================================================================
st.set_page_config(page_title="📊 Dashboard Analisis Saham IDX", layout="wide", page_icon="📈")

FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP"
FILE_NAME = "Kompilasi_Data_1Tahun.csv"

# Bobot skor
W = dict(
    trend_akum=0.40, trend_ff=0.30, trend_mfv=0.20, trend_mom=0.10,
    mom_price=0.40, mom_vol=0.25, mom_akum=0.25, mom_ff=0.10,
    blend_trend=0.35, blend_mom=0.35, blend_nbsa=0.20, blend_fcontrib=0.05, blend_unusual=0.05
)

# ==============================================================================
# 🧰 3) HELPER FUNCTIONS
# ==============================================================================
def safe_get(df, col, default=None):
    """Ambil kolom dari df, jika tidak ada kembalikan default Series"""
    return df[col] if col in df.columns else pd.Series(default, index=df.index)

def ensure_cols(df, cols):
    """Pastikan df hanya ambil kolom yang tersedia"""
    return [c for c in cols if c in df.columns]

# ==============================================================================
# 📦 4) LOAD DATA DARI GOOGLE DRIVE
# ==============================================================================
def get_gdrive_service():
    """Autentikasi ke Google Drive"""
    try:
        creds_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            creds_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        st.error(f"❌ Gagal autentikasi GDrive: {e}")
        return None

@st.cache_data(ttl=3600)
def load_data():
    """Download CSV dari GDrive & load ke Pandas"""
    service = get_gdrive_service()
    if service is None:
        return pd.DataFrame()

    try:
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(q=query, fields="files(id,name)", orderBy="modifiedTime desc", pageSize=1).execute()
        items = results.get("files", [])
        if not items:
            st.error(f"❌ File '{FILE_NAME}' tidak ditemukan di folder GDrive.")
            return pd.DataFrame()

        file_id = items[0]["id"]
        st.success(f"File ditemukan (ID: {file_id}). Mengunduh data...")

        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)

        df = pd.read_csv(fh)
        df.columns = df.columns.str.strip()

        # Konversi tanggal & numeric
        if "Last Trading Date" in df.columns:
            df["Last Trading Date"] = pd.to_datetime(df["Last Trading Date"], errors="coerce")

        cols_to_num = [
            "Change %", "Typical Price", "TPxV", "VWMA_20D", "MA20_vol", "MA5_vol",
            "Volume Spike (x)", "Net Foreign Flow", "Foreign Buy", "Foreign Sell",
            "Bid/Offer Imbalance", "Money Flow Value", "Close", "Volume", "Value",
            "High", "Low", "Change", "Previous"
        ]
        for c in cols_to_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Bersihkan Unusual Volume
        if "Unusual Volume" not in df.columns:
            df["Unusual Volume"] = False
        else:
            df["Unusual Volume"] = df["Unusual Volume"].astype(str).str.strip().str.lower().isin(
                ["spike volume signifikan", "true", "yes", "1"]
            )

        if "Final Signal" in df.columns:
            df["Final Signal"] = df["Final Signal"].astype(str).str.strip()

        for col in ["Volume Spike (x)", "Value", "Net Foreign Flow"]:
            if col not in df.columns:
                df[col] = 0

        # Drop baris tanpa Stock Code (tanpa argumen errors)
        if "Stock Code" in df.columns:
            df = df.dropna(subset=["Stock Code"])

        st.success("✅ Data berhasil dimuat dan dibersihkan.")
        return df

    except Exception as e:
        st.error(f"❌ Terjadi error saat memuat data: {e}")
        return pd.DataFrame()

# ==============================================================================
# 🧮 5) HITUNG SKOR POTENSIAL
# ==============================================================================
def pct_rank(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True).fillna(0) * 100

def to_pct(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1:
        return pd.Series(50, index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(50, index=s.index)
    return (s - mn) / (mx - mn) * 100

@st.cache_data(ttl=3600)
def calculate_potential_score(df, latest_date):
    if df.empty or "Last Trading Date" not in df.columns:
        st.warning("⚠️ Data tidak tersedia untuk hitung skor.")
        return pd.DataFrame()

    trend_df = df[df["Last Trading Date"] >= (latest_date - pd.Timedelta(days=30))].copy()
    mom_df = df[df["Last Trading Date"] >= (latest_date - pd.Timedelta(days=7))].copy()
    last_df = df[df["Last Trading Date"] == latest_date].copy()
    if trend_df.empty or mom_df.empty or last_df.empty:
        st.warning("⚠️ Data kurang untuk hitung skor (minimal 30 hari).")
        return pd.DataFrame()

    tr = trend_df.groupby("Stock Code").agg(
        last_price=("Close", "last"),
        last_final_signal=("Final Signal", "last"),
        total_net_ff=("Net Foreign Flow", "sum"),
        total_money_flow=("Money Flow Value", "sum"),
        avg_change_pct=("Change %", "mean"),
        sector=("Sector", "last")
    ).reset_index()

    score_akum = tr["last_final_signal"].map({
        "Strong Akumulasi": 100, "Akumulasi": 75, "Netral": 30,
        "Distribusi": 10, "Strong Distribusi": 0
    }).fillna(30)
    score_ff = pct_rank(tr["total_net_ff"])
    score_mfv = pct_rank(tr["total_money_flow"])
    score_mom = pct_rank(tr["avg_change_pct"])
    tr["Trend Score"] = (score_akum * W["trend_akum"] + score_ff * W["trend_ff"] +
                         score_mfv * W["trend_mfv"] + score_mom * W["trend_mom"])

    mo = mom_df.groupby("Stock Code").agg(
        total_change_pct=("Change %", "sum"),
        had_unusual_volume=("Unusual Volume", "any"),
        last_final_signal=("Final Signal", "last"),
        total_net_ff=("Net Foreign Flow", "sum")
    ).reset_index()

    s_price = pct_rank(mo["total_change_pct"])
    s_vol = mo["had_unusual_volume"].map({True: 100, False: 20}).fillna(20)
    s_akum = mo["last_final_signal"].map({
        "Strong Akumulasi": 100, "Akumulasi": 80, "Netral": 40,
        "Distribusi": 10, "Strong Distribusi": 0
    }).fillna(40)
    s_ff7 = pct_rank(mo["total_net_ff"])
    mo["Momentum Score"] = (s_price * W["mom_price"] + s_vol * W["mom_vol"] +
                            s_akum * W["mom_akum"] + s_ff7 * W["mom_ff"])

    nbsa = trend_df.groupby("Stock Code").agg(total_net_ff_30d=("Net Foreign Flow", "sum")).reset_index()
    tmp = trend_df.copy()
    if {"Foreign Buy", "Foreign Sell", "Value"}.issubset(df.columns):
        tmp["Foreign Value"] = tmp["Foreign Buy"].fillna(0) + tmp["Foreign Sell"].fillna(0)
        contrib = tmp.groupby("Stock Code").agg(
            total_foreign_value_30d=("Foreign Value", "sum"),
            total_value_30d=("Value", "sum")
        ).reset_index()
        contrib["foreign_contrib_pct"] = (contrib["total_foreign_value_30d"] / (contrib["total_value_30d"] + 1)) * 100
    else:
        contrib = pd.DataFrame({"Stock Code": [], "foreign_contrib_pct": []})

    uv = last_df.set_index("Stock Code")["Unusual Volume"].map({True: 1, False: 0})
    rank = tr.merge(mo[["Stock Code", "Momentum Score"]], on="Stock Code", how="outer")
    rank = rank.merge(nbsa, on="Stock Code", how="left")
    rank = rank.merge(contrib[["Stock Code", "foreign_contrib_pct"]], on="Stock Code", how="left")

    rank["NBSA Score"] = to_pct(rank["total_net_ff_30d"])
    rank["Foreign Contrib Score"] = to_pct(rank["foreign_contrib_pct"])
    rank["Potential Score"] = (
        rank["Trend Score"].fillna(0) * W["blend_trend"] +
        rank["Momentum Score"].fillna(0) * W["blend_mom"] +
        rank["NBSA Score"].fillna(50) * W["blend_nbsa"] +
        rank["Foreign Contrib Score"].fillna(50) * W["blend_fcontrib"] +
        uv.reindex(rank["Stock Code"]).fillna(0).values * W["blend_unusual"]
    )

    top20 = rank.sort_values("Potential Score", ascending=False).head(20).copy()
    top20.insert(0, "Analysis Date", latest_date.strftime("%Y-%m-%d"))
    for c in ["Potential Score", "Trend Score", "Momentum Score", "NBSA Score", "Foreign Contrib Score"]:
        if c in top20.columns:
            top20[c] = pd.to_numeric(top20[c], errors="coerce").round(2)
    st.success("✅ Skor potensial berhasil dihitung.")
    return top20

# ==============================================================================
# 💎 6) DASHBOARD UTAMA
# ==============================================================================
st.title("📈 Dashboard Analisis Saham IDX")
st.caption("Menganalisis data historis harian untuk menemukan saham potensial.")

df = load_data()
if df.empty:
    st.stop()

st.sidebar.header("🎛️ Filter")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

max_date = df["Last Trading Date"].max().date()
selected_date = st.sidebar.date_input("Pilih Tanggal Analisis", max_date)
df_day = df[df["Last Trading Date"].dt.date == selected_date].copy()
st.caption(f"Data per: **{selected_date.strftime('%d %B %Y')}**")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Individual", "📋 Data", "🏆 Top 20"])

# === TAB 1 ===
with tab1:
    if df_day.empty:
        st.warning("Tidak ada data untuk tanggal ini.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Saham", f"{len(df_day['Stock Code'].unique()):,.0f}")
        col2.metric("Unusual Volume", f"{int(df_day['Unusual Volume'].sum()):,.0f}")
        col3.metric("Total Value", f"Rp {df_day['Value'].sum():,.0f}")

# === TAB 4 ===
with tab4:
    df_top20 = calculate_potential_score(df, pd.Timestamp(max_date))
    if not df_top20.empty:
        st.dataframe(df_top20, use_container_width=True, hide_index=True)
