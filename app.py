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

# Import library Google
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ 2) KONFIGURASI DASHBOARD & G-DRIVE
# ==============================================================================
st.set_page_config(
    page_title="📊 Dashboard Analisis Saham IDX",
    layout="wide",
    page_icon="📈"
)

FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP"
FILE_NAME = "Kompilasi_Data_1Tahun.csv"

W = dict(
    trend_akum=0.40, trend_ff=0.30, trend_mfv=0.20, trend_mom=0.10,
    mom_price=0.40,  mom_vol=0.25,  mom_akum=0.25,  mom_ff=0.10,
    blend_trend=0.35, blend_mom=0.35, blend_nbsa=0.20, blend_fcontrib=0.05, blend_unusual=0.05
)

# ==============================================================================
# 🧰 3) HELPER FUNGSI
# ==============================================================================
def safe_get(df, col, default=None):
    """Ambil kolom dari df, kalau tidak ada kembalikan Series default"""
    return df[col] if col in df.columns else pd.Series(default, index=df.index)

def ensure_cols(df, cols):
    """Pastikan df hanya ambil kolom yang tersedia"""
    return [c for c in cols if c in df.columns]

# ==============================================================================
# 📦 4) FUNGSI MEMUAT DATA DARI GOOGLE DRIVE
# ==============================================================================
def get_gdrive_service():
    """Otentikasi service account dari secrets.toml"""
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        st.error(f"❌ Gagal otentikasi Google Drive: {e}")
        return None

@st.cache_data(ttl=3600)
def load_data():
    """Cari file CSV di folder GDrive, download, dan baca ke DataFrame"""
    service = get_gdrive_service()
    if service is None:
        return pd.DataFrame()

    try:
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(
            q=query, fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1
        ).execute()
        items = results.get('files', [])
        if not items:
            st.error(f"❌ File '{FILE_NAME}' tidak ditemukan di folder GDrive.")
            return pd.DataFrame()

        file_id = items[0]['id']
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

        if 'Last Trading Date' in df.columns:
            df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')

        cols_to_numeric = [
            'Change %', 'Typical Price', 'TPxV', 'VWMA_20D', 'MA20_vol', 'MA5_vol',
            'Volume Spike (x)', 'Net Foreign Flow', 'Foreign Buy', 'Foreign Sell',
            'Bid/Offer Imbalance', 'Money Flow Value', 'Close', 'Volume', 'Value',
            'High', 'Low', 'Change', 'Previous'
        ]
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Unusual Volume' not in df.columns:
            df['Unusual Volume'] = False
        else:
            df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(
                ['spike volume signifikan', 'true', 'yes', '1']
            )

        if 'Final Signal' in df.columns:
            df['Final Signal'] = df['Final Signal'].astype(str).str.strip()

        for col in ["Volume Spike (x)", "Value", "Net Foreign Flow"]:
            if col not in df.columns:
                df[col] = 0

        df = df.dropna(subset=['Stock Code'], errors='ignore')
        st.success("✅ Data berhasil dimuat dan dibersihkan.")
        return df

    except Exception as e:
        st.error(f"❌ Terjadi error saat memuat data: {e}")
        return pd.DataFrame()

# ==============================================================================
# 🧮 5) FUNGSI SKOR POTENSIAL
# ==============================================================================
def pct_rank(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, method="average").fillna(0) * 100

def to_pct(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1: return pd.Series(50, index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx: return pd.Series(50, index=s.index)
    return (s - mn) / (mx - mn) * 100

@st.cache_data(ttl=3600)
def calculate_potential_score(df, latest_date):
    if df.empty or 'Last Trading Date' not in df.columns:
        st.warning("⚠️ Data tidak tersedia untuk perhitungan skor.")
        return pd.DataFrame()

    trend_start = latest_date - pd.Timedelta(days=30)
    mom_start = latest_date - pd.Timedelta(days=7)
    trend_df = df[df['Last Trading Date'] >= trend_start].copy()
    mom_df = df[df['Last Trading Date'] >= mom_start].copy()
    last_df = df[df['Last Trading Date'] == latest_date].copy()

    if trend_df.empty or mom_df.empty or last_df.empty:
        st.warning("⚠️ Data kurang untuk hitung skor (minimal 30 hari).")
        return pd.DataFrame()

    tr = trend_df.groupby('Stock Code').agg(
        last_price=('Close', 'last'),
        last_final_signal=('Final Signal', 'last'),
        total_net_ff=('Net Foreign Flow', 'sum'),
        total_money_flow=('Money Flow Value', 'sum'),
        avg_change_pct=('Change %', 'mean'),
        sector=('Sector', 'last')
    ).reset_index()

    score_akum = tr['last_final_signal'].map({
        'Strong Akumulasi': 100, 'Akumulasi': 75, 'Netral': 30,
        'Distribusi': 10, 'Strong Distribusi': 0
    }).fillna(30)
    score_ff = pct_rank(tr['total_net_ff'])
    score_mfv = pct_rank(tr['total_money_flow'])
    score_mom = pct_rank(tr['avg_change_pct'])
    tr['Trend Score'] = (score_akum * W['trend_akum'] + score_ff * W['trend_ff'] +
                         score_mfv * W['trend_mfv'] + score_mom * W['trend_mom'])

    mo = mom_df.groupby('Stock Code').agg(
        total_change_pct=('Change %', 'sum'),
        had_unusual_volume=('Unusual Volume', 'any'),
        last_final_signal=('Final Signal', 'last'),
        total_net_ff=('Net Foreign Flow', 'sum')
    ).reset_index()

    s_price = pct_rank(mo['total_change_pct'])
    s_vol = mo['had_unusual_volume'].map({True: 100, False: 20}).fillna(20)
    s_akum = mo['last_final_signal'].map({
        'Strong Akumulasi': 100, 'Akumulasi': 80, 'Netral': 40,
        'Distribusi': 10, 'Strong Distribusi': 0
    }).fillna(40)
    s_ff7 = pct_rank(mo['total_net_ff'])
    mo['Momentum Score'] = (s_price * W['mom_price'] + s_vol * W['mom_vol'] +
                            s_akum * W['mom_akum'] + s_ff7 * W['mom_ff'])

    nbsa = trend_df.groupby('Stock Code').agg(total_net_ff_30d=('Net Foreign Flow', 'sum')).reset_index()
    tmp = trend_df.copy()
    if {'Foreign Buy', 'Foreign Sell', 'Value'}.issubset(df.columns):
        tmp['Foreign Value'] = tmp['Foreign Buy'].fillna(0) + tmp['Foreign Sell'].fillna(0)
        contrib = tmp.groupby('Stock Code').agg(
            total_foreign_value_30d=('Foreign Value', 'sum'),
            total_value_30d=('Value', 'sum')
        ).reset_index()
        contrib['foreign_contrib_pct'] = (contrib['total_foreign_value_30d'] / (contrib['total_value_30d'] + 1)) * 100
    else:
        contrib = pd.DataFrame({'Stock Code': [], 'foreign_contrib_pct': []})

    uv = last_df.set_index('Stock Code')['Unusual Volume'].map({True: 1, False: 0})
    rank = tr.merge(mo[['Stock Code', 'Momentum Score']], on='Stock Code', how='outer')
    rank = rank.merge(nbsa, on='Stock Code', how='left')
    rank = rank.merge(contrib[['Stock Code', 'foreign_contrib_pct']], on='Stock Code', how='left')

    rank['NBSA Score'] = to_pct(rank['total_net_ff_30d'])
    rank['Foreign Contrib Score'] = to_pct(rank['foreign_contrib_pct'])
    unusual_bonus = uv.reindex(rank['Stock Code']).fillna(0) * 5

    rank['Potential Score'] = (
        rank['Trend Score'].fillna(0) * W['blend_trend'] +
        rank['Momentum Score'].fillna(0) * W['blend_mom'] +
        rank['NBSA Score'].fillna(50) * W['blend_nbsa'] +
        rank['Foreign Contrib Score'].fillna(50) * W['blend_fcontrib'] +
        unusual_bonus.values * W['blend_unusual']
    )

    top20 = rank.sort_values('Potential Score', ascending=False).head(20).copy()
    top20.insert(0, 'Analysis Date', latest_date.strftime('%Y-%m-%d'))
    for c in ['Potential Score', 'Trend Score', 'Momentum Score', 'NBSA Score', 'Foreign Contrib Score']:
        if c in top20.columns: top20[c] = pd.to_numeric(top20[c], errors='coerce').round(2)
    st.success("✅ Skor potensial berhasil dihitung.")
    return top20

# ==============================================================================
# 💎 6) DASHBOARD LAYOUT
# ==============================================================================
st.title("📈 Dashboard Analisis Saham IDX")
st.caption("Menganalisis data historis harian untuk menemukan saham potensial.")

df = load_data()
if df.empty:
    st.stop()

st.sidebar.header("🎛️ Filter Analisis Harian")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

max_date = df['Last Trading Date'].max().date() if 'Last Trading Date' in df.columns else pd.Timestamp.now().date()
selected_date = st.sidebar.date_input(
    "Pilih Tanggal Analisis",
    max_date,
    min_value=df['Last Trading Date'].min().date() if 'Last Trading Date' in df.columns else max_date,
    max_value=max_date,
    format="DD-MM-YYYY"
)

df_day = df[df['Last Trading Date'].dt.date == selected_date].copy()
st.caption(f"Menampilkan data untuk tanggal: **{selected_date.strftime('%d %B %Y')}**")

# Filter Sidebar lanjutan
st.sidebar.header("Filter Data Lanjutan (Tab 3)")
selected_stocks = st.sidebar.multiselect("Pilih Saham", sorted(df_day["Stock Code"].dropna().unique()))
selected_sectors = st.sidebar.multiselect("Pilih Sektor", sorted(df_day["Sector"].dropna().unique()))
selected_signals = st.sidebar.multiselect("Pilih Final Signal", sorted(df_day["Final Signal"].dropna().unique()))

max_spike_val = 50.0
if "Volume Spike (x)" in df_day.columns and df_day["Volume Spike (x)"].notna().any():
    max_spike_val = float(df_day["Volume Spike (x)"].max())

min_spike = st.sidebar.slider("Minimal Volume Spike (x)", 1.0, max_spike_val, min(2.0, max_spike_val), 0.5)
show_only_spike = st.sidebar.checkbox("Hanya tampilkan Unusual Volume", value=False)

df_filtered = df_day.copy()
if selected_stocks:
    df_filtered = df_filtered[df_filtered["Stock Code"].isin(selected_stocks)]
if selected_sectors:
    df_filtered = df_filtered[df_filtered["Sector"].isin(selected_sectors)]
if selected_signals:
    df_filtered = df_filtered[df_filtered["Final Signal"].isin(selected_signals)]
df_filtered = df_filtered[df_filtered["Volume Spike (x)"] >= min_spike]
if show_only_spike:
    df_filtered = df_filtered[df_filtered["Unusual Volume"] == True]

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard Harian",
    "📈 Analisis Individual",
    "📋 Data Filter",
    "🏆 Saham Potensial (TOP 20)"
])

# === TAB 1 ===
with tab1:
    if not df_day.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Saham Aktif", f"{len(df_day['Stock Code'].unique()):,.0f}")
        unusual_count = int(df_day["Unusual Volume"].sum()) if "Unusual Volume" in df_day.columns else 0
        total_value = df_day["Value"].sum() if "Value" in df_day.columns else 0
        col2.metric("Saham Unusual Volume", f"{unusual_count:,}")
        col3.metric("Total Nilai Transaksi", f"Rp {total_value:,.0f}")

        col_g, col_l, col_v = st.columns(3)
        for title, data, sort_col, asc in [
            ("Top 10 Gainers (%)", df_day, "Change %", False),
            ("Top 10 Losers (%)", df_day, "Change %", True),
            ("Top 10 by Value", df_day, "Value", False)
        ]:
            with (col_g if "Gainers" in title else col_l if "Losers" in title else col_v):
                st.markdown(f"**{title}**")
                if sort_col in data.columns:
                    df_show = data.sort_values(sort_col, ascending=asc).head(10)
                    st.dataframe(df_show[ensure_cols(df_show, ["Stock Code", "Close", sort_col])],
                                 use_container_width=True, hide_index=True)
    else:
        st.warning("Tidak ada data untuk tanggal terpilih.")

# === TAB 2 ===
with tab2:
    all_stocks = sorted(df["Stock Code"].dropna().unique())
    stock_to_analyze = st.selectbox("Pilih Saham", all_stocks)
    df_stock = df[df['Stock Code'] == stock_to_analyze].sort_values('Last Trading Date')
    if not df_stock.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['Net Foreign Flow'], name='Net FF'), 1, 1)
        fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['Close'], name='Close', line=dict(color='blue')), 1, 1, secondary_y=True)
        fig.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['Volume'], name='Volume', marker_color='gray'), 2, 1)
        fig.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['MA20_vol'], name='MA20 Vol', line=dict(color='red', dash='dot')), 2, 1)
        st.plotly_chart(fig, use_container_width=True)

# === TAB 3 ===
with tab3:
    st.dataframe(df_filtered[ensure_cols(df_filtered, ["Stock Code", "Close", "Change %", "Value", "Net Foreign Flow", "Volume Spike (x)"])],
                 use_container_width=True, hide_index=True)

# === TAB 4 ===
with tab4:
    df_top20 = calculate_potential_score(df, pd.Timestamp(max_date))
    if not df_top20.empty:
        st.dataframe(df_top20, use_container_width=True, hide_index=True)
