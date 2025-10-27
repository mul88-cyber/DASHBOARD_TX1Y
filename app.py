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

# --- KONFIGURASI G-DRIVE ---
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP" 
# Nama file transaksi harian
FILE_NAME = "Kompilasi_Data_1Tahun.csv" 

# Bobot skor (dari skrip Colab Anda)
W = dict(
    trend_akum=0.40, trend_ff=0.30, trend_mfv=0.20, trend_mom=0.10,
    mom_price=0.40,  mom_vol=0.25,  mom_akum=0.25,  mom_ff=0.10,
    blend_trend=0.35, blend_mom=0.35, blend_nbsa=0.20, blend_fcontrib=0.05, blend_unusual=0.05
)

# ==============================================================================
# 📦 3) FUNGSI MEMUAT DATA (via SERVICE ACCOUNT)
# ==============================================================================
def get_gdrive_service():
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None
    except KeyError:
        msg = "❌ Gagal otentikasi: 'st.secrets' tidak menemukan key [gcp_service_account]. Pastikan 'secrets.toml' sudah benar."
        return None, msg
    except Exception as e:
        msg = f"❌ Gagal otentikasi Google Drive: {e}."
        return None, msg

@st.cache_data(ttl=3600)
def load_data():
    """Mencari file transaksi, men-download, membersihkan, dan membacanya ke Pandas."""
    service, error_msg = get_gdrive_service()
    if error_msg:
        return pd.DataFrame(), error_msg, "error"

    try:
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(
            q=query, fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1
        ).execute()
        items = results.get('files', [])

        if not items:
            msg = f"❌ File '{FILE_NAME}' tidak ditemukan di folder GDrive."
            return pd.DataFrame(), msg, "error"

        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)
        
        # Baca CSV
        df = pd.read_csv(fh, dtype=object) # Baca sebagai string dulu
        
        # Pembersihan
        df.columns = df.columns.str.strip()
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce') 
        
        # Kolom angka yang perlu dibersihkan (termasuk Free Float)
        cols_to_numeric = [
            'High', 'Low', 'Close', 'Volume', 'Value', 'Foreign Buy', 'Foreign Sell',
            'Bid Volume', 'Offer Volume', 'Previous', 'Change', 'Open Price', 'First Trade', 
            'Frequency', 'Index Individual', 'Offer', 'Bid', 'Listed Shares', 'Tradeble Shares', 
            'Weight For Index', 'Non Regular Volume', 'Change %', 'Typical Price', 'TPxV', 
            'VWMA_20D', 'MA20_vol', 'MA5_vol', 'Volume Spike (x)', 'Net Foreign Flow', 
            'Bid/Offer Imbalance', 'Money Flow Value', 'Free Float' # Tambahkan Free Float
        ]
        
        # Logika pembersihan string sebelum konversi numerik
        for col in cols_to_numeric:
            if col in df.columns:
                cleaned_col = df[col].astype(str).str.strip()
                # Hapus karakter non-numerik (koma, spasi, Rp, %)
                cleaned_col = cleaned_col.str.replace(r'[,\sRp\%]', '', regex=True) 
                df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0) # Isi NaN dengan 0

        # Pembersihan boolean dan string
        if 'Unusual Volume' in df.columns:
            df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(['spike volume signifikan', 'true', 'True', 'TRUE'])
            df['Unusual Volume'] = df['Unusual Volume'].astype(bool)
        
        if 'Final Signal' in df.columns:
            df['Final Signal'] = df['Final Signal'].astype(str).str.strip()
            
        if 'Sector' in df.columns:
             df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
             df['Sector'] = 'Others' # Fallback jika tidak ada
             
        df = df.dropna(subset=['Last Trading Date', 'Stock Code'])
        
        # Hitung NFF (Rp) (jika belum ada/perlu dihitung ulang)
        if 'NFF (Rp)' not in df.columns:
             if 'Typical Price' in df.columns:
                 df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Typical Price']
             else:
                 df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Close']
        
        msg = f"Data Transaksi Harian berhasil dimuat (file ID: {file_id})."
        return df, msg, "success"
    
    except Exception as e:
        msg = f"❌ Terjadi error saat memuat data Transaksi Harian: {e}."
        return pd.DataFrame(), msg, "error"

# ==============================================================================
# 🛠️ 4) FUNGSI KALKULASI SKOR & NFF (Sama seperti sebelumnya)
# ==============================================================================
def pct_rank(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, method="average").fillna(0) * 100

def to_pct(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1: return pd.Series(50, index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx: return pd.Series(50, index=s.index)
    return (s - mn) / (mx - mn) * 100

@st.cache_data(ttl=3600)
def calculate_potential_score(df: pd.DataFrame, latest_date: pd.Timestamp):
    """Menjalankan logika scoring dari skrip Colab pada data yang ada."""
    trend_start = latest_date - pd.Timedelta(days=30)
    mom_start = latest_date - pd.Timedelta(days=7)
    trend_df = df[df['Last Trading Date'] >= trend_start].copy()
    mom_df = df[df['Last Trading Date'] >= mom_start].copy()
    last_df = df[df['Last Trading Date'] == latest_date].copy()

    if trend_df.empty or mom_df.empty or last_df.empty:
        msg = "Data tidak cukup untuk menghitung skor (kurang dari 30 hari)."
        return pd.DataFrame(), msg, "warning"

    # Trend Score
    tr = trend_df.groupby('Stock Code').agg(
        last_price=('Close', 'last'), last_final_signal=('Final Signal', 'last'),
        total_net_ff_rp=('NFF (Rp)', 'sum'), total_money_flow=('Money Flow Value', 'sum'),
        avg_change_pct=('Change %', 'mean'), sector=('Sector', 'last')
    ).reset_index()
    score_akum = tr['last_final_signal'].map({'Strong Akumulasi': 100, 'Akumulasi': 75, 'Netral': 30, 'Distribusi': 10, 'Strong Distribusi': 0}).fillna(30)
    score_ff = pct_rank(tr['total_net_ff_rp'])
    score_mfv = pct_rank(tr['total_money_flow'])
    score_mom = pct_rank(tr['avg_change_pct'])
    tr['Trend Score'] = (score_akum * W['trend_akum'] + score_ff * W['trend_ff'] +
                         score_mfv * W['trend_mfv'] + score_mom * W['trend_mom'])

    # Momentum Score
    mo = mom_df.groupby('Stock Code').agg(
        total_change_pct=('Change %', 'sum'), had_unusual_volume=('Unusual Volume', 'any'),
        last_final_signal=('Final Signal', 'last'), total_net_ff_rp=('NFF (Rp)', 'sum')
    ).reset_index()
    s_price = pct_rank(mo['total_change_pct'])
    s_vol = mo['had_unusual_volume'].map({True: 100, False: 20}).fillna(20)
    s_akum = mo['last_final_signal'].map({'Strong Akumulasi': 100, 'Akumulasi': 80, 'Netral': 40, 'Distribusi': 10, 'Strong Distribusi': 0}).fillna(40)
    s_ff7 = pct_rank(mo['total_net_ff_rp'])
    mo['Momentum Score'] = (s_price * W['mom_price'] + s_vol * W['mom_vol'] +
                            s_akum * W['mom_akum'] + s_ff7 * W['mom_ff'])

    # NBSA & Foreign Contribution
    nbsa = trend_df.groupby('Stock Code').agg(total_net_ff_30d_rp=('NFF (Rp)', 'sum')).reset_index()
    if {'Foreign Buy', 'Foreign Sell', 'Value'}.issubset(df.columns):
        tmp = trend_df.copy()
        # Perbaiki perhitungan foreign value (pakai NFF Rp saja sebagai proxy)
        tmp['Foreign Value proxy'] = tmp['NFF (Rp)'] # Gunakan NFF Rp sbg proxy nilai asing
        contrib = tmp.groupby('Stock Code').agg(
            total_foreign_value_proxy=('Foreign Value proxy', 'sum'),
            total_value_30d=('Value', 'sum')
        ).reset_index()
        # Hitung kontribusi berdasarkan proxy
        contrib['foreign_contrib_pct'] = np.where(contrib['total_value_30d'] > 0, 
                                                (contrib['total_foreign_value_proxy'].abs() / contrib['total_value_30d']) * 100, 
                                                0)
    else:
        contrib = pd.DataFrame({'Stock Code': [], 'foreign_contrib_pct': []})

    uv = last_df.set_index('Stock Code')['Unusual Volume'].map({True: 1, False: 0})

    # Gabung skor
    rank = tr[['Stock Code', 'Trend Score', 'last_price', 'last_final_signal', 'sector']].merge(
        mo[['Stock Code', 'Momentum Score']], on='Stock Code', how='outer'
    ).merge(
        nbsa, on='Stock Code', how='left'
    ).merge(
        contrib[['Stock Code', 'foreign_contrib_pct']], on='Stock Code', how='left'
    )
    rank['NBSA Score'] = to_pct(rank['total_net_ff_30d_rp'])
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
    score_cols = ['Potential Score', 'Trend Score', 'Momentum Score', 'NBSA Score', 'Foreign Contrib Score']
    for c in score_cols:
        if c in top20.columns: top20[c] = pd.to_numeric(top20[c], errors='coerce').round(2)

    cols_order = ['Analysis Date', 'Stock Code', 'Potential Score', 'Trend Score', 'Momentum Score',
                  'total_net_ff_30d_rp', 'foreign_contrib_pct', 'last_price', 'last_final_signal', 'sector']
    for c in cols_order:
        if c not in top20.columns: top20[c] = np.nan
    top20 = top20[cols_order]
    
    msg = "Skor potensial berhasil dihitung."
    return top20, msg, "success"

@st.cache_data(ttl=3600)
def calculate_nff_top_stocks(df: pd.DataFrame, max_date: pd.Timestamp):
    """Menghitung agregat NFF (Rp) untuk beberapa periode."""
    periods = {'7D': 7, '30D': 30, '90D': 90, '180D': 180}
    results = {}
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    latest_prices = latest_data.get('Close', pd.Series(dtype='float64')) # Handle jika Close tidak ada
    latest_sectors = latest_data.get('Sector', pd.Series(dtype='object')) # Handle jika Sector tidak ada

    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        df_period = df[df['Last Trading Date'] >= start_date].copy()
        nff_agg = df_period.groupby('Stock Code')['NFF (Rp)'].sum()
        df_agg = pd.DataFrame(nff_agg)
        df_agg.columns = ['Total Net FF (Rp)'] 
        df_agg = df_agg.join(latest_prices).join(latest_sectors)
        df_agg.rename(columns={'Close': 'Harga Terakhir'}, inplace=True)
        df_agg = df_agg.sort_values(by='Total Net FF (Rp)', ascending=False)
        results[name] = df_agg.reset_index()

    return results['7D'], results['30D'], results['90D'], results['180D']

# [BARU] Fungsi kalkulasi Money Flow Value (MFV)
@st.cache_data(ttl=3600)
def calculate_mfv_top_stocks(df: pd.DataFrame, max_date: pd.Timestamp):
    """Menghitung agregat MFV (Rp) untuk beberapa periode."""
    periods = {'7D': 7, '30D': 30, '90D': 90, '180D': 180}
    results = {}
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    latest_prices = latest_data.get('Close', pd.Series(dtype='float64'))
    latest_sectors = latest_data.get('Sector', pd.Series(dtype='object'))

    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        df_period = df[df['Last Trading Date'] >= start_date].copy()
        
        # Agregasi Money Flow Value
        mfv_agg = df_period.groupby('Stock Code')['Money Flow Value'].sum() 
        
        df_agg = pd.DataFrame(mfv_agg)
        df_agg.columns = ['Total Money Flow (Rp)'] 
        df_agg = df_agg.join(latest_prices).join(latest_sectors)
        df_agg.rename(columns={'Close': 'Harga Terakhir'}, inplace=True)
        
        # Urutkan berdasarkan MFV
        df_agg = df_agg.sort_values(by='Total Money Flow (Rp)', ascending=False)
        
        results[name] = df_agg.reset_index()

    return results['7D'], results['30D'], results['90D'], results['180D']


# ==============================================================================
# 💎 5) LAYOUT UTAMA (HEADER)
# ==============================================================================
st.title("📈 Dashboard Analisis Saham IDX")
st.caption("Menganalisis data historis harian untuk menemukan saham potensial.")

df, status_msg, status_level = load_data()

if status_level == "success":
    st.toast(status_msg, icon="✅")
elif status_level == "error":
    st.error(status_msg)

# ==============================================================================
# 🧭 6) SIDEBAR FILTER
# ==============================================================================
st.sidebar.header("🎛️ Filter Analisis Harian")

if st.sidebar.button("🔄 Refresh Data (Tarik Ulang dari GDrive)"):
    # Clear SEMUA cache agar kalkulasi ulang
    st.cache_data.clear() 
    st.rerun()

if df.empty:
    st.warning("⚠️ Data belum berhasil dimuat. Dashboard tidak dapat dilanjutkan.")
    st.stop() 

# Filter Tanggal
max_date = df['Last Trading Date'].max().date()
selected_date = st.sidebar.date_input(
    "Pilih Tanggal Analisis",
    max_date,
    min_value=df['Last Trading Date'].min().date(),
    max_value=max_date,
    format="DD-MM-YYYY"
)

df_day = df[df['Last Trading Date'].dt.date == selected_date].copy()

# Filter Lanjutan (untuk Tab 3)
st.sidebar.header("Filter Data Lanjutan (u/ Tab 3)")
selected_stocks_filter = st.sidebar.multiselect( # Ubah nama variabel agar tidak konflik
    "Pilih Saham (Stock Code)",
    options=sorted(df_day["Stock Code"].dropna().unique()),
    placeholder="Ketik kode saham",
    key="filter_stock_multiselect" # Key unik
)

selected_sectors_filter = st.sidebar.multiselect( # Ubah nama variabel
    "Pilih Sektor",
    options=sorted(df_day["Sector"].dropna().unique()),
    placeholder="Pilih sektor",
    key="filter_sector_multiselect" # Key unik
)

selected_signals_filter = st.sidebar.multiselect( # Ubah nama variabel
    "Filter Berdasarkan Final Signal",
    options=sorted(df_day["Final Signal"].dropna().unique()),
    placeholder="Pilih signal",
    key="filter_signal_multiselect" # Key unik
)

min_spike_filter = st.sidebar.slider( # Ubah nama variabel
    "Minimal Volume Spike (x)",
    min_value=1.0,
    max_value=float(df_day["Volume Spike (x)"].max() if not df_day.empty and pd.notna(df_day["Volume Spike (x)"].max()) else 50.0),
    value=1.0, # Default 1.0 agar tidak memfilter by default
    step=0.5,
    key="filter_spike_slider" # Key unik
)

show_only_spike_filter = st.sidebar.checkbox( # Ubah nama variabel
    "Hanya tampilkan Unusual Volume (True)",
    value=False,
    key="filter_spike_checkbox" # Key unik
)

# Terapkan Filter (untuk Tab 3)
df_filtered = df_day.copy()
if selected_stocks_filter:
    df_filtered = df_filtered[df_filtered["Stock Code"].isin(selected_stocks_filter)]
if selected_sectors_filter:
    df_filtered = df_filtered[df_filtered["Sector"].isin(selected_sectors_filter)]
if selected_signals_filter:
    df_filtered = df_filtered[df_filtered["Final Signal"].isin(selected_signals_filter)]
if min_spike_filter > 1.0:
    df_filtered = df_filtered[df_filtered["Volume Spike (x)"] >= min_spike_filter]
if show_only_spike_filter:
    df_filtered = df_filtered[df_filtered["Unusual Volume"] == True]

# ==============================================================================
#  LAYOUT UTAMA (DENGAN TABS BARU)
# ==============================================================================
st.caption(f"Menampilkan data untuk tanggal: **{selected_date.strftime('%d %B %Y')}**")

# [PERUBAHAN] Tambahkan Tab 6
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 **Dashboard Harian**",
    "📈 **Analisis Individual**",
    "📋 **Data Filter**",
    "🏆 **Saham Potensial (TOP 20)**",
    "🌊 **Analisis NFF (Rp)**",
    "💰 **Analisis Money Flow (Rp)**" # Tab Baru
])

# --- TAB 1: DASHBOARD HARIAN ---
with tab1:
    st.subheader("Ringkasan Pasar (pada tanggal terpilih)")
    
    if df_day.empty:
        st.warning(f"Tidak ada data transaksi untuk tanggal {selected_date.strftime('%d-%m-%Y')}.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Saham Aktif", f"{len(df_day['Stock Code'].unique()):,.0f}")
        col2.metric("Saham Unusual Volume", f"{int(df_day['Unusual Volume'].sum()):,.0f}")
        # Format metric Total Nilai Transaksi
        total_value_today = df_day['Value'].sum()
        col3.metric("Total Nilai Transaksi", f"Rp {total_value_today:,.0f}" if pd.notna(total_value_today) else "Rp 0")

        st.markdown("---")
        st.subheader("Top Movers & Most Active")
        
        col_g, col_l, col_v = st.columns(3)
        
        # Fungsi helper format tabel (untuk hindari repetisi)
        def format_movers_df(df_in, value_col=None):
            df_out = df_in[['Stock Code', 'Close', 'Change %']].copy()
            if value_col and value_col in df_in.columns:
                 df_out[value_col] = df_in[value_col]
                 
            df_display = df_out.copy()
            # Format manual
            df_display['Close'] = df_display['Close'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
            if value_col and value_col in df_display.columns:
                 df_display[value_col] = df_display[value_col].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
            return df_display

        with col_g:
            st.markdown("**Top 10 Gainers (%)**")
            top_gainers = df_day.sort_values("Change %", ascending=False).head(10)
            st.dataframe(
                format_movers_df(top_gainers), 
                use_container_width=True, hide_index=True,
                column_config={
                    "Stock Code": st.column_config.TextColumn("Saham"),
                    "Close": st.column_config.TextColumn("Harga"), # Jadi Text krn format manual
                    "Change %": st.column_config.NumberColumn("Change %", format="%.2f") 
                }
            )

        with col_l:
            st.markdown("**Top 10 Losers (%)**")
            top_losers = df_day.sort_values("Change %", ascending=True).head(10)
            st.dataframe(
                format_movers_df(top_losers), 
                use_container_width=True, hide_index=True,
                column_config={
                    "Stock Code": st.column_config.TextColumn("Saham"),
                    "Close": st.column_config.TextColumn("Harga"),
                    "Change %": st.column_config.NumberColumn("Change %", format="%.2f")
                }
            )
            
        with col_v:
            st.markdown("**Top 10 by Value**")
            top_value = df_day.sort_values("Value", ascending=False).head(10)
            st.dataframe(
                format_movers_df(top_value, value_col='Value'), # Tambah value col
                use_container_width=True, hide_index=True,
                column_config={
                    "Stock Code": st.column_config.TextColumn("Saham"),
                    "Close": st.column_config.TextColumn("Harga"),
                    "Value": st.column_config.TextColumn("Nilai") # Jadi Text
                }
            )

        st.markdown("---")
        st.subheader("Distribusi Sektor & Signal")
        
        col_sig, col_sec = st.columns(2)
        
        with col_sig:
            st.markdown("**Distribusi Final Signal (Semua Saham)**")
            if not df_day.empty and 'Final Signal' in df_day.columns:
                signal_counts = df_day["Final Signal"].value_counts().reset_index()
                fig_sig = px.bar(signal_counts, x="Final Signal", y="count", text='count')
                fig_sig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Jumlah: %{y:,.0f}<extra></extra>')
                fig_sig.update_layout(yaxis_title="Jumlah Saham", yaxis=dict(showticklabels=True))
                st.plotly_chart(fig_sig, use_container_width=True)

        with col_sec:
            st.markdown("**Sektor dengan Unusual Volume Terbanyak**")
            # Pastikan kolom Sector ada
            if 'Sector' in df_day.columns:
                spike_df = df_day[df_day['Unusual Volume'] == True]
                if not spike_df.empty:
                    sector_counts = spike_df["Sector"].value_counts().reset_index()
                    fig_sec = px.bar(sector_counts, x="Sector", y="count", text='count')
                    fig_sec.update_traces(texttemplate='%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Jumlah: %{y:,.0f}<extra></extra>')
                    fig_sec.update_layout(yaxis_title="Jumlah Saham", yaxis=dict(showticklabels=True))
                    st.plotly_chart(fig_sec, use_container_width=True)
                else:
                    st.info("Tidak ada saham dengan 'Unusual Volume' pada tanggal ini.")
            else:
                 st.warning("Kolom 'Sector' tidak ada untuk analisis unusual volume.")

# --- TAB 2: ANALISIS INDIVIDUAL ---
with tab2:
    st.subheader("Analisis Time Series Saham Individual")
    
    all_stocks = sorted(df["Stock Code"].dropna().unique())
    stock_to_analyze = st.selectbox(
        "Pilih Saham untuk dianalisis:",
        all_stocks,
        index=all_stocks.index("AADI") if "AADI" in all_stocks else 0,
        key="individual_stock_select" # Key unik
    )
    
    if stock_to_analyze:
        df_stock = df[df['Stock Code'] == stock_to_analyze].sort_values('Last Trading Date')
        
        if df_stock.empty:
            st.warning(f"Tidak ditemukan data historis untuk {stock_to_analyze}")
        else:
            latest_price = df_stock.iloc[-1]['Close']
            # [PERUBAHAN] Ambil Free Float
            free_float = df_stock.iloc[-1]['Free Float'] if 'Free Float' in df_stock.columns else np.nan
            stock_sector = df_stock.iloc[-1]['Sector']
            
            st.markdown(f"**Analisis: {stock_to_analyze} ({stock_sector})**")
            # [PERUBAHAN] Tampilkan Free Float
            col1, col2, col3 = st.columns(3) 
            col1.metric("Harga Terakhir", f"Rp {latest_price:,.0f}" if pd.notna(latest_price) else "N/A")
            col2.metric("Free Float Saham", f"{free_float:.2f}%" if pd.notna(free_float) else "N/A")
            col3.metric("Sektor", stock_sector if pd.notna(stock_sector) else "N/A")

            st.markdown("---")
            
            # [PERUBAHAN] Buat Subplot Gabungan dengan 3 baris
            fig_combined = make_subplots(
                rows=3, # Tiga baris
                cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, # Perkecil jarak
                row_heights=[0.5, 0.25, 0.25], # Atur proporsi tinggi
                specs=[[{"secondary_y": True}], # Baris 1: Harga vs NFF
                       [{"secondary_y": False}],# Baris 2: MFV
                       [{"secondary_y": False}]] # Baris 3: Volume
            )

            # --- Plot 1: Harga (Y-Kanan) vs NFF (Rp) (Y-Kiri) ---
            nff_colors = np.where(df_stock['NFF (Rp)'] >= 0, 'green', 'red')
            fig_combined.add_trace(go.Bar(
                x=df_stock['Last Trading Date'], y=df_stock['NFF (Rp)'], 
                name='Net Foreign Flow (Rp)', marker_color=nff_colors,
                hovertemplate='Tanggal: %{x|%d %b %Y}<br>NFF (Rp): %{y:,.0f}<extra></extra>'
            ), row=1, col=1, secondary_y=False)
            fig_combined.add_trace(go.Scatter(
                x=df_stock['Last Trading Date'], y=df_stock['Close'], 
                name='Harga Penutupan (Rp)', line=dict(color='blue'),
                hovertemplate='Tanggal: %{x|%d %b %Y}<br>Harga: %{y:,.0f}<extra></extra>'
            ), row=1, col=1, secondary_y=True)

            # --- [BARU] Plot 2: Money Flow Value (MFV) (Y-Kiri) ---
            mfv_colors = np.where(df_stock['Money Flow Value'] >= 0, 'lightgreen', 'lightcoral') # Warna berbeda
            fig_combined.add_trace(go.Bar(
                x=df_stock['Last Trading Date'], y=df_stock['Money Flow Value'],
                name='Money Flow Value (Rp)', marker_color=mfv_colors,
                hovertemplate='Tanggal: %{x|%d %b %Y}<br>MFV (Rp): %{y:,.0f}<extra></extra>'
            ), row=2, col=1, secondary_y=False)

            # --- Plot 3: Volume (Y-Kiri) ---
            fig_combined.add_trace(go.Bar(
                x=df_stock['Last Trading Date'], y=df_stock['Volume'], 
                name='Volume (Shares)', marker_color='gray',
                hovertemplate='Tanggal: %{x|%d %b %Y}<br>Volume: %{y:,.0f}<extra></extra>'
            ), row=3, col=1, secondary_y=False)
            fig_combined.add_trace(go.Scatter(
                x=df_stock['Last Trading Date'], y=df_stock['MA20_vol'], 
                name='MA20 Volume (Shares)', line=dict(color='orange', dash='dot'), # Ganti warna MA
                hovertemplate='Tanggal: %{x|%d %b %Y}<br>MA20 Vol: %{y:,.0f}<extra></extra>'
            ), row=3, col=1, secondary_y=False)

            # --- Konfigurasi Layout ---
            fig_combined.update_layout(
                title_text=f"Analisis Harga, Foreign Flow, Money Flow, dan Volume: {stock_to_analyze}",
                height=700, # Perbesar tinggi chart
                xaxis_rangeslider_visible=False,
                xaxis3_rangeslider_visible=True, # Tampilkan slider di plot bawah
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Update label sumbu Y
            fig_combined.update_yaxes(title_text="NFF (Rp)", row=1, col=1, secondary_y=False, showticklabels=True) 
            fig_combined.update_yaxes(title_text="Harga (Rp)", row=1, col=1, secondary_y=True, showticklabels=True)
            fig_combined.update_yaxes(title_text="MFV (Rp)", row=2, col=1, secondary_y=False, showticklabels=True)
            fig_combined.update_yaxes(title_text="Volume", row=3, col=1, secondary_y=False, showticklabels=True)

            st.plotly_chart(fig_combined, use_container_width=True)

# --- TAB 3: DATA FILTER ---
with tab3:
    st.subheader(f"Data Filter (Total: {len(df_filtered)} baris)")
    st.info("Gunakan filter di sidebar kiri untuk menyaring data pada tanggal terpilih.")
    
    # [PERUBAHAN] Tambahkan 'Free Float' ke tampilan tabel
    cols_to_display = [
        "Stock Code", "Close", "Change %", "Value", 
        "Net Foreign Flow", "NFF (Rp)", "Money Flow Value", # Tambah MFV
        "Volume Spike (x)", "Unusual Volume", "Final Signal", "Sector", "Free Float" # Tambah Free Float
    ]
    cols_in_df = [col for col in cols_to_display if col in df_filtered.columns]
    
    df_display_filtered = df_filtered[cols_in_df].copy()
    
    # Format manual
    format_cols_rp = ['Close', 'Value', 'NFF (Rp)', 'Money Flow Value']
    for col in format_cols_rp:
         if col in df_display_filtered.columns:
             df_display_filtered[col] = df_display_filtered[col].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
    
    if 'Net Foreign Flow' in df_display_filtered.columns:
        df_display_filtered['Net Foreign Flow'] = df_display_filtered['Net Foreign Flow'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
        
    st.dataframe(
        df_display_filtered,
        use_container_width=True, hide_index=True,
        column_config={
            "Stock Code": st.column_config.TextColumn("Saham"),
            "Close": st.column_config.TextColumn("Harga"),
            "Change %": st.column_config.NumberColumn("Change %", format="%.2f"),
            "Value": st.column_config.TextColumn("Nilai"),
            "Net Foreign Flow": st.column_config.TextColumn("Net FF (Shares)"), 
            "NFF (Rp)": st.column_config.TextColumn("Net FF (Rp)"), 
            "Money Flow Value": st.column_config.TextColumn("Money Flow (Rp)"), # Tambah MFV
            "Volume Spike (x)": st.column_config.NumberColumn("Spike (x)", format="%.1fx"),
            "Free Float": st.column_config.NumberColumn("Free Float %", format="%.2f%%") # Tambah Free Float
        }
    )

# --- TAB 4: SAHAM POTENSIAL (TOP 20) ---
with tab4:
    st.subheader("🏆 Top 20 Saham Paling Potensial (Overall)")
    st.info(f"Kalkulasi skor didasarkan pada data 30 hari terakhir dari tanggal data terbaru ({max_date.strftime('%d %B %Y')}).")
    
    df_top20, score_msg, score_status = calculate_potential_score(df, pd.Timestamp(max_date))
    
    if score_status == "success":
        st.toast(score_msg, icon="🏆")
    elif score_status == "warning":
        st.warning(score_msg)
    
    if not df_top20.empty:
        # Format manual
        df_top20_display = df_top20.copy()
        format_cols_rp_top20 = ['total_net_ff_30d_rp', 'last_price']
        for col in format_cols_rp_top20:
            if col in df_top20_display.columns:
                 df_top20_display[col] = df_top20_display[col].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')

        st.dataframe(
            df_top20_display,
            use_container_width=True, hide_index=True,
            column_config={
                "Stock Code": st.column_config.TextColumn("Saham"),
                "Potential Score": st.column_config.NumberColumn("Skor", format="%.2f"),
                "Trend Score": st.column_config.NumberColumn("Skor Trend (30h)", format="%.2f"),
                "Momentum Score": st.column_config.NumberColumn("Skor Momentum (7h)", format="%.2f"),
                "total_net_ff_30d_rp": st.column_config.TextColumn("Net FF (30h, Rp)"), 
                "foreign_contrib_pct": st.column_config.NumberColumn("Kontribusi Asing %", format="%.1f%%"),
                "last_price": st.column_config.TextColumn("Harga"),
                "last_final_signal": st.column_config.TextColumn("Signal Terakhir"),
                "sector": st.column_config.TextColumn("Sektor")
            }
        )
    else:
        st.warning("Gagal menghitung skor.")

# --- TAB 5: ANALISIS NFF (Rp) ---
with tab5:
    st.subheader("🌊 Top Akumulasi Net Foreign Flow (NFF) dalam Rupiah") 
    st.info(f"Dihitung berdasarkan data akumulasi dari tanggal data terbaru ({max_date.strftime('%d %B %Y')}) ke belakang.")
    
    try:
        df_nff_7d, df_nff_30d, df_nff_90d, df_nff_180d = calculate_nff_top_stocks(df, pd.Timestamp(max_date))
        
        # Fungsi helper format tabel NFF/MFV
        def format_flow_agg_df(df_in, flow_col_name):
             df_display = df_in.head(20).copy() # Ambil Top 20
             if flow_col_name in df_display.columns:
                  df_display[flow_col_name] = df_display[flow_col_name].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
             if 'Harga Terakhir' in df_display.columns:
                  df_display['Harga Terakhir'] = df_display['Harga Terakhir'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
             return df_display
             
        nff_col_config = {
            "Stock Code": st.column_config.TextColumn("Saham"),
            "Total Net FF (Rp)": st.column_config.TextColumn("Total Net FF (Rp)"), 
            "Harga Terakhir": st.column_config.TextColumn("Harga"),
            "Sector": st.column_config.TextColumn("Sektor")
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**1 Minggu Terakhir (7 Hari)**")
            st.dataframe(format_flow_agg_df(df_nff_7d, 'Total Net FF (Rp)'), use_container_width=True, hide_index=True, column_config=nff_col_config)
            st.markdown("**3 Bulan Terakhir (90 Hari)**")
            st.dataframe(format_flow_agg_df(df_nff_90d, 'Total Net FF (Rp)'), use_container_width=True, hide_index=True, column_config=nff_col_config)
        with col2:
            st.markdown("**1 Bulan Terakhir (30 Hari)**")
            st.dataframe(format_flow_agg_df(df_nff_30d, 'Total Net FF (Rp)'), use_container_width=True, hide_index=True, column_config=nff_col_config)
            st.markdown("**6 Bulan Terakhir (180 Hari)**")
            st.dataframe(format_flow_agg_df(df_nff_180d, 'Total Net FF (Rp)'), use_container_width=True, hide_index=True, column_config=nff_col_config)

    except Exception as e:
         st.error(f"Gagal menghitung analisis NFF: {e}")

# --- TAB 6: [BARU] ANALISIS Money Flow (Rp) ---
with tab6:
    st.subheader("💰 Top Akumulasi Money Flow Value (MFV) dalam Rupiah") 
    st.info(f"MFV adalah proxy untuk inflow/outflow keseluruhan (Lokal + Asing), dihitung dari pergerakan harga intra-hari dan nilai transaksi.")
    st.info(f"Dihitung berdasarkan data akumulasi dari tanggal data terbaru ({max_date.strftime('%d %B %Y')}) ke belakang.")
    
    try:
        # Panggil fungsi kalkulasi MFV
        df_mfv_7d, df_mfv_30d, df_mfv_90d, df_mfv_180d = calculate_mfv_top_stocks(df, pd.Timestamp(max_date))
        
        # Konfigurasi kolom (mirip NFF tapi ganti nama kolom flow)
        mfv_col_config = {
            "Stock Code": st.column_config.TextColumn("Saham"),
            "Total Money Flow (Rp)": st.column_config.TextColumn("Total Money Flow (Rp)"), # Nama kolom MFV
            "Harga Terakhir": st.column_config.TextColumn("Harga"),
            "Sector": st.column_config.TextColumn("Sektor")
        }
        
        col1_mfv, col2_mfv = st.columns(2)
        with col1_mfv:
            st.markdown("**1 Minggu Terakhir (7 Hari)**")
            st.dataframe(format_flow_agg_df(df_mfv_7d, 'Total Money Flow (Rp)'), use_container_width=True, hide_index=True, column_config=mfv_col_config)
            st.markdown("**3 Bulan Terakhir (90 Hari)**")
            st.dataframe(format_flow_agg_df(df_mfv_90d, 'Total Money Flow (Rp)'), use_container_width=True, hide_index=True, column_config=mfv_col_config)
        with col2_mfv:
            st.markdown("**1 Bulan Terakhir (30 Hari)**")
            st.dataframe(format_flow_agg_df(df_mfv_30d, 'Total Money Flow (Rp)'), use_container_width=True, hide_index=True, column_config=mfv_col_config)
            st.markdown("**6 Bulan Terakhir (180 Hari)**")
            st.dataframe(format_flow_agg_df(df_mfv_180d, 'Total Money Flow (Rp)'), use_container_width=True, hide_index=True, column_config=mfv_col_config)

    except Exception as e:
         st.error(f"Gagal menghitung analisis Money Flow: {e}")

