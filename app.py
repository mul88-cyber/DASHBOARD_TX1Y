# ==============================================================================
# 📦 1) IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io # Diperlukan untuk men-download file dari GDrive

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

# ID Folder tempat file CSV Anda disimpan
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP" 
# Nama file CSV yang dicari di dalam folder tersebut
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

# Fungsi helper untuk otentikasi
def get_gdrive_service():
    """Membuat service client untuk Google Drive API menggunakan Streamlit Secrets."""
    try:
        # Coba ambil kredensial dari st.secrets
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None # (service, no_error)
    except KeyError:
        # Error jika [gcp_service_account] tidak ada di secrets.toml
        msg = "❌ Gagal otentikasi: 'st.secrets' tidak menemukan key [gcp_service_account]. Pastikan 'secrets.toml' sudah benar."
        return None, msg
    except Exception as e:
        # Error umum lainnya
        msg = f"❌ Gagal otentikasi Google Drive: {e}. Pastikan 'secrets.toml' benar dan Service Account memiliki akses API GDrive."
        return None, msg

# Fungsi utama untuk memuat data
@st.cache_data(ttl=3600) # Cache data selama 1 jam
def load_data():
    """Mencari file di GDrive, men-download, membersihkan, dan membacanya ke Pandas."""
    service, error_msg = get_gdrive_service()
    if error_msg:
        # Kembalikan error jika otentikasi gagal
        return pd.DataFrame(), error_msg, "error"

    try:
        # 1. Cari file ID terbaru di dalam folder
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name)",
            orderBy="modifiedTime desc", # Ambil yang terbaru
            pageSize=1
        ).execute()
        
        items = results.get('files', [])

        if not items:
            # Error jika file tidak ditemukan di folder
            msg = f"❌ File '{FILE_NAME}' tidak ditemukan di folder GDrive (ID: {FOLDER_ID}). Pastikan file ada DAN sudah di-share ke email robot."
            return pd.DataFrame(), msg, "error"

        # 2. Dapatkan File ID dinamis
        file_id = items[0]['id']
        
        # 3. Download file content
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        fh.seek(0) # Kembali ke awal file di memori

        # 4. Baca ke Pandas
        # ==================== PERBAIKAN ANALISIS MENDALAM ====================
        
        # Daftar kolom yang kita yakini seharusnya angka
        cols_to_numeric = [
            'Change %', 'Typical Price', 'TPxV', 'VWMA_20D', 'MA20_vol', 
            'MA5_vol', 'Volume Spike (x)', 'Net Foreign Flow', 'Foreign Buy', 'Foreign Sell',
            'Bid/Offer Imbalance', 'Money Flow Value', 'Close', 'Volume', 'Value',
            'High','Low','Change','Previous'
        ]
        
        # Paksa baca semua kolom sebagai string dulu (dtype=object)
        # Ini PENTING agar kita bisa membersihkan string "kotor"
        df = pd.read_csv(fh, dtype=object)
        
        # 5. Lakukan pembersihan data
        df.columns = df.columns.str.strip()
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce') 
        
        # Loop khusus untuk membersihkan kolom angka "kotor"
        for col in cols_to_numeric:
            if col in df.columns:
                # 1. Ubah ke string (jaga-jaga), hapus spasi di awal/akhir
                # 2. Hapus koma (pemisah ribuan)
                # 3. Hapus 'Rp' dan spasi lagi
                cleaned_col = df[col].astype(str).str.strip()
                cleaned_col = cleaned_col.str.replace(',', '', regex=False) # Hapus koma
                cleaned_col = cleaned_col.str.replace('Rp', '', regex=False).str.strip()
                
                # 4. Baru ubah ke numeric. Jika masih gagal, jadikan NaN (errors='coerce')
                df[col] = pd.to_numeric(cleaned_col, errors='coerce')

        # ==================== PERBAIKAN SELESAI ====================

        # Logika pembersihan standar (setelah jadi numerik)
        if 'Unusual Volume' in df.columns:
            if df['Unusual Volume'].dtype == 'object':
                df['Unusual Volume'] = df['Unusual Volume'].str.strip().str.lower().isin(['spike volume signifikan', 'true', 'True', 'TRUE'])
            df['Unusual Volume'] = df['Unusual Volume'].astype(bool)
        
        if 'Final Signal' in df.columns:
             if df['Final Signal'].dtype == 'object':
                df['Final Signal'] = df['Final Signal'].str.strip()
            
        df = df.dropna(subset=['Last Trading Date', 'Stock Code'])
        
        # Buat kolom NFF (Rp) (sekarang dijamin numerik)
        if 'Typical Price' in df.columns and 'Net Foreign Flow' in df.columns:
            df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Typical Price']
        else:
            # Fallback jika 'Typical Price' tidak ada, gunakan Close
            df['NFF (Rp)'] = df['Net Foreign Flow'] * df['Close']
            
        msg = f"Data berhasil dimuat (file ID: {file_id}). Pembersihan angka selesai."
        return df, msg, "success"
    
    except Exception as e:
        # Error umum saat download/baca
        msg = f"❌ Terjadi error saat memuat data: {e}. Jika ini HttpError 403, pastikan Anda sudah 'Share' folder GDrive ke email Service Account."
        return pd.DataFrame(), msg, "error"

# ==============================================================================
# 🛠️ 4) FUNGSI KALKULASI SKOR
# ==============================================================================

def pct_rank(s: pd.Series):
    """Menghitung Percentile Rank (0-100)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, method="average").fillna(0) * 100

def to_pct(s: pd.Series):
    """Melakukan normalisasi Min-Max (0-100)."""
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1: return pd.Series(50, index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx: return pd.Series(50, index=s.index)
    return (s - mn) / (mx - mn) * 100

@st.cache_data(ttl=3600)
def calculate_potential_score(df: pd.DataFrame, latest_date: pd.Timestamp):
    """Menjalankan logika scoring dari skrip Colab pada data yang ada."""
    
    # === Window tanggal
    trend_start = latest_date - pd.Timedelta(days=30)
    mom_start = latest_date - pd.Timedelta(days=7)

    trend_df = df[df['Last Trading Date'] >= trend_start].copy()
    mom_df = df[df['Last Trading Date'] >= mom_start].copy()
    last_df = df[df['Last Trading Date'] == latest_date].copy()

    if trend_df.empty or mom_df.empty or last_df.empty:
        msg = "Data tidak cukup untuk menghitung skor (kurang dari 30 hari)."
        return pd.DataFrame(), msg, "warning"

    # === TR E N D (30 hari)
    tr = trend_df.groupby('Stock Code').agg(
        last_price=('Close', 'last'),
        last_final_signal=('Final Signal', 'last'),
        total_net_ff_rp=('NFF (Rp)', 'sum'), # Pakai (Rp) untuk skor
        total_money_flow=('Money Flow Value', 'sum'),
        avg_change_pct=('Change %', 'mean'),
        sector=('Sector', 'last')
    ).reset_index()

    score_akum = tr['last_final_signal'].map({'Strong Akumulasi': 100, 'Akumulasi': 75, 'Netral': 30, 'Distribusi': 10, 'Strong Distribusi': 0}).fillna(30)
    score_ff = pct_rank(tr['total_net_ff_rp']) # Pakai (Rp) untuk skor
    score_mfv = pct_rank(tr['total_money_flow'])
    score_mom = pct_rank(tr['avg_change_pct'])
    tr['Trend Score'] = (score_akum * W['trend_akum'] + score_ff * W['trend_ff'] +
                         score_mfv * W['trend_mfv'] + score_mom * W['trend_mom'])

    # === M O M E N T U M (7 hari)
    mo = mom_df.groupby('Stock Code').agg(
        total_change_pct=('Change %', 'sum'),
        had_unusual_volume=('Unusual Volume', 'any'),
        last_final_signal=('Final Signal', 'last'),
        total_net_ff_rp=('NFF (Rp)', 'sum') # Pakai (Rp) untuk skor
    ).reset_index()

    s_price = pct_rank(mo['total_change_pct'])
    s_vol = mo['had_unusual_volume'].map({True: 100, False: 20}).fillna(20)
    s_akum = mo['last_final_signal'].map({'Strong Akumulasi': 100, 'Akumulasi': 80, 'Netral': 40, 'Distribusi': 10, 'Strong Distribusi': 0}).fillna(40)
    s_ff7 = pct_rank(mo['total_net_ff_rp']) # Pakai (Rp) untuk skor
    mo['Momentum Score'] = (s_price * W['mom_price'] + s_vol * W['mom_vol'] +
                            s_akum * W['mom_akum'] + s_ff7 * W['mom_ff'])

    # === NBSA & Foreign Contribution (30 hari)
    nbsa = trend_df.groupby('Stock Code').agg(total_net_ff_30d_rp=('NFF (Rp)', 'sum')).reset_index() # Pakai (Rp) untuk skor
    
    # Kolom 'Foreign Buy' dan 'Foreign Sell' adalah volume (shares), bukan value (Rp).
    # Untuk FContrib%, kita butuh Foreign Value vs Total Value.
    # Karena data kita tidak punya Foreign Value (Rp), kita set FContrib% ke NaN
    # dan biarkan bobotnya diisi 50 (netral).
    rank_contrib = pd.DataFrame({'Stock Code': nbsa['Stock Code'], 'foreign_contrib_pct': np.nan})

    # === Unusual bonus (hari terakhir)
    uv = last_df.set_index('Stock Code')['Unusual Volume'].map({True: 1, False: 0})

    # === GABUNG skor → Potential Score
    rank = tr[['Stock Code', 'Trend Score', 'last_price', 'last_final_signal', 'sector']].merge(
        mo[['Stock Code', 'Momentum Score']], on='Stock Code', how='outer'
    ).merge(
        nbsa, on='Stock Code', how='left' # nbsa (Rp)
    ).merge(
        rank_contrib[['Stock Code', 'foreign_contrib_pct']],
        on='Stock Code', how='left'
    )

    rank['NBSA Score'] = to_pct(rank['total_net_ff_30d_rp']) # Pakai (Rp) untuk skor
    rank['Foreign Contrib Score'] = to_pct(rank['foreign_contrib_pct']) # Akan jadi 50 (netral)
    unusual_bonus = uv.reindex(rank['Stock Code']).fillna(0) * 5
    rank['Potential Score'] = (
        rank['Trend Score'].fillna(0) * W['blend_trend'] +
        rank['Momentum Score'].fillna(0) * W['blend_mom'] +
        rank['NBSA Score'].fillna(50) * W['blend_nbsa'] +
        rank['Foreign Contrib Score'].fillna(50) * W['blend_fcontrib'] + 
        unusual_bonus.values * W['blend_unusual']
    )

    # === TOP 20 & format
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

# ==============================================================================
# 🛠️ 5) FUNGSI KALKULASI NFF (BARU)
# ==============================================================================
@st.cache_data(ttl=3600)
def calculate_nff_top_stocks(df: pd.DataFrame, max_date: pd.Timestamp):
    """Menghitung agregat NFF untuk beberapa periode dari max_date."""
    
    periods = {
        '7D': 7,
        '30D': 30,
        '90D': 90,
        '180D': 180
    }
    
    results = {}
    
    # Ambil data harga dan sektor terakhir untuk merge
    latest_data = df[df['Last Trading Date'] == max_date].set_index('Stock Code')
    latest_prices = latest_data['Close']
    latest_sectors = latest_data['Sector']

    for name, days in periods.items():
        start_date = max_date - pd.Timedelta(days=days)
        
        # Filter data untuk periode
        df_period = df[df['Last Trading Date'] >= start_date].copy()
        
        # Agregasi NFF (Rp) - Pakai (Rp) untuk tab ini
        nff_agg = df_period.groupby('Stock Code')['NFF (Rp)'].sum()
        
        # Gabungkan dengan data terakhir
        df_agg = pd.DataFrame(nff_agg)
        df_agg.columns = ['Total Net FF (Rp)'] 
        df_agg = df_agg.join(latest_prices).join(latest_sectors)
        
        # Ganti nama 'Close'
        df_agg.rename(columns={'Close': 'Harga Terakhir'}, inplace=True)
        
        # Urutkan
        df_agg = df_agg.sort_values(by='Total Net FF (Rp)', ascending=False) # Sort by Rp
        
        results[name] = df_agg.reset_index()

    return results['7D'], results['30D'], results['90D'], results['180D']

# ==============================================================================
# 💎 6) LAYOUT UTAMA (HEADER)
# ==============================================================================
st.title("📈 Dashboard Analisis Saham IDX")
st.caption("Menganalisis data historis harian untuk menemukan saham potensial.")

# Pindahkan pemanggilan status ke sini, di luar cache
status_container = st.empty()

# Panggil data dan tangkap statusnya
df, status_msg, status_level = load_data()

# Tampilkan notifikasi (toast/error) di sini, BUKAN di dalam load_data
if status_level == "success":
    st.toast(status_msg, icon="✅")
elif status_level == "error":
    st.error(status_msg)

# ==============================================================================
# 🧭 7) SIDEBAR FILTER
# ==============================================================================
st.sidebar.header("🎛️ Filter Analisis Harian")

if st.sidebar.button("🔄 Refresh Data (Tarik Ulang dari GDrive)"):
    st.cache_data.clear() # Hapus cache
    st.rerun() # Jalankan ulang skrip

# Pindahkan pengecekan df.empty ke sini
if df.empty:
    st.warning("⚠️ Data belum berhasil dimuat. Silakan cek 'secrets.toml' dan izin GDrive Anda, lalu klik 'Refresh Data'.")
    st.stop() # Hentikan eksekusi skrip jika data gagal dimuat

# --- Filter Tanggal ---
max_date = df['Last Trading Date'].max().date()
selected_date = st.sidebar.date_input(
    "Pilih Tanggal Analisis",
    max_date,
    min_value=df['Last Trading Date'].min().date(),
    max_value=max_date,
    format="DD-MM-YYYY"
)

df_day = df[df['Last Trading Date'].dt.date == selected_date].copy()

# --- Filter Lanjutan (untuk Tab 3) ---
st.sidebar.header("Filter Data Lanjutan (u/ Tab 3)")
selected_stocks = st.sidebar.multiselect(
    "Pilih Saham (Stock Code)",
    options=sorted(df_day["Stock Code"].dropna().unique()),
    placeholder="Ketik kode saham"
)

selected_sectors = st.sidebar.multiselect(
    "Pilih Sektor",
    options=sorted(df_day["Sector"].dropna().unique()),
    placeholder="Pilih sektor"
)

selected_signals = st.sidebar.multiselect(
    "Filter Berdasarkan Final Signal",
    options=sorted(df_day["Final Signal"].dropna().unique()),
    placeholder="Pilih signal"
)

min_spike = st.sidebar.slider(
    "Minimal Volume Spike (x)",
    min_value=1.0,
    max_value=float(df_day["Volume Spike (x)"].max() if not df_day.empty and df_day["Volume Spike (x)"].max() > 1.0 else 50.0),
    value=2.0,
    step=0.5
)

show_only_spike = st.sidebar.checkbox(
    "Hanya tampilkan Unusual Volume (True)",
    value=False
)

# --- Terapkan Filter (untuk Tab 3) ---
df_filtered = df_day.copy()
if selected_stocks:
    df_filtered = df_filtered[df_filtered["Stock Code"].isin(selected_stocks)]
if selected_sectors:
    df_filtered = df_filtered[df_filtered["Sector"].isin(selected_sectors)]
if selected_signals:
    df_filtered = df_filtered[df_filtered["Final Signal"].isin(selected_signals)]

if min_spike > 1.0:
    df_filtered = df_filtered[df_filtered["Volume Spike (x)"] >= min_spike]

if show_only_spike:
    df_filtered = df_filtered[df_filtered["Unusual Volume"] == True]

# ==============================================================================
#  LAYOUT UTAMA (DENGAN TABS)
# ==============================================================================
st.caption(f"Menampilkan data untuk tanggal: **{selected_date.strftime('%d %B %Y')}**")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 **Dashboard Harian**",
    "📈 **Analisis Individual**",
    "📋 **Data Filter**",
    "🏆 **Saham Potensial (TOP 20)**",
    "🌊 **Analisis NFF (Rp)**" 
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
        
        # --- PERBAIKAN MANUAL FORMAT (METRIC) ---
        metric_value = df_day['Value'].sum()
        col3.metric("Total Nilai Transaksi", f"Rp {metric_value:,.0f}" if pd.notna(metric_value) else "N/A")

        st.markdown("---")
        st.subheader("Top Movers & Most Active")
        
        col_g, col_l, col_v = st.columns(3)
        
        with col_g:
            st.markdown("**Top 10 Gainers (%)**")
            top_gainers = df_day.sort_values("Change %", ascending=False).head(10)
            
            # --- PERBAIKAN MANUAL FORMAT ---
            # Buat copy untuk ditampilkan
            df_display_g = top_gainers[['Stock Code', 'Close', 'Change %']].copy()
            # Terapkan format string manual
            df_display_g['Close'] = df_display_g['Close'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
            
            st.dataframe(
                df_display_g, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Stock Code": st.column_config.TextColumn("Saham"),
                    "Close": st.column_config.TextColumn("Harga"), # Tampilkan sebagai Teks
                    "Change %": st.column_config.NumberColumn("Change %", format="%.2f") # Angka bisa diformat
                }
            )

        with col_l:
            st.markdown("**Top 10 Losers (%)**")
            top_losers = df_day.sort_values("Change %", ascending=True).head(10)

            # --- PERBAIKAN MANUAL FORMAT ---
            df_display_l = top_losers[['Stock Code', 'Close', 'Change %']].copy()
            df_display_l['Close'] = df_display_l['Close'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')

            st.dataframe(
                df_display_l, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Stock Code": st.column_config.TextColumn("Saham"),
                    "Close": st.column_config.TextColumn("Harga"), # Tampilkan sebagai Teks
                    "Change %": st.column_config.NumberColumn("Change %", format="%.2f")
                }
            )
            
        with col_v:
            st.markdown("**Top 10 by Value**")
            top_value = df_day.sort_values("Value", ascending=False).head(10)

            # --- PERBAIKAN MANUAL FORMAT ---
            df_display_v = top_value[['Stock Code', 'Close', 'Value']].copy()
            df_display_v['Close'] = df_display_v['Close'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
            df_display_v['Value'] = df_display_v['Value'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')

            st.dataframe(
                df_display_v, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Stock Code": st.column_config.TextColumn("Saham"),
                    "Close": st.column_config.TextColumn("Harga"), # Tampilkan sebagai Teks
                    "Value": st.column_config.TextColumn("Nilai")  # Tampilkan sebagai Teks
                }
            )

        st.markdown("---")
        st.subheader("Distribusi Sektor & Signal")
        
        col_sig, col_sec = st.columns(2)
        
        with col_sig:
            st.markdown("**Distribusi Final Signal (Semua Saham)**")
            if not df_day.empty and 'Final Signal' in df_day.columns:
                signal_counts = df_day["Final Signal"].value_counts().reset_index()
                fig_sig = px.bar(
                    signal_counts, 
                    x="Final Signal", 
                    y="count", 
                    title="Distribusi Final Signal",
                    text='count'
                )
                fig_sig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Jumlah: %{y:,.0f}<extra></extra>')
                fig_sig.update_layout(yaxis_title="Jumlah Saham", yaxis=dict(showticklabels=True))
                st.plotly_chart(fig_sig, use_container_width=True)

        with col_sec:
            st.markdown("**Sektor dengan Unusual Volume Terbanyak**")
            spike_df = df_day[df_day['Unusual Volume'] == True]
            if not spike_df.empty and 'Sector' in spike_df.columns:
                sector_counts = spike_df["Sector"].value_counts().reset_index()
                fig_sec = px.bar(
                    sector_counts, 
                    x="Sector", 
                    y="count", 
                    title="Distribusi Sektor (Hanya Unusual Volume)",
                    text='count'
                )
                fig_sec.update_traces(texttemplate='%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Jumlah: %{y:,.0f}<extra></extra>')
                fig_sec.update_layout(yaxis_title="Jumlah Saham", yaxis=dict(showticklabels=True))
                st.plotly_chart(fig_sec, use_container_width=True)
            else:
                st.info("Tidak ada saham dengan 'Unusual Volume' pada tanggal ini.")

# --- TAB 2: ANALISIS INDIVIDUAL ---
with tab2:
    st.subheader("Analisis Time Series Saham Individual")
    
    all_stocks = sorted(df["Stock Code"].dropna().unique())
    
    if not all_stocks:
        st.warning("Tidak ada data saham untuk dipilih.")
    else:
        # Cari 'AADI' atau default ke saham pertama
        default_index = 0
        if "AADI" in all_stocks:
            default_index = all_stocks.index("AADI")
            
        stock_to_analyze = st.selectbox(
            "Pilih Saham untuk dianalisis:",
            all_stocks,
            index=default_index
        )
        
        if stock_to_analyze:
            df_stock = df[df['Stock Code'] == stock_to_analyze].sort_values('Last Trading Date')
            
            if df_stock.empty:
                st.warning(f"Tidak ditemukan data historis untuk {stock_to_analyze}")
            else:
                # Menampilkan kode saham, bukan 'Company Name'
                st.info(f"Menampilkan data untuk: **{stock_to_analyze}**")
                
                # --- Buat Subplot Gabungan ---
                fig_combined = make_subplots(
                    rows=2, 
                    cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    specs=[[{"secondary_y": True}],
                           [{"secondary_y": False}]]
                )

                # --- Plot 1: Harga (Y-Kanan) vs NFF (Y-Kiri) ---
                # Menggunakan 'Net Foreign Flow' (Shares)
                nff_colors = np.where(df_stock['Net Foreign Flow'] >= 0, 'green', 'red')

                fig_combined.add_trace(go.Bar(
                    x=df_stock['Last Trading Date'],
                    y=df_stock['Net Foreign Flow'], # Pakai Shares
                    name='Net Foreign Flow (Shares)', # Label Shares
                    marker_color=nff_colors,
                    hovertemplate='Tanggal: %{x}<br>NFF (Shares): %{y:,.0f}<extra></extra>' # Format hover
                ), row=1, col=1, secondary_y=False)

                fig_combined.add_trace(go.Scatter(
                    x=df_stock['Last Trading Date'],
                    y=df_stock['Close'],
                    name='Harga Penutupan (Rp)',
                    line=dict(color='blue'),
                    hovertemplate='Tanggal: %{x}<br>Harga: %{y:,.0f}<extra></extra>' # Format hover
                ), row=1, col=1, secondary_y=True)

                # --- Plot 2: Volume (Y-Kiri) ---
                fig_combined.add_trace(go.Bar(
                    x=df_stock['Last Trading Date'],
                    y=df_stock['Volume'],
                    name='Volume (Shares)',
                    marker_color='gray',
                    hovertemplate='Tanggal: %{x}<br>Volume: %{y:,.0f}<extra></extra>' # Format hover
                ), row=2, col=1, secondary_y=False)

                if 'MA20_vol' in df_stock.columns:
                    fig_combined.add_trace(go.Scatter(
                        x=df_stock['Last Trading Date'],
                        y=df_stock['MA20_vol'],
                        name='MA20 Volume (Shares)',
                        line=dict(color='red', dash='dot'),
                        hovertemplate='Tanggal: %{x}<br>MA20 Vol: %{y:,.0f}<extra></extra>' # Format hover
                    ), row=2, col=1, secondary_y=False)

                # --- Konfigurasi Layout ---
                fig_combined.update_layout(
                    title_text=f"Analisis Harga, Foreign Flow (Shares), dan Volume: {stock_to_analyze}",
                    height=600,
                    xaxis_rangeslider_visible=False,
                    xaxis2_rangeslider_visible=True, # Tampilkan rangeslider di chart bawah
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Biarkan Plotly auto-range dan tampilkan label
                fig_combined.update_yaxes(title_text="Net Foreign Flow (Shares)", row=1, col=1, secondary_y=False, showticklabels=True)
                fig_combined.update_yaxes(title_text="Harga Penutupan (Rp)", row=1, col=1, secondary_y=True, showticklabels=True)
                fig_combined.update_yaxes(title_text="Volume (Shares)", row=2, col=1, secondary_y=False, showticklabels=True)

                st.plotly_chart(fig_combined, use_container_width=True)

# --- TAB 3: DATA FILTER ---
with tab3:
    st.subheader(f"Data Filter (Total: {len(df_filtered)} baris)")
    st.info("Gunakan filter di sidebar kiri untuk menyaring data pada tanggal terpilih.")
    
    # Menampilkan 'NFF (Rp)' dan 'Net Foreign Flow' (Shares)
    cols_to_display = [
        "Stock Code", "Close", "Change %", "Value", 
        "Net Foreign Flow", "NFF (Rp)", "Volume Spike (x)", 
        "Unusual Volume", "Final Signal", "Sector"
    ]
    # Pastikan semua kolom ada sebelum ditampilkan
    cols_in_df = [col for col in cols_to_display if col in df_filtered.columns]
    
    # --- PERBAIKAN MANUAL FORMAT ---
    df_display_filtered = df_filtered[cols_in_df].copy()
    format_cols_rp = ['Close', 'Value', 'NFF (Rp)']
    for col in format_cols_rp:
        if col in df_display_filtered.columns:
            # Terapkan format string manual
            df_display_filtered[col] = df_display_filtered[col].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
            
    if 'Net Foreign Flow' in df_display_filtered.columns:
         # Terapkan format string manual (tanpa Rp)
         df_display_filtered['Net Foreign Flow'] = df_display_filtered['Net Foreign Flow'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')

    st.dataframe(
        df_display_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Stock Code": st.column_config.TextColumn("Saham"),
            "Close": st.column_config.TextColumn("Harga"), # Tampilkan sebagai Teks
            "Change %": st.column_config.NumberColumn("Change %", format="%.2f"),
            "Value": st.column_config.TextColumn("Nilai"), # Tampilkan sebagai Teks
            "Net Foreign Flow": st.column_config.TextColumn("Net FF (Shares)"), # Tampilkan sebagai Teks
            "NFF (Rp)": st.column_config.TextColumn("Net FF (Rp)"), # Tampilkan sebagai Teks
            "Volume Spike (x)": st.column_config.NumberColumn("Spike (x)", format="%.1fx")
            # Kolom lain (Unusual Volume, Signal, Sector) akan tampil apa adanya (OK)
        }
    )

# --- TAB 4: SAHAM POTENSIAL (TOP 20) ---
with tab4:
    st.subheader("🏆 Top 20 Saham Paling Potensial (Overall)")
    st.info(f"Kalkulasi skor ini didasarkan pada data 30 hari terakhir, dihitung dari tanggal data terbaru ({max_date.strftime('%d %B %Y')}).")
    
    # Panggil fungsi kalkulasi skor (pakai NFF Rp)
    df_top20, score_msg, score_status = calculate_potential_score(df, pd.Timestamp(max_date))
    
    # Tampilkan notifikasi (toast/warning) di luar fungsi cache
    if score_status == "success":
        st.toast(score_msg, icon="🏆")
    elif score_status == "warning":
        st.warning(score_msg)
    
    if not df_top20.empty:
        # --- PERBAIKAN MANUAL FORMAT ---
        df_display_top20 = df_top20.copy()
        # Terapkan format string manual
        df_display_top20['total_net_ff_30d_rp'] = df_display_top20['total_net_ff_30d_rp'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
        df_display_top20['last_price'] = df_display_top20['last_price'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')

        st.dataframe(
            df_display_top20,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Stock Code": st.column_config.TextColumn("Saham"),
                "Potential Score": st.column_config.NumberColumn("Skor", format="%.2f", help="Skor gabungan dari Trend, Momentum, NBSA, dll."),
                "Trend Score": st.column_config.NumberColumn("Skor Trend (30h)", format="%.2f"),
                "Momentum Score": st.column_config.NumberColumn("Skor Momentum (7h)", format="%.2f"),
                "total_net_ff_30d_rp": st.column_config.TextColumn("Net FF (30h, Rp)"), # Tampilkan sebagai Teks
                "foreign_contrib_pct": st.column_config.NumberColumn("Kontribusi Asing %", format="%.1f%%"),
                "last_price": st.column_config.TextColumn("Harga"), # Tampilkan sebagai Teks
                "last_final_signal": st.column_config.TextColumn("Signal Terakhir"),
                "sector": st.column_config.TextColumn("Sektor")
            }
        )
    else:
        st.warning("Gagal menghitung skor. Periksa apakah data cukup.")

# --- TAB 5: ANALISIS NFF (BARU) ---
with tab5:
    st.subheader("🌊 Top Akumulasi Net Foreign Flow (NFF) dalam Rupiah") 
    st.info(f"Dihitung berdasarkan data akumulasi dari tanggal data terbaru ({max_date.strftime('%d %B %Y')}) ke belakang.")
    
    # Panggil fungsi kalkulasi NFF (pakai NFF Rp)
    df_7d, df_30d, df_90d, df_180d = calculate_nff_top_stocks(df, pd.Timestamp(max_date))
    
    # --- FUNGSI HELPER UNTUK FORMAT MANUAL DI TAB INI ---
    def format_nff_df(df_in):
        df_display = df_in.head(20).copy()
        df_display['Total Net FF (Rp)'] = df_display['Total Net FF (Rp)'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
        df_display['Harga Terakhir'] = df_display['Harga Terakhir'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
        return df_display

    # Konfigurasi kolom (sekarang hanya untuk rename)
    nff_column_config = {
        "Stock Code": st.column_config.TextColumn("Saham"),
        "Total Net FF (Rp)": st.column_config.TextColumn("Total Net FF (Rp)"), # Tampilkan sebagai Teks
        "Harga Terakhir": st.column_config.TextColumn("Harga"), # Tampilkan sebagai Teks
        "Sector": st.column_config.TextColumn("Sektor")
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1 Minggu Terakhir (7 Hari)**")
        st.dataframe(
            format_nff_df(df_7d), # Terapkan format manual
            use_container_width=True,
            hide_index=True,
            column_config=nff_column_config
        )
        
        st.markdown("**3 Bulan Terakhir (90 Hari)**")
        st.dataframe(
            format_nff_df(df_90d), # Terapkan format manual
            use_container_width=True,
            hide_index=True,
            column_config=nff_column_config
        )

    with col2:
        st.markdown("**1 Bulan Terakhir (30 Hari)**")
        st.dataframe(
            format_nff_df(df_30d), # Terapkan format manual
            use_container_width=True,
            hide_index=True,
            column_config=nff_column_config
        )
        
        st.markdown("**6 Bulan Terakhir (180 Hari)**")
        st.dataframe(
            format_nff_df(df_180d), # Terapkan format manual
            use_container_width=True,
            hide_index=True,
            column_config=nff_column_config
        )

