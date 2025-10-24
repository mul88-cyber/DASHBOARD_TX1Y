# ==============================================================================
# 📦 1) IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io # Diperlukan untuk men-download file dari GDrive API
import numpy as np # Diperlukan untuk kalkulasi skor

# Import untuk Google Service Account
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ 2) KONFIGURASI DASHBOARD & DATA
# ==============================================================================
st.set_page_config(
    page_title="📊 Dashboard Analisis Saham IDX",
    layout="wide",
    page_icon="📈"
)

st.title("📈 Dashboard Analisis Saham IDX")
st.caption("Menganalisis data historis untuk menemukan saham potensial (Data dari Google Drive).")

# ID Folder Google Drive (Statis, tidak berubah)
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP"
FILE_NAME = "Kompilasi_Data_1Tahun.csv"

# Bobot skor (dari skrip Colab)
W = dict(
    trend_akum=0.40, trend_ff=0.30, trend_mfv=0.20, trend_mom=0.10,
    mom_price=0.40,  mom_vol=0.25,  mom_akum=0.25,  mom_ff=0.10,
    blend_trend=0.35, blend_mom=0.35, blend_nbsa=0.20, blend_fcontrib=0.05, blend_unusual=0.05
)

# ==============================================================================
# 🛠️ 3) FUNGSI UTILITAS SKOR (DARI COLAB)
# ==============================================================================
def pct_rank(s: pd.Series):
    """Menghitung Percentile Rank, menangani non-numerik."""
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, method="average").fillna(0) * 100

def to_pct(s: pd.Series):
    """Men-skala data ke 0-100, menangani edge cases."""
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1:
        return pd.Series(50, index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(50, index=s.index)
    return (s - mn) / (mx - mn) * 100

# ==============================================================================
# 📦 4) MEMUAT DAN MEMBERSIHKAN DATA (WORKFLOW BARU)
# ==============================================================================

def get_gdrive_service():
    """Membuat service GDrive menggunakan Streamlit Secrets."""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict)
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"❌ Gagal koneksi ke Google Service Account: {e}")
        st.error("Pastikan 'secrets.toml' Anda sudah di-setup dengan benar di Streamlit Secrets.")
        return None

def get_latest_file_id(service, folder_id, file_name):
    """Mencari FILE_ID terbaru di dalam FOLDER_ID berdasarkan nama file."""
    if service is None:
        return None
    try:
        query = f"'{folder_id}' in parents and name='{file_name}' and trashed=false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name, modifiedTime)').execute()
        files = response.get('files', [])
        
        if not files:
            st.error(f"❌ File '{file_name}' tidak ditemukan di folder GDrive Anda.")
            return None
        
        # Sortir berdasarkan tanggal modifikasi, ambil yang terbaru
        latest_file = sorted(files, key=lambda x: x['modifiedTime'], reverse=True)[0]
        st.info(f"File '{latest_file['name']}' ditemukan (ID: ...{latest_file['id'][-10:]})")
        return latest_file['id']
        
    except HttpError as error:
        st.error(f"❌ Error HTTP saat mencari file: {error}")
        st.error("Pastikan Anda sudah 'Share' folder GDrive Anda ke email Service Account (sebagai Viewer).")
        return None
    except Exception as e:
        st.error(f"❌ Error saat mencari file: {e}")
        return None

def download_file_from_drive(service, file_id):
    """Men-download file dari GDrive menggunakan FILE_ID dan mengembalikannya sebagai DataFrame."""
    if service is None or file_id is None:
        return pd.DataFrame()
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            # st.progress(int(status.progress() * 100)) # Opsional: progress bar
        
        fh.seek(0)
        return fh
    except Exception as e:
        st.error(f"❌ Gagal men-download file: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600) # Cache selama 1 jam
def load_data():
    """Fungsi utama: login, cari file, download, dan bersihkan data."""
    
    service = get_gdrive_service()
    file_id = get_latest_file_id(service, FOLDER_ID, FILE_NAME)
    
    if file_id is None:
        return pd.DataFrame()
        
    file_handle = download_file_from_drive(service, file_id)

    try:
        df = pd.read_csv(file_handle)
        df.columns = df.columns.str.strip()
        
        # PERBAIKAN PENTING: Menangani tanggal rusak (misal: '00-00-00')
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')
        
        cols_to_numeric = [
            'Change %', 'Typical Price', 'TPxV', 'VWMA_20D', 'MA20_vol', 
            'MA5_vol', 'Volume Spike (x)', 'Net Foreign Flow', 
            'Bid/Offer Imbalance', 'Money Flow Value', 'Close', 'Volume', 
            'Value', 'Foreign Buy', 'Foreign Sell'
        ]
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Unusual Volume' in df.columns:
            if df['Unusual Volume'].dtype == 'object':
                df['Unusual Volume'] = df['Unusual Volume'].str.strip().str.lower().isin(['spike volume signifikan', 'true'])
            df['Unusual Volume'] = df['Unusual Volume'].astype(bool)
        
        # Buang baris yang tanggalnya rusak/kosong
        df = df.dropna(subset=['Last Trading Date', 'Stock Code'])
        return df
    
    except Exception as e:
        st.error(f"❌ Gagal mem-parsing file CSV: {e}")
        return pd.DataFrame()

# ==============================================================================
# 🔬 5) FUNGSI KALKULASI SKOR (LOGIKA COLAB)
# ==============================================================================
@st.cache_data(ttl=3600)
def calculate_potential_score(df_all_history):
    """Menjalankan logika scoring dari Colab pada DataFrame."""
    
    if df_all_history.empty:
        return pd.DataFrame()

    print("Menghitung skor potensial...")
    
    # === Window tanggal
    latest_date = df_all_history['Last Trading Date'].max()
    trend_start = latest_date - pd.Timedelta(days=30)
    mom_start   = latest_date - pd.Timedelta(days=7)
    
    trend_df = df_all_history[df_all_history['Last Trading Date'] >= trend_start].copy()
    mom_df   = df_all_history[df_all_history['Last Trading Date'] >= mom_start].copy()
    last_df  = df_all_history[df_all_history['Last Trading Date'] == latest_date].copy()

    if trend_df.empty or mom_df.empty or last_df.empty:
        st.warning("Data tidak cukup untuk kalkulasi skor (kurang dari 30 hari).")
        return pd.DataFrame()

    # === TR E N D (30 hari)
    tr = trend_df.groupby('Stock Code').agg(
        last_price=('Close','last'),
        last_final_signal=('Final Signal','last'),
        total_net_ff=('Net Foreign Flow','sum'),
        total_money_flow=('Money Flow Value','sum'),
        avg_change_pct=('Change %','mean'),
        sector=('Sector','last')
    ).reset_index()

    score_akum = tr['last_final_signal'].map({'Strong Akumulasi':100,'Akumulasi':75,'Netral':30,'Distribusi':10,'Strong Distribusi':0}).fillna(30)
    score_ff   = pct_rank(tr['total_net_ff'])
    score_mfv  = pct_rank(tr['total_money_flow'])
    score_mom  = pct_rank(tr['avg_change_pct'])
    tr['Trend Score'] = (score_akum*W['trend_akum'] + score_ff*W['trend_ff'] +
                         score_mfv*W['trend_mfv'] + score_mom*W['trend_mom'])

    # === M O M E N T U M (7 hari)
    mo = mom_df.groupby('Stock Code').agg(
        total_change_pct=('Change %','sum'),
        had_unusual_volume=('Unusual Volume','any'),
        last_final_signal=('Final Signal','last'),
        total_net_ff=('Net Foreign Flow','sum')
    ).reset_index()

    s_price = pct_rank(mo['total_change_pct'])
    s_vol   = mo['had_unusual_volume'].map({True:100, False:20}).fillna(20)
    s_akum  = mo['last_final_signal'].map({'Strong Akumulasi':100,'Akumulasi':80,'Netral':40,'Distribusi':10,'Strong Distribusi':0}).fillna(40)
    s_ff7   = pct_rank(mo['total_net_ff'])
    mo['Momentum Score'] = (s_price*W['mom_price'] + s_vol*W['mom_vol'] +
                            s_akum*W['mom_akum'] + s_ff7*W['mom_ff'])

    # === NBSA & Foreign Contribution (30 hari)
    nbsa = trend_df.groupby('Stock Code').agg(total_net_ff_30d=('Net Foreign Flow','sum')).reset_index()
    
    if {'Foreign Buy','Foreign Sell','Value'}.issubset(df_all_history.columns):
        tmp = trend_df.copy()
        tmp['Foreign Value'] = tmp['Foreign Buy'].fillna(0) + tmp['Foreign Sell'].fillna(0)
        contrib = tmp.groupby('Stock Code').agg(
            total_foreign_value_30d=('Foreign Value','sum'),
            total_value_30d=('Value','sum')
        ).reset_index()
        contrib['foreign_contrib_pct'] = (contrib['total_foreign_value_30d'] / (contrib['total_value_30d'] + 1))*100
    else:
        contrib = pd.DataFrame({'Stock Code':[], 'foreign_contrib_pct':[]})

    # === Unusual bonus (hari terakhir)
    uv = last_df.set_index('Stock Code')['Unusual Volume'].map({True:1, False:0})

    # === GABUNG skor → Potential Score
    rank = tr[['Stock Code','Trend Score','last_price','last_final_signal','sector']].merge(
        mo[['Stock Code','Momentum Score']], on='Stock Code', how='outer'
    ).merge(
        nbsa, on='Stock Code', how='left'
    ).merge(
        contrib[['Stock Code','foreign_contrib_pct']] if not contrib.empty else pd.DataFrame({'Stock Code':[],'foreign_contrib_pct':[]}),
        on='Stock Code', how='left'
    )

    rank['NBSA Score']            = to_pct(rank['total_net_ff_30d'])
    rank['Foreign Contrib Score'] = to_pct(rank['foreign_contrib_pct'])
    unusual_bonus = uv.reindex(rank['Stock Code']).fillna(0)*5 # skala 0/5

    rank['Potential Score'] = (
        rank['Trend Score'].fillna(0)*W['blend_trend'] +
        rank['Momentum Score'].fillna(0)*W['blend_mom'] +
        rank['NBSA Score'].fillna(50)*W['blend_nbsa'] +
        rank['Foreign Contrib Score'].fillna(50)*W['blend_fcontrib'] +
        unusual_bonus.values*W['blend_unusual']
    )

    # === TOP 20 & format
    rank.insert(0, 'Analysis Date', latest_date.strftime('%Y-%m-%d'))
    
    # Susun kolom untuk Sheet
    cols = ['Analysis Date','Stock Code','Potential Score','Trend Score','Momentum Score',
            'total_net_ff_30d','foreign_contrib_pct','last_price','last_final_signal','sector']
    for c in cols:
        if c not in rank.columns: rank[c] = np.nan
    rank = rank[cols]
    
    print("Skor potensial selesai dihitung.")
    return rank.sort_values('Potential Score', ascending=False)


# ==============================================================================
# 🏁 6) LOAD DATA UTAMA
# ==============================================================================

df = load_data()

if df.empty:
    st.warning("⚠️ Data belum berhasil dimuat. Aplikasi tidak dapat dilanjutkan.")
    st.stop()

# ==============================================================================
#  sidebar 7) SIDEBAR FILTER
# ==============================================================================
st.sidebar.header("🎛️ Filter Analisis Harian")

# Tombol Refresh Data
if st.sidebar.button("🔄 Refresh Data (Paksa Download Ulang)"):
    st.cache_data.clear() # Hapus cache
    st.experimental_rerun() # Jalankan ulang skrip

max_date = df['Last Trading Date'].max().date()
selected_date = st.sidebar.date_input(
    "Pilih Tanggal Analisis",
    max_date,
    min_value=df['Last Trading Date'].min().date(),
    max_value=max_date,
    format="DD-MM-YYYY"
)

# Filter data berdasarkan tanggal terpilih
df_day = df[df['Last Trading Date'].dt.date == selected_date].copy()

st.sidebar.header("Filter Data Lanjutan")
selected_stocks = st.sidebar.multiselect(
    "Pilih Saham (Stock Code)",
    options=sorted(df_day["Stock Code"].dropna().unique()),
    placeholder="Ketik kode saham"
)

selected_sectors = st.sidebar.multiseiytdf.columnslect(
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
    max_value=float(df_day["Volume Spike (x)"].max() if not df_day.empty else 50.0),
    value=2.0,
    step=0.5
)

show_only_spike = st.sidebar.checkbox(
    "Hanya tampilkan Unusual Volume (True)",
    value=True
)

# --- Terapkan Filter ---
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

# ==============================================================================
#  layout 8) LAYOUT UTAMA (DENGAN TABS)
# ==============================================================================
st.caption(f"Menampilkan data untuk tanggal: **{selected_date.strftime('%d %B %Y')}**")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 **Dashboard Harian**",
    "📈 **Analisis Individual**",
    "📋 **Data Filter**",
    "🏆 **Saham Potensial (TOP 20)**" # Tab Baru
])

# --- TAB 1: DASHBOARD HARIAN ---
with tab1:
    st.subheader("Ringkasan Pasar (pada tanggal terpilih)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Saham Aktif", f"{len(df_day['Stock Code'].unique()):,.0f}")
    col2.metric("Saham Unusual Volume", f"{int(df_day['Unusual Volume'].sum()):,.0f}")
    col3.metric("Total Nilai Transaksi", f"Rp {df_day['Value'].sum():,.0f}")

    st.markdown("---")
    st.subheader("Top Movers & Most Active")
    
    col_g, col_l, col_v = st.columns(3)
    
    with col_g:
        st.markdown("**Top 10 Gainers**")
        top_gainers = df_day.sort_values("Change %", ascending=False).head(10)
        st.dataframe(
            top_gainers[['Stock Code', 'Close', 'Change %']], 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Close": st.column_config.NumberColumn("Close", format="Rp %d"),
                "Change %": st.column_config.NumberColumn("Change %", format="%.2f")
            }
        )

    with col_l:
        st.markdown("**Top 10 Losers**")
        top_losers = df_day.sort_values("Change %", ascending=True).head(10)
        st.dataframe(
            top_losers[['Stock Code', 'Close', 'Change %']], 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Close": st.column_config.NumberColumn("Close", format="Rp %d"),
                "Change %": st.column_config.NumberColumn("Change %", format="%.2f")
            }
        )
        
    with col_v:
        st.markdown("**Top 10 by Value**")
        top_value = df_day.sort_values("Value", ascending=False).head(10)
        st.dataframe(
            top_value[['Stock Code', 'Close', 'Value']], 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Close": st.column_config.NumberColumn("Close", format="Rp %d"),
                "Value": st.column_config.NumberColumn("Value", format="Rp %d")
            }
        )

    st.markdown("---")
    st.subheader("Distribusi Sektor & Signal")
    
    col_sig, col_sec = st.columns(2)
    
    with col_sig:
        st.markdown("**Distribusi Final Signal (Semua Saham)**")
        if not df_day.empty:
            signal_counts = df_day["Final Signal"].value_counts().reset_index()
            fig_sig = px.bar(
                signal_counts, 
                x="Final Signal", 
                y="count", 
                title="Distribusi Final Signal",
                text='count'
            )
            fig_sig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Jumlah: %{y:,.0f}<extra></extra>')
            # Perbaikan Bug: Biarkan Plotly mengatur label sumbu Y secara default
            fig_sig.update_layout(yaxis_title="Jumlah Saham")
            st.plotly_chart(fig_sig, use_container_width=True)
        else:
            st.info("Tidak ada data signal untuk tanggal ini.")

    with col_sec:
        st.markdown("**Sektor dengan Unusual Volume Terbanyak**")
        spike_df = df_day[df_day['Unusual Volume'] == True]
        if not spike_df.empty:
            sector_counts = spike_df["Sector"].value_counts().reset_index()
            fig_sec = px.bar(
                sector_counts, 
                x="Sector", 
                y="count", 
                title="Distribusi Sektor (Hanya Unusual Volume)",
                text='count'
            )
            fig_sec.update_traces(texttemplate='%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Jumlah: %{y:,.0f}<extra></extra>')
            # Perbaikan Bug: Biarkan Plotly mengatur label sumbu Y secara default
            fig_sec.update_layout(yaxis_title="Jumlah Saham")
            st.plotly_chart(fig_sec, use_container_width=True)
        else:
            st.info("Tidak ada saham dengan 'Unusual Volume' pada tanggal ini.")


# --- TAB 2: ANALISIS INDIVIDUAL ---
with tab2:
    st.subheader("Analisis Time Series Saham Individual")
    
    all_stocks = sorted(df["Stock Code"].dropna().unique())
    stock_to_analyze = st.selectbox(
        "Pilih Saham untuk dianalisis:",
        all_stocks,
        index=all_stocks.index("AADI") if "AADI" in all_stocks else 0
    )
    
    if stock_to_analyze:
        df_stock = df[df['Stock Code'] == stock_to_analyze].sort_values('Last Trading Date')
        
        if df_stock.empty:
            st.warning(f"Tidak ditemukan data historis untuk {stock_to_analyze}")
        else:
            st.info(f"Menampilkan data untuk: **{df_stock.iloc[0]['Company Name']} ({stock_to_analyze})**")
            
            # --- Chart Gabungan (Subplot) ---
            fig_combined = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3], # Plot harga 70%, volume 30%
                specs=[[{"secondary_y": True}], # Baris 1 punya 2 sumbu Y
                       [{"secondary_y": False}]] # Baris 2 punya 1 sumbu Y
            )

            # --- Plot 1 Atas: Harga (Kanan) vs NFF (Kiri) ---
            
            # Y1 - Kiri: Net Foreign Flow (Batang)
            fig_combined.add_trace(go.Bar(
                x=df_stock['Last Trading Date'],
                y=df_stock['Net Foreign Flow'],
                name="Net Foreign Flow (Shares)",
                marker_color='orange',
                opacity=0.6
            ), row=1, col=1, secondary_y=False) # secondary_y=False -> Kiri
            
            # Y2 - Kanan: Harga Close (Garis)
            fig_combined.add_trace(go.Scatter(
                x=df_stock['Last Trading Date'],
                y=df_stock['Close'],
                name="Harga Close (Rp)",
                line=dict(color='blue')
            ), row=1, col=1, secondary_y=True) # secondary_y=True -> Kanan

            # --- Plot 2 Bawah: Volume ---
            
            # Volume (Batang)
            fig_combined.add_trace(go.Bar(
                x=df_stock['Last Trading Date'],
                y=df_stock['Volume'],
                name="Volume",
                marker_color='grey',
                opacity=0.5
            ), row=2, col=1)
            
            # MA20 Volume (Garis)
            fig_combined.add_trace(go.Scatter(
                x=df_stock['Last Trading Date'],
                y=df_stock['MA20_vol'],
                name="MA20 Volume",
                line=dict(color='red', dash='dot')
            ), row=2, col=1)

            # --- Styling ---
            fig_combined.update_layout(
                title=f"Analisis Harga, Foreign Flow, dan Volume untuk {stock_to_analyze}",
                height=600,
                xaxis_rangeslider_visible=False, # Sembunyikan slider di plot atas
                xaxis2_rangeslider_visible=True, # Tampilkan slider di plot bawah
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Perbaikan Bug: Biarkan Plotly mengatur label sumbu Y secara default
            fig_combined.update_yaxes(title_text="Net Foreign Flow (Shares)", row=1, col=1, secondary_y=False)
            fig_combined.update_yaxes(title_text="Harga Close (Rp)", row=1, col=1, secondary_y=True)
            fig_combined.update_yaxes(title_text="Volume (Shares)", row=2, col=1, secondary_y=False)

            st.plotly_chart(fig_combined, use_container_width=True)


# --- TAB 3: DATA FILTER ---
with tab3:
    st.subheader(f"Data Filter: {len(df_filtered)} saham ditemukan")
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Close": st.column_config.NumberColumn("Close", format="Rp %d"),
            "Change %": st.column_config.NumberColumn("Change %", format="%.2f"),
            "Value": st.column_config.NumberColumn("Value", format="Rp %d"),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
            "Net Foreign Flow": st.column_config.NumberColumn("Net Foreign Flow", format="%d"),
            "Last Trading Date": st.column_config.DateColumn("Last Trading Date", format="DD-MMM-YYYY")
        }
    )

# --- TAB 4: SAHAM POTENSIAL (TOP 20) ---
with tab4:
    st.subheader("🏆 Top 20 Saham Potensial (Analisis 30 Hari Terakhir)")
    st.caption(f"Skor dihitung berdasarkan data historis 30 hari terakhir dari tanggal data terbaru ({max_date.strftime('%d %B %Y')}).")
    st.info("Kalkulasi skor ini (Trend, Momentum, NBSA) didasarkan pada logika skrip Colab Anda.")
    
    with st.spinner("Menghitung skor potensial... Ini mungkin perlu waktu beberapa detik..."):
        df_potential = calculate_potential_score(df)
    
    if df_potential.empty:
        st.error("Gagal menghitung skor potensial. Data mungkin tidak cukup.")
    else:
        st.dataframe(
            df_potential.head(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Analysis Date": st.column_config.DateColumn("Analysis Date", format="DD-MMM-YYYY"),
                "Potential Score": st.column_config.NumberColumn("Potential Score", format="%.2f", help="Skor gabungan (Trend, Momentum, NBSA, dll)"),
                "Trend Score": st.column_config.NumberColumn("Trend Score", format="%.2f", help="Skor 30 hari (Akumulasi, NFF, MFV)"),
                "Momentum Score": st.column_config.NumberColumn("Momentum Score", format="%.2f", help="Skor 7 hari (Harga, Vol, Akumulasi)"),
                "total_net_ff_30d": st.column_config.NumberColumn("NFF 30D (Shares)", format="%d"),
                "foreign_contrib_pct": st.column_config.NumberColumn("Foreign Contrib %", format="%.2f%%"),
                "last_price": st.column_config.NumberColumn("Last Price", format="Rp %d"),
                "last_final_signal": "Signal Terakhir"
            }
        )

