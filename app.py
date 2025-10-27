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
    page_title="📊 Dashboard Analisis Code IDX",
    layout="wide",
    page_icon="📈"
)

# --- KONFIGURASI G-DRIVE ---
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP"
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

        df = pd.read_csv(fh, dtype=object)

        df.columns = df.columns.str.strip()
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'], errors='coerce')

        # [PERUBAHAN] Tambahkan 'Money Flow Ratio (20D)' ke daftar kolom numerik
        cols_to_numeric = [
            'High', 'Low', 'Close', 'Volume', 'Value', 'Foreign Buy', 'Foreign Sell',
            'Bid Volume', 'Offer Volume', 'Previous', 'Change', 'Open Price', 'First Trade',
            'Frequency', 'Index Individual', 'Offer', 'Bid', 'Listed Shares', 'Tradeble Shares',
            'Weight For Index', 'Non Regular Volume', 'Change %', 'Typical Price', 'TPxV',
            'VWMA_20D', 'MA20_vol', 'MA5_vol', 'Volume Spike (x)', 'Net Foreign Flow',
            'Bid/Offer Imbalance', 'Money Flow Value', 'Free Float', 'Money Flow Ratio (20D)' # <-- Tambahkan di sini
        ]

        for col in cols_to_numeric:
            if col in df.columns:
                cleaned_col = df[col].astype(str).str.strip()
                cleaned_col = cleaned_col.str.replace(r'[,\sRp\%]', '', regex=True)
                df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0)

        if 'Unusual Volume' in df.columns:
            df['Unusual Volume'] = df['Unusual Volume'].astype(str).str.strip().str.lower().isin(['spike volume signifikan', 'true', 'True', 'TRUE'])
            df['Unusual Volume'] = df['Unusual Volume'].astype(bool)

        if 'Final Signal' in df.columns:
            df['Final Signal'] = df['Final Signal'].astype(str).str.strip()

        if 'Sector' in df.columns:
             df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
             df['Sector'] = 'Others'

        df = df.dropna(subset=['Last Trading Date', 'Stock Code'])

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
        tmp['Foreign Value proxy'] = tmp['NFF (Rp)']
        contrib = tmp.groupby('Stock Code').agg(
            total_foreign_value_proxy=('Foreign Value proxy', 'sum'),
            total_value_30d=('Value', 'sum')
        ).reset_index()
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
    latest_prices = latest_data.get('Close', pd.Series(dtype='float64'))
    latest_sectors = latest_data.get('Sector', pd.Series(dtype='object'))

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

    # Pastikan semua hasil ada sebelum return
    return results.get('7D', pd.DataFrame()), results.get('30D', pd.DataFrame()), \
           results.get('90D', pd.DataFrame()), results.get('180D', pd.DataFrame())


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
        mfv_agg = df_period.groupby('Stock Code')['Money Flow Value'].sum()
        df_agg = pd.DataFrame(mfv_agg)
        df_agg.columns = ['Total Money Flow (Rp)']
        df_agg = df_agg.join(latest_prices).join(latest_sectors)
        df_agg.rename(columns={'Close': 'Harga Terakhir'}, inplace=True)
        df_agg = df_agg.sort_values(by='Total Money Flow (Rp)', ascending=False)
        results[name] = df_agg.reset_index()

    return results.get('7D', pd.DataFrame()), results.get('30D', pd.DataFrame()), \
           results.get('90D', pd.DataFrame()), results.get('180D', pd.DataFrame())


# ==============================================================================
# 💎 5) LAYOUT UTAMA (HEADER)
# ==============================================================================
st.title("📈 Dashboard Analisis Code IDX")
st.caption("Menganalisis data historis harian untuk menemukan Code potensial.")

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
selected_stocks_filter = st.sidebar.multiselect(
    "Pilih Code (Stock Code)",
    options=sorted(df_day["Stock Code"].dropna().unique()),
    placeholder="Ketik kode Code",
    key="filter_stock_multiselect"
)
selected_sectors_filter = st.sidebar.multiselect(
    "Pilih Sektor",
    options=sorted(df_day.get("Sector", pd.Series(dtype='object')).dropna().unique()), # Handle jika Sector tdk ada
    placeholder="Pilih sektor",
    key="filter_sector_multiselect"
)
selected_signals_filter = st.sidebar.multiselect(
    "Filter Berdasarkan Final Signal",
    options=sorted(df_day.get("Final Signal", pd.Series(dtype='object')).dropna().unique()), # Handle jika Final Signal tdk ada
    placeholder="Pilih signal",
    key="filter_signal_multiselect"
)
# Pastikan Volume Spike (x) ada dan valid sebelum mencari max
max_spike_val = 50.0 # Default max value
if "Volume Spike (x)" in df_day.columns and not df_day.empty:
     max_val_candidate = df_day["Volume Spike (x)"].max()
     if pd.notna(max_val_candidate):
          max_spike_val = float(max_val_candidate)

min_spike_filter = st.sidebar.slider(
    "Minimal Volume Spike (x)",
    min_value=1.0,
    max_value=max_spike_val,
    value=1.0,
    step=0.5,
    key="filter_spike_slider"
)
show_only_spike_filter = st.sidebar.checkbox(
    "Hanya tampilkan Unusual Volume (True)",
    value=False,
    key="filter_spike_checkbox"
)

# Terapkan Filter (untuk Tab 3)
df_filtered = df_day.copy()
if selected_stocks_filter:
    df_filtered = df_filtered[df_filtered["Stock Code"].isin(selected_stocks_filter)]
if selected_sectors_filter and 'Sector' in df_filtered.columns: # Cek kolom Sector
    df_filtered = df_filtered[df_filtered["Sector"].isin(selected_sectors_filter)]
if selected_signals_filter and 'Final Signal' in df_filtered.columns: # Cek kolom Final Signal
    df_filtered = df_filtered[df_filtered["Final Signal"].isin(selected_signals_filter)]
if min_spike_filter > 1.0 and 'Volume Spike (x)' in df_filtered.columns: # Cek kolom Volume Spike
    df_filtered = df_filtered[df_filtered["Volume Spike (x)"] >= min_spike_filter]
if show_only_spike_filter and 'Unusual Volume' in df_filtered.columns: # Cek kolom Unusual Volume
    df_filtered = df_filtered[df_filtered["Unusual Volume"] == True]

# ==============================================================================
#  LAYOUT UTAMA (DENGAN TABS BARU)
# ==============================================================================
st.caption(f"Menampilkan data untuk tanggal: **{selected_date.strftime('%d %B %Y')}**")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 **Dashboard Harian**",
    "📈 **Analisis Individual**",
    "📋 **Data Filter**",
    "🏆 **Code Potensial (TOP 20)**",
    "🌊 **Analisis NFF (Rp)**",
    "💰 **Analisis Money Flow (Rp)**"
])

# --- TAB 1: DASHBOARD HARIAN ---
with tab1:
    st.subheader("Ringkasan Pasar (pada tanggal terpilih)")
    if df_day.empty:
        st.warning(f"Tidak ada data transaksi untuk tanggal {selected_date.strftime('%d-%m-%Y')}.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Code Aktif", f"{len(df_day['Stock Code'].unique()):,.0f}")
        unusual_vol_count = int(df_day['Unusual Volume'].sum()) if 'Unusual Volume' in df_day.columns else 0
        col2.metric("Code Unusual Volume", f"{unusual_vol_count:,}")
        total_value_today = df_day['Value'].sum() if 'Value' in df_day.columns else 0
        col3.metric("Total Nilai Transaksi", f"Rp {total_value_today:,.0f}" if pd.notna(total_value_today) else "Rp 0")
        st.markdown("---")
        st.subheader("Top Movers & Most Active")
        col_g, col_l, col_v = st.columns(3)

        def format_movers_df(df_in, value_col=None):
            cols_needed = ['Stock Code', 'Close', 'Change %']
            if value_col: cols_needed.append(value_col)
            # Cek jika semua kolom ada
            if not all(c in df_in.columns for c in cols_needed):
                return pd.DataFrame() # Kembalikan df kosong jika ada kolom hilang
                
            df_out = df_in[cols_needed].copy()
            df_display = df_out.copy()
            df_display['Close'] = df_display['Close'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
            if value_col:
                df_display[value_col] = df_display[value_col].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
            return df_display

        with col_g:
            st.markdown("**Top 10 Gainers (%)**")
            if 'Change %' in df_day.columns:
                top_gainers = df_day.sort_values("Change %", ascending=False).head(10)
                st.dataframe(format_movers_df(top_gainers), use_container_width=True, hide_index=True,
                             column_config={"Stock Code": "Code","Close": "Harga","Change %": st.column_config.NumberColumn("Change %", format="%.2f")})
            else: st.warning("Kolom 'Change %' tidak ditemukan.")

        with col_l:
            st.markdown("**Top 10 Losers (%)**")
            if 'Change %' in df_day.columns:
                top_losers = df_day.sort_values("Change %", ascending=True).head(10)
                st.dataframe(format_movers_df(top_losers), use_container_width=True, hide_index=True,
                             column_config={"Stock Code": "Code","Close": "Harga","Change %": st.column_config.NumberColumn("Change %", format="%.2f")})
            else: st.warning("Kolom 'Change %' tidak ditemukan.")

        with col_v:
            st.markdown("**Top 10 by Value**")
            if 'Value' in df_day.columns:
                top_value = df_day.sort_values("Value", ascending=False).head(10)
                st.dataframe(format_movers_df(top_value, value_col='Value'), use_container_width=True, hide_index=True,
                             column_config={"Stock Code": "Code","Close": "Harga","Value": "Nilai"})
            else: st.warning("Kolom 'Value' tidak ditemukan.")

        st.markdown("---")
        st.subheader("Distribusi Sektor & Signal")
        col_sig, col_sec = st.columns(2)
        with col_sig:
            st.markdown("**Distribusi Final Signal**")
            if not df_day.empty and 'Final Signal' in df_day.columns:
                signal_counts = df_day["Final Signal"].value_counts().reset_index()
                fig_sig = px.bar(signal_counts, x="Final Signal", y="count", text='count')
                fig_sig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Jumlah: %{y:,.0f}<extra></extra>')
                fig_sig.update_layout(yaxis_title="Jumlah Code", yaxis=dict(showticklabels=True))
                st.plotly_chart(fig_sig, use_container_width=True)
            else: st.warning("Kolom 'Final Signal' tidak ditemukan.")
        with col_sec:
            st.markdown("**Sektor Unusual Volume**")
            if 'Sector' in df_day.columns and 'Unusual Volume' in df_day.columns:
                spike_df = df_day[df_day['Unusual Volume'] == True]
                if not spike_df.empty:
                    sector_counts = spike_df["Sector"].value_counts().reset_index()
                    fig_sec = px.bar(sector_counts, x="Sector", y="count", text='count')
                    fig_sec.update_traces(texttemplate='%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Jumlah: %{y:,.0f}<extra></extra>')
                    fig_sec.update_layout(yaxis_title="Jumlah Code", yaxis=dict(showticklabels=True))
                    st.plotly_chart(fig_sec, use_container_width=True)
                else: st.info("Tidak ada unusual volume hari ini.")
            else: st.warning("Kolom 'Sector'/'Unusual Volume' tidak ditemukan.")

# --- TAB 2: ANALISIS INDIVIDUAL ---
with tab2:
    st.subheader("Analisis Time Series Code Individual")
    all_stocks = sorted(df["Stock Code"].dropna().unique())
    stock_to_analyze = st.selectbox("Pilih Code:", all_stocks, index=all_stocks.index("AADI") if "AADI" in all_stocks else 0, key="individual_stock_select")

    if stock_to_analyze:
        df_stock = df[df['Stock Code'] == stock_to_analyze].sort_values('Last Trading Date')
        if df_stock.empty:
            st.warning(f"Tidak ada data historis {stock_to_analyze}")
        else:
            latest_row = df_stock.iloc[-1]
            latest_price = latest_row.get('Close', np.nan)
            free_float = latest_row.get('Free Float', np.nan)
            stock_sector = latest_row.get('Sector', 'N/A')

            st.markdown(f"**Analisis: {stock_to_analyze} ({stock_sector})**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Harga Terakhir", f"Rp {latest_price:,.0f}" if pd.notna(latest_price) else "N/A")
            col2.metric("Free Float Code", f"{free_float:.2f}%" if pd.notna(free_float) else "N/A")
            col3.metric("Sektor", stock_sector if pd.notna(stock_sector) else "N/A")
            st.markdown("---")

            # [PERUBAHAN] Subplot jadi 4 baris
            fig_combined = make_subplots(
                rows=4, # Empat baris
                cols=1, shared_xaxes=True, vertical_spacing=0.03,
                row_heights=[0.4, 0.2, 0.2, 0.2], # Atur proporsi
                specs=[ [{"secondary_y": True}], # Harga vs NFF
                        [{"secondary_y": False}],# MFV
                        [{"secondary_y": False}],# Volume
                        [{"secondary_y": False}] # Money Flow Ratio
                      ]
            )

            # --- Plot 1: Harga vs NFF ---
            nff_colors = np.where(df_stock['NFF (Rp)'] >= 0, 'green', 'red')
            fig_combined.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['NFF (Rp)'], name='NFF (Rp)', marker_color=nff_colors, hovertemplate='Tgl: %{x|%d%b%y}<br>NFF: %{y:,.0f}<extra></extra>'), row=1, col=1, secondary_y=False)
            fig_combined.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['Close'], name='Harga (Rp)', line=dict(color='blue'), hovertemplate='Tgl: %{x|%d%b%y}<br>Harga: %{y:,.0f}<extra></extra>'), row=1, col=1, secondary_y=True)

            # --- Plot 2: Money Flow Value ---
            mfv_colors = np.where(df_stock['Money Flow Value'] >= 0, 'lightgreen', 'lightcoral')
            fig_combined.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['Money Flow Value'], name='MFV (Rp)', marker_color=mfv_colors, hovertemplate='Tgl: %{x|%d%b%y}<br>MFV: %{y:,.0f}<extra></extra>'), row=2, col=1, secondary_y=False)

            # --- Plot 3: Volume ---
            fig_combined.add_trace(go.Bar(x=df_stock['Last Trading Date'], y=df_stock['Volume'], name='Volume', marker_color='gray', hovertemplate='Tgl: %{x|%d%b%y}<br>Vol: %{y:,.0f}<extra></extra>'), row=3, col=1, secondary_y=False)
            fig_combined.add_trace(go.Scatter(x=df_stock['Last Trading Date'], y=df_stock['MA20_vol'], name='MA20 Vol', line=dict(color='orange', dash='dot'), hovertemplate='Tgl: %{x|%d%b%y}<br>MA20: %{y:,.0f}<extra></extra>'), row=3, col=1, secondary_y=False)

            # --- [BARU] Plot 4: Money Flow Ratio ---
            if 'Money Flow Ratio (20D)' in df_stock.columns:
                fig_combined.add_trace(go.Scatter(
                    x=df_stock['Last Trading Date'],
                    y=df_stock['Money Flow Ratio (20D)'],
                    name='MF Ratio (20D)',
                    line=dict(color='purple'),
                    hovertemplate='Tgl: %{x|%d%b%y}<br>MF Ratio: %{y:.3f}<extra></extra>'
                ), row=4, col=1, secondary_y=False)
                # Tambah garis horizontal di 0
                fig_combined.add_hline(y=0, line_dash="dash", line_color="black", row=4, col=1)
            # ----------------------------------------

            # Layout Update
            fig_combined.update_layout(
                title_text=f"Analisis Harga, Flow, Volume & MF Ratio: {stock_to_analyze}",
                height=800, # Perbesar lagi
                xaxis_rangeslider_visible=False,
                xaxis4_rangeslider_visible=True, # Slider di plot paling bawah
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
            )
            # Update label sumbu Y
            fig_combined.update_yaxes(title_text="NFF (Rp)", row=1, col=1, secondary_y=False, showticklabels=True)
            fig_combined.update_yaxes(title_text="Harga (Rp)", row=1, col=1, secondary_y=True, showticklabels=True)
            fig_combined.update_yaxes(title_text="MFV (Rp)", row=2, col=1, secondary_y=False, showticklabels=True)
            fig_combined.update_yaxes(title_text="Volume", row=3, col=1, secondary_y=False, showticklabels=True)
            # [BARU] Label Y untuk MF Ratio
            fig_combined.update_yaxes(title_text="MF Ratio (20D)", row=4, col=1, secondary_y=False, showticklabels=True, tickformat=".2f") # Format 2 desimal

            st.plotly_chart(fig_combined, use_container_width=True)

# --- TAB 3: DATA FILTER ---
with tab3:
    st.subheader(f"Data Filter (Total: {len(df_filtered)} baris)")
    st.info("Gunakan filter di sidebar kiri untuk menyaring data pada tanggal terpilih.")
    # [PERUBAHAN] Tambah Money Flow Ratio ke tampilan tabel
    cols_to_display = [
        "Stock Code", "Close", "Change %", "Value", "Net Foreign Flow", "NFF (Rp)",
        "Money Flow Value", "Money Flow Ratio (20D)", # <-- Tambah di sini
        "Volume Spike (x)", "Unusual Volume", "Final Signal", "Sector", "Free Float"
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
            "Stock Code": st.column_config.TextColumn("Code"),
            "Close": st.column_config.TextColumn("Harga"),
            "Change %": st.column_config.NumberColumn("Change %", format="%.2f"),
            "Value": st.column_config.TextColumn("Nilai"),
            "Net Foreign Flow": st.column_config.TextColumn("Net FF (Sh)"),
            "NFF (Rp)": st.column_config.TextColumn("Net FF (Rp)"),
            "Money Flow Value": st.column_config.TextColumn("MFV (Rp)"),
            "Money Flow Ratio (20D)": st.column_config.NumberColumn("MF Ratio", format="%.3f"), # <-- Tambah config
            "Volume Spike (x)": st.column_config.NumberColumn("Spike (x)", format="%.1fx"),
            "Free Float": st.column_config.NumberColumn("FF %", format="%.2f%%"),
            "Unusual Volume": st.column_config.CheckboxColumn("Unusual Vol"),
            "Final Signal": st.column_config.TextColumn("Signal"),
            "Sector": st.column_config.TextColumn("Sektor"),
        }
    )

# --- TAB 4: Code POTENSIAL (TOP 20) ---
with tab4:
    st.subheader("🏆 Top 20 Code Paling Potensial (Overall)")
    st.info(f"Kalkulasi skor dari data 30 hari terakhir ({max_date.strftime('%d %B %Y')}).")
    df_top20, score_msg, score_status = calculate_potential_score(df, pd.Timestamp(max_date))
    if score_status == "success": st.toast(score_msg, icon="🏆")
    elif score_status == "warning": st.warning(score_msg)
    if not df_top20.empty:
        df_top20_display = df_top20.copy()
        format_cols_rp_top20 = ['total_net_ff_30d_rp', 'last_price']
        for col in format_cols_rp_top20:
            if col in df_top20_display.columns:
                df_top20_display[col] = df_top20_display[col].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
        st.dataframe(df_top20_display, use_container_width=True, hide_index=True,
                     column_config={ "Stock Code": "Code", "Potential Score": st.column_config.NumberColumn("Skor", format="%.2f"),
                                     "Trend Score": st.column_config.NumberColumn("Skor Trend", format="%.2f"),
                                     "Momentum Score": st.column_config.NumberColumn("Skor Mom", format="%.2f"),
                                     "total_net_ff_30d_rp": "NFF 30h (Rp)", "foreign_contrib_pct": st.column_config.NumberColumn("Kontr. Asing %", format="%.1f%%"),
                                     "last_price": "Harga", "last_final_signal": "Signal", "sector": "Sektor" })
    else: st.warning("Gagal hitung skor.")

# --- TAB 5: ANALISIS NFF (Rp) ---
with tab5:
    st.subheader("🌊 Top Akumulasi Net Foreign Flow (NFF) dalam Rupiah")
    st.info(f"Akumulasi dari {max_date.strftime('%d %B %Y')} ke belakang.")
    try:
        df_nff_7d, df_nff_30d, df_nff_90d, df_nff_180d = calculate_nff_top_stocks(df, pd.Timestamp(max_date))
        def format_flow_agg_df(df_in, flow_col_name):
             df_display = df_in.head(20).copy()
             if flow_col_name in df_display.columns: df_display[flow_col_name] = df_display[flow_col_name].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
             if 'Harga Terakhir' in df_display.columns: df_display['Harga Terakhir'] = df_display['Harga Terakhir'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
             return df_display
        nff_col_config = { "Stock Code": "Code", "Total Net FF (Rp)": "Total NFF (Rp)", "Harga Terakhir": "Harga", "Sector": "Sektor" }
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**1 Minggu (7 Hari)**"); st.dataframe(format_flow_agg_df(df_nff_7d, 'Total Net FF (Rp)'), use_container_width=True, hide_index=True, column_config=nff_col_config)
            st.markdown("**3 Bulan (90 Hari)**"); st.dataframe(format_flow_agg_df(df_nff_90d, 'Total Net FF (Rp)'), use_container_width=True, hide_index=True, column_config=nff_col_config)
        with col2:
            st.markdown("**1 Bulan (30 Hari)**"); st.dataframe(format_flow_agg_df(df_nff_30d, 'Total Net FF (Rp)'), use_container_width=True, hide_index=True, column_config=nff_col_config)
            st.markdown("**6 Bulan (180 Hari)**"); st.dataframe(format_flow_agg_df(df_nff_180d, 'Total Net FF (Rp)'), use_container_width=True, hide_index=True, column_config=nff_col_config)
    except Exception as e: st.error(f"Gagal hitung NFF: {e}")

# --- TAB 6: ANALISIS Money Flow (Rp) ---
with tab6:
    st.subheader("💰 Top Akumulasi Money Flow Value (MFV) & Rasio Konsistensi")
    st.info(f"MFV (Nilai): Proxy inflow/outflow keseluruhan. Rasio (Konsistensi): Seberapa konsisten MFV positif/negatif selama 20 hari.")
    st.info(f"Akumulasi Nilai dihitung dari {max_date.strftime('%d %B %Y')} ke belakang. Rasio adalah nilai TERBARU.")

    # --- Bagian 1: Top MFV (Nilai Akumulasi) ---
    st.markdown("**Top Akumulasi Money Flow Value (MFV)**")
    try:
        df_mfv_7d, df_mfv_30d, df_mfv_90d, df_mfv_180d = calculate_mfv_top_stocks(df, pd.Timestamp(max_date))
        mfv_col_config = { "Stock Code": "Code", "Total Money Flow (Rp)": "Total MFV (Rp)", "Harga Terakhir": "Harga", "Sector": "Sektor" }
        col1_mfv, col2_mfv = st.columns(2)
        with col1_mfv:
            st.markdown("**1 Minggu (7 Hari)**"); st.dataframe(format_flow_agg_df(df_mfv_7d, 'Total Money Flow (Rp)'), use_container_width=True, hide_index=True, column_config=mfv_col_config)
            st.markdown("**3 Bulan (90 Hari)**"); st.dataframe(format_flow_agg_df(df_mfv_90d, 'Total Money Flow (Rp)'), use_container_width=True, hide_index=True, column_config=mfv_col_config)
        with col2_mfv:
            st.markdown("**1 Bulan (30 Hari)**"); st.dataframe(format_flow_agg_df(df_mfv_30d, 'Total Money Flow (Rp)'), use_container_width=True, hide_index=True, column_config=mfv_col_config)
            st.markdown("**6 Bulan (180 Hari)**"); st.dataframe(format_flow_agg_df(df_mfv_180d, 'Total Money Flow (Rp)'), use_container_width=True, hide_index=True, column_config=mfv_col_config)
    except Exception as e: st.error(f"Gagal hitung MFV: {e}")

    st.markdown("---") # Pemisah

    # --- [BARU] Bagian 2: Top Money Flow Ratio (Konsistensi) ---
    st.markdown("**Top Konsistensi Money Flow (Rasio 20 Hari Terakhir)**")
    # Ambil data terbaru (df_day) dan pastikan kolom rasio ada
    if not df_day.empty and 'Money Flow Ratio (20D)' in df_day.columns:
        col_r1, col_r2 = st.columns(2)
        
        # Format tabel rasio
        def format_ratio_df(df_in):
             df_display = df_in[['Stock Code', 'Close', 'Money Flow Ratio (20D)', 'Sector']].head(10).copy()
             df_display['Close'] = df_display['Close'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
             return df_display
             
        ratio_col_config = {
             "Stock Code": st.column_config.TextColumn("Code"),
             "Close": st.column_config.TextColumn("Harga"),
             "Money Flow Ratio (20D)": st.column_config.NumberColumn("Rasio MF (20D)", format="%.3f"), # 3 desimal
             "Sector": st.column_config.TextColumn("Sektor")
        }

        with col_r1:
            st.markdown("**Top 10 Rasio Tertinggi (Konsisten Inflow)**")
            top_ratio_inflow = df_day.sort_values('Money Flow Ratio (20D)', ascending=False)
            st.dataframe(format_ratio_df(top_ratio_inflow), use_container_width=True, hide_index=True, column_config=ratio_col_config)

        with col_r2:
            st.markdown("**Top 10 Rasio Terendah (Konsisten Outflow)**")
            top_ratio_outflow = df_day.sort_values('Money Flow Ratio (20D)', ascending=True)
            st.dataframe(format_ratio_df(top_ratio_outflow), use_container_width=True, hide_index=True, column_config=ratio_col_config)
            
    else:
         st.warning("Kolom 'Money Flow Ratio (20D)' tidak ditemukan atau tidak ada data untuk tanggal terpilih.")

