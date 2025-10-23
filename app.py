import streamlit as st
import pandas as pd
import numpy as np # Ditambahkan untuk kalkulasi skor
import plotly.express as px
import plotly.graph_objects as go # Diperlukan untuk dual-axis chart
from plotly.subplots import make_subplots # Diperlukan untuk dual-axis chart
# gdown sudah tidak diperlukan

# =====================================================================
# ⚙️ KONFIGURASI DASHBOARD
# =====================================================================
st.set_page_config(
    page_title="📊 Dashboard Analisis Saham IDX",
    layout="wide",
    page_icon="📈"
)

st.title("📈 Dashboard Analisis Saham IDX")
st.caption("Menganalisis data historis untuk menemukan saham potensial.")

# =====================================================================
# 🧠 LOGIKA SCORING (dari Skrip Colab)
# =====================================================================
# Bobot skor (bisa diutak-atik)
W = dict(
    trend_akum=0.40, trend_ff=0.30, trend_mfv=0.20, trend_mom=0.10,
    mom_price=0.40,  mom_vol=0.25,  mom_akum=0.25,  mom_ff=0.10,
    blend_trend=0.35, blend_mom=0.35, blend_nbsa=0.20, blend_fcontrib=0.05, blend_unusual=0.05
)

def pct_rank(s: pd.Series):
    """Menghitung percentile rank, di-normalisasi 0-100."""
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, method="average").fillna(0)*100

def to_pct(s: pd.Series):
    """Konversi seri ke persentase 0-100 (min-max scaling)."""
    s = pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan)
    if s.notna().sum() <= 1: return pd.Series(50, index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn==mx: return pd.Series(50, index=s.index)
    return (s-mn)/(mx-mn)*100

def calculate_potential_score(df: pd.DataFrame, latest_date: pd.Timestamp, W: dict):
    """
    Menjalankan logika scoring dari skrip Colab untuk menemukan
    TOP 20 saham potensial berdasarkan data lengkap.
    """
    try:
        # === Window tanggal ===
        trend_start = latest_date - pd.Timedelta(days=30)
        mom_start   = latest_date - pd.Timedelta(days=7)
        analysis_date_str = latest_date.strftime('%Y-%m-%d')

        trend_df = df[df['Last Trading Date'] >= trend_start].copy()
        mom_df   = df[df['Last Trading Date'] >= mom_start].copy()
        last_df  = df[df['Last Trading Date'] == latest_date].copy()

        if trend_df.empty or mom_df.empty or last_df.empty:
            st.error(f"Data tidak cukup untuk analisis TOP 20 pada tanggal {analysis_date_str}.")
            return pd.DataFrame()

        # === TR E N D (30 hari) ===
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

        # === M O M E N T U M (7 hari) ===
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

        # === NBSA & Foreign Contribution (30 hari) ===
        nbsa = trend_df.groupby('Stock Code').agg(total_net_ff_30d=('Net Foreign Flow','sum')).reset_index()
        
        tmp = trend_df.copy()
        tmp['Foreign Value'] = tmp['Foreign Buy'].fillna(0) + tmp['Foreign Sell'].fillna(0)
        contrib = tmp.groupby('Stock Code').agg(
            total_foreign_value_30d=('Foreign Value','sum'),
            total_value_30d=('Value','sum')
        ).reset_index()
        contrib['foreign_contrib_pct'] = (contrib['total_foreign_value_30d'] / (contrib['total_value_30d'] + 1))*100

        # === Unusual bonus (hari terakhir) ===
        uv = last_df.set_index('Stock Code')['Unusual Volume'].map({True:1, False:0})

        # === GABUNG skor → Potential Score ===
        rank = tr[['Stock Code','Trend Score','last_price','last_final_signal','sector']].merge(
            mo[['Stock Code','Momentum Score']], on='Stock Code', how='outer'
        ).merge(
            nbsa, on='Stock Code', how='left'
        ).merge(
            contrib[['Stock Code','foreign_contrib_pct']], on='Stock Code', how='left'
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

        # === TOP 20 & format ===
        top20 = rank.sort_values('Potential Score', ascending=False).head(20).copy()
        top20.insert(0, 'Analysis Date', analysis_date_str)
        
        # Susun kolom untuk Sheet
        cols = ['Analysis Date','Stock Code','Potential Score','Trend Score','Momentum Score',
                'total_net_ff_30d','foreign_contrib_pct','last_price','last_final_signal','sector']
        for c in cols:
            if c not in top20.columns: top20[c] = np.nan
        top20 = top20[cols]
        
        return top20

    except Exception as e:
        st.error(f"Gagal menghitung skor TOP 20: {e}")
        return pd.DataFrame()

# =====================================================================
# 📦 MEMUAT DAN MEMBERSIHKAN DATA (dari Google Drive)
# =====================================================================

# ID File Google Drive Anda
FILE_ID = "1A3eqXBUhzOTOQ1QR72ArEbLhGCTtYQ3L" 
# Buat URL download langsung yang bisa dibaca pandas
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

@st.cache_data(ttl=3600)
def load_data():
    """
    Memuat data dari URL Google Drive, membersihkan, 
    dan mengonversi tipe data.
    """
    try:
        # Langsung baca URL ke pandas, tidak perlu gdown
        df = pd.read_csv(DOWNLOAD_URL)
        
        # 1. Bersihkan nama kolom (hapus spasi)
        df.columns = df.columns.str.strip()
        
        # 2. Ganti nama kolom yang salah (jika ada)
        # Tambahkan pembersihan untuk kolom 'Foreign Buy' dan 'Foreign Sell'
        cols_to_clean_rename = {
            ' Change % ': 'Change %',
            ' Volume Spike (x) ': 'Volume Spike (x)',
            'Foreign Buy': 'Foreign Buy',
            'Foreign Sell': 'Foreign Sell'
        }
        
        # Buat set kolom yang ada
        existing_cols = set(df.columns)
        
        # Rename kolom yang ada
        rename_map = {}
        for old_name, new_name in cols_to_clean_rename.items():
            if old_name in existing_cols:
                rename_map[old_name] = new_name
        
        if rename_map:
             df = df.rename(columns=rename_map)

        # 3. Konversi Tanggal
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'])
        
        # 4. Konversi kolom-kolom penting ke numerik
        cols_to_numeric = [
            'Change %', 'Typical Price', 'TPxV', 'VWMA_20D', 'MA20_vol', 
            'MA5_vol', 'Volume Spike (x)', 'Net Foreign Flow', 
            'Bid/Offer Imbalance', 'Money Flow Value', 'Close', 'Volume', 'Value',
            'Foreign Buy', 'Foreign Sell' # Pastikan ini numerik
        ]
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 5. Bersihkan kolom 'Unusual Volume' (ubah ke boolean)
        if 'Unusual Volume' in df.columns:
            if df['Unusual Volume'].dtype == 'object':
                df['Unusual Volume'] = df['Unusual Volume'].str.strip().str.lower().isin(['spike volume signifikan', 'true'])
            df['Unusual Volume'] = df['Unusual Volume'].astype(bool)
        
        # 6. Hapus baris yang tidak memiliki data penting
        df = df.dropna(subset=['Last Trading Date', 'Stock Code'])
        
        # 7. Pastikan kolom Sektor ada (penting untuk skor)
        if 'Sector' not in df.columns:
            df['Sector'] = 'Others' # Default jika tidak ada
            
        return df
    
    except Exception as e:
        st.error(f"❌ Gagal membaca data dari Google Drive: {e}")
        st.error(f"Pastikan FILE_ID ('{FILE_ID}') sudah benar dan file disetel ke 'Publik' (Siapa saja yang memiliki link).")
        st.error(f"URL yang dicoba: {DOWNLOAD_URL}")
        return pd.DataFrame()

# Muat data
df = load_data()

# Stop aplikasi jika data gagal dimuat
if df.empty:
    st.warning("⚠️ Data belum berhasil dimuat. Aplikasi tidak dapat dilanjutkan.")
    st.stop()

# =====================================================================
# 🧭 FILTER DATA (SIDEBAR)
# =====================================================================
st.sidebar.header("🎛️ Filter Analisis Harian") # Perbaikan error Unicode

# --- Filter Tanggal ---
max_date = df['Last Trading Date'].max() # Ambil sebagai Timestamp, bukan date
selected_date = st.sidebar.date_input(
    "Pilih Tanggal Analisis",
    max_date.date(), # Tampilkan sebagai date
    min_value=df['Last Trading Date'].min().date(),
    max_value=max_date.date(),
    format="DD-MM-YYYY"
)

# Konversi selected_date (date) ke Timestamp untuk perbandingan
selected_timestamp = pd.to_datetime(selected_date)

# Filter dataframe utama berdasarkan tanggal terpilih
df_day = df[df['Last Trading Date'] == selected_timestamp].copy()

# --- Filter Lanjutan ---
st.sidebar.header("Filter Data Lanjutan")
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

min_spike_value = 1.0
if not df_day.empty and "Volume Spike (x)" in df_day.columns and df_day["Volume Spike (x)"].max() > 0:
    min_spike_value = float(df_day["Volume Spike (x)"].max())

min_spike = st.sidebar.slider(
    "Minimal Volume Spike (x)",
    min_value=1.0,
    max_value=max(min_spike_value, 50.0), # Pastikan max_value >= min_value
    value=min(2.0, min_spike_value), # Pastikan value <= max_value
    step=0.5
)

show_only_spike = st.sidebar.checkbox(
    "Hanya tampilkan Unusual Volume (True)",
    value=True
)

# --- Terapkan Filter Lanjutan ---
df_filtered = df_day.copy()
if selected_stocks:
    df_filtered = df_filtered[df_filtered["Stock Code"].isin(selected_stocks)]
if selected_sectors:
    df_filtered = df_filtered[df_filtered["Sector"].isin(selected_sectors)]
if selected_signals:
    df_filtered = df_filtered[df_filtered["Final Signal"].isin(selected_signals)]

if not df_filtered.empty:
    if "Volume Spike (x)" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Volume Spike (x)"] >= min_spike]

if show_only_spike:
    if not df_filtered.empty:
        if "Unusual Volume" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Unusual Volume"] == True]

# =====================================================================
#  LAYOUT UTAMA (DENGAN TABS)
# =====================================================================
st.caption(f"Menampilkan data untuk tanggal: **{selected_timestamp.strftime('%d %B %Y')}**")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 **Dashboard Harian**",
    "📈 **Analisis Individual**",
    "📋 **Data Filter**",
    "🏆 **Saham Potensial (TOP 20)**" # <-- TAB BARU
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
        st.markdown("**Top 10 Gainers (%)**")
        top_gainers = df_day.sort_values("Change %", ascending=False).head(10)
        st.dataframe(
            top_gainers[['Stock Code', 'Close', 'Change %']], 
            use_container_width=True, 
            hide_index=True,
            column_config={ 
                "Close": st.column_config.NumberColumn("Close", format="Rp %,.0f"),
                "Change %": st.column_config.NumberColumn("Change %", format="%.2f") 
            }
        )

    with col_l:
        st.markdown("**Top 10 Losers (%)**")
        top_losers = df_day.sort_values("Change %", ascending=True).head(10)
        st.dataframe(
            top_losers[['Stock Code', 'Close', 'Change %']], 
            use_container_width=True, 
            hide_index=True,
            column_config={ 
                "Close": st.column_config.NumberColumn("Close", format="Rp %,.0f"),
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
                "Close": st.column_config.NumberColumn("Close", format="Rp %,.0f"),
                "Value": st.column_config.NumberColumn("Value", format="Rp %,.0f")
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
            # PERBAIKAN: Hapus tickformat untuk menghindari ValueError
            fig_sig.update_layout(yaxis_title="Jumlah Saham")
            fig_sig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Jumlah: %{y:,.0f}<extra></extra>')
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
            # PERBAIKAN: Hapus tickformat untuk menghindari ValueError
            fig_sec.update_layout(yaxis_title="Jumlah Saham")
            fig_sec.update_traces(texttemplate='%{text:,.0f}', textposition='outside', hovertemplate='<b>%{x}</b><br>Jumlah: %{y:,.0f}<extra></extra>')
            st.plotly_chart(fig_sec, use_container_width=True)
        else:
            st.info("Tidak ada saham dengan 'Unusual Volume' pada tanggal ini.")


# --- TAB 2: ANALISIS INDIVIDUAL ---
with tab2:
    st.subheader("Analisis Time Series Saham Individual")
    
    all_stocks = sorted(df["Stock Code"].dropna().unique())
    # Cari AADI sebagai default, jika tidak ada, pakai saham pertama
    default_index = all_stocks.index("AADI") if "AADI" in all_stocks else 0
    
    stock_to_analyze = st.selectbox(
        "Pilih Saham untuk dianalisis:",
        all_stocks,
        index=default_index
    )
    
    if stock_to_analyze:
        # Ambil SEMUA riwayat data saham ini, bukan hanya df_day
        df_stock = df[df['Stock Code'] == stock_to_analyze].sort_values('Last Trading Date')
        
        if df_stock.empty:
            st.warning(f"Tidak ditemukan data historis untuk {stock_to_analyze}")
        else:
            st.info(f"Menampilkan data untuk: **{df_stock.iloc[0]['Company Name']} ({stock_to_analyze})**")
            
            # --- START: Chart GABUNGAN (Harga, NFF, Volume) ---
            
            # 1. Buat Subplots (2 baris, 1 kolom, X-axis terhubung)
            fig_combined = make_subplots(
                rows=2, 
                cols=1, 
                shared_xaxes=True, 
                row_heights=[0.7, 0.3], # 70% untuk harga, 30% untuk volume
                vertical_spacing=0.03, # Jarak antar chart
                specs=[[{"secondary_y": True}], # Spek baris 1 (dual Y-axis)
                       [{}]]                    # Spek baris 2 (standar)
            )

            # --- Baris 1: Harga vs NFF ---
            
            # Trace 1 (Baris 1, KIRI) = Net Foreign Flow (NFF)
            colors_nff = ['#2ca02c' if v > 0 else '#d62728' for v in df_stock['Net Foreign Flow']]
            fig_combined.add_trace(
                go.Bar(
                    x=df_stock['Last Trading Date'],
                    y=df_stock['Net Foreign Flow'],
                    name="Net Foreign Flow",
                    marker_color=colors_nff,
                    opacity=0.6,
                    hovertemplate='<b>NFF</b>: %{y:,.0f}<br><b>Tanggal</b>: %{x|%d %b %Y}<extra></extra>'
                ),
                row=1, col=1, secondary_y=False # <-- Baris 1, KIRI
            )
            
            # Trace 2 (Baris 1, KANAN) = Harga (Close)
            fig_combined.add_trace(
                go.Scatter(
                    x=df_stock['Last Trading Date'],
                    y=df_stock['Close'],
                    name="Harga (Close)",
                    mode='lines',
                    line=dict(color='#1f77b4'), 
                    hovertemplate='<b>Harga</b>: Rp %{y:,.0f}<br><b>Tanggal</b>: %{x|%d %b %Y}<extra></extra>'
                ),
                row=1, col=1, secondary_y=True # <-- Baris 1, KANAN
            )
            
            # --- Baris 2: Volume vs MA20_vol ---
            
            # Trace 3 (Baris 2, KIRI) = Volume
            fig_combined.add_trace(
                go.Bar(
                    x=df_stock['Last Trading Date'],
                    y=df_stock['Volume'],
                    name="Volume",
                    marker_color='#8c564b', # Warna coklat
                    opacity=0.6,
                    hovertemplate='<b>Volume</b>: %{y:,.0f}<br><b>Tanggal</b>: %{x|%d %b %Y}<extra></extra>'
                ),
                row=2, col=1 # <-- Baris 2
            )

            # Trace 4 (Baris 2, KIRI) = MA20 Volume
            fig_combined.add_trace(
                go.Scatter(
                    x=df_stock['Last Trading Date'], 
                    y=df_stock['MA20_vol'], 
                    mode='lines', 
                    name='MA20 Volume',
                    line=dict(color='orange', dash='dash'),
                    hovertemplate='<b>MA20 Vol</b>: %{y:,.0f}<br><b>Tanggal</b>: %{x|%d %b %Y}<extra></extra>'
                ),
                row=2, col=1 # <-- Baris 2
            )

            # --- Konfigurasi Layout Gabungan ---
            
            # Sembunyikan label X-axis di chart atas (karena di-share)
            fig_combined.update_xaxes(showticklabels=False, row=1, col=1)
            
            # Atur judul-judul Sumbu Y
            fig_combined.update_yaxes(title_text="Net Foreign Flow (Shares)", row=1, col=1, secondary_y=False, showticklabels=True) # Paksa Tampil
            fig_combined.update_yaxes(title_text="Harga (Rp)", row=1, col=1, secondary_y=True, showticklabels=True) # Paksa Tampil
            fig_combined.update_yaxes(title_text="Volume (Shares)", row=2, col=1, showticklabels=True) # Paksa Tampil
            
            # Atur layout utama
            fig_combined.update_layout(
                height=600, # Bikin lebih tinggi
                title_text=f"Analisis Lengkap - {stock_to_analyze}",
                hovermode="x unified", # Tooltip terhubung
                legend_traceorder="normal",
                xaxis2_title="Tanggal" # Judul X-axis di chart bawah
            )
            
            st.plotly_chart(fig_combined, use_container_width=True)
            # --- END: Chart GABUNGAN ---

# --- TAB 3: DATA FILTER ---
with tab3:
    st.subheader("Data Terfilter (Sesuai Pilihan Sidebar)")
    st.markdown(f"**Menampilkan {len(df_filtered)} baris data**")
    
    columns_to_show = [
        "Stock Code", "Company Name", "Sector",
        "Close", "Change %", "Volume", "Volume Spike (x)",
        "Unusual Volume", "Net Foreign Flow", "Final Signal"
    ]
    # Pastikan kolom ada sebelum menampilkannya
    available_columns = [col for col in columns_to_show if col in df_filtered.columns]
    
    if not df_filtered.empty:
        st.dataframe(
            df_filtered[available_columns].sort_values("Volume Spike (x)", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={ 
                "Close": st.column_config.NumberColumn("Close", format="Rp %,.0f"),
                "Volume": st.column_config.NumberColumn("Volume", format="%,.0f"),
                "Volume Spike (x)": st.column_config.NumberColumn("Volume Spike (x)", format="%.2fx"),
                "Net Foreign Flow": st.column_config.NumberColumn("Net Foreign Flow", format="%,.0f"),
                "Change %": st.column_config.NumberColumn("Change %", format="%.2f")
            }
        )
    else:
        st.info("Tidak ada data yang sesuai dengan filter Anda.")

# --- TAB 4: SAHAM POTENSIAL (TOP 20) ---
with tab4:
    st.subheader(f"🏆 TOP 20 Saham Potensial (Analisis per {max_date.strftime('%d %B %Y')})")
    st.caption("Dihitung menggunakan data 30 hari (Trend) dan 7 hari (Momentum) terakhir.")

    # Hitung skor. Gunakan max_date (Timestamp)
    top20_df = calculate_potential_score(df, max_date, W)

    if not top20_df.empty:
        st.dataframe(
            top20_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Potential Score": st.column_config.NumberColumn("Skor", format="%.2f", help="Skor gabungan (Trend, Momentum, NBSA, dll)"),
                "Trend Score": st.column_config.NumberColumn("Skor Trend (30d)", format="%.2f"),
                "Momentum Score": st.column_config.NumberColumn("Skor Momentum (7d)", format="%.2f"),
                "total_net_ff_30d": st.column_config.NumberColumn("Net FF (30d)", format="%,.0f"),
                "foreign_contrib_pct": st.column_config.NumberColumn("Kontribusi Asing (%)", format="%.2f%%"),
                "last_price": st.column_config.NumberColumn("Harga Close", format="Rp %,.0f"),
                "last_final_signal": st.column_config.TextColumn("Signal Terakhir"),
                "sector": st.column_config.TextColumn("Sektor"),
            }
        )
    else:
        st.warning("Gagal menghitung skor TOP 20. Cek kembali data sumber.")


st.markdown("---")
st.info("Data diambil dari Google Drive dan di-cache selama 1 jam. Refresh halaman jika data di GDrive baru diperbarui.")

