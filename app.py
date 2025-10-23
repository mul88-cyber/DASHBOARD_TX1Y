import streamlit as st
import pandas as pd
import plotly.express as px
# import gdown (Sudah tidak diperlukan di versi ini)

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
# 📦 MEMUAT DAN MEMBERSIHKAN DATA (dari File Lokal)
# =====================================================================
LOCAL_FILE = "Kompilasi_Data_1Tahun.csv" # Membaca dari file lokal

@st.cache_data(ttl=3600)
def load_data():
    try:
        # Langsung baca file lokal
        df = pd.read_csv(LOCAL_FILE) 
        df.columns = df.columns.str.strip()
        df['Last Trading Date'] = pd.to_datetime(df['Last Trading Date'])
        
        # Konversi kolom-kolom penting ke numerik
        # (Versi awal Anda memiliki spasi di ' Change % ' dan ' Volume Spike (x) ')
        cols_to_numeric = [
            'Change %', 'Typical Price', 'TPxV', 'VWMA_20D', 'MA20_vol', 
            'MA5_vol', 'Volume Spike (x)', 'Net Foreign Flow', 
            'Bid/Offer Imbalance', 'Money Flow Value', 'Close', 'Volume', 'Value',
            ' Change % ', ' Volume Spike (x) ' # Menambahkan nama kolom yang salah
        ]
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Mengganti nama kolom yang salah (jika ada)
        if ' Change % ' in df.columns and 'Change %' not in df.columns:
             df = df.rename(columns={' Change % ': 'Change %'})
        if ' Volume Spike (x) ' in df.columns and 'Volume Spike (x)' not in df.columns:
             df = df.rename(columns={' Volume Spike (x) ': 'Volume Spike (x)'})

        # Bersihkan kolom 'Unusual Volume'
        if 'Unusual Volume' in df.columns:
            if df['Unusual Volume'].dtype == 'object':
                df['Unusual Volume'] = df['Unusual Volume'].str.strip().str.lower().isin(['spike volume signifikan', 'true'])
            df['Unusual Volume'] = df['Unusual Volume'].astype(bool)
        
        df = df.dropna(subset=['Last Trading Date', 'Stock Code'])
        return df
    
    except FileNotFoundError:
        st.error(f"❌ File '{LOCAL_FILE}' tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Gagal membaca file lokal: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("⚠️ Data belum berhasil dimuat. Aplikasi tidak dapat dilanjutkan.")
    st.stop()

# =====================================================================
# 🧭 FILTER DATA (SIDEBAR)
# =====================================================================
st.sidebar.header("🎛️ Filter Analisis Harian")

max_date = df['Last Trading Date'].max().date()
selected_date = st.sidebar.date_input(
    "Pilih Tanggal Analisis",
    max_date,
    min_value=df['Last Trading Date'].min().date(),
    max_value=max_date,
    format="DD-MM-YYYY"
)

df_day = df[df['Last Trading Date'].dt.date == selected_date].copy()

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

if not df_filtered.empty:
    df_filtered = df_filtered[df_filtered["Volume Spike (x)"] >= min_spike]

if show_only_spike:
    if not df_filtered.empty:
        df_filtered = df_filtered[df_filtered["Unusual Volume"] == True]

# =====================================================================
#  LAYOUT UTAMA (DENGAN TABS)
# =====================================================================
st.caption(f"Menampilkan data untuk tanggal: **{selected_date.strftime('%d %B %Y')}**")

tab1, tab2, tab3 = st.tabs([
    "📊 **Dashboard Harian**",
    "📈 **Analisis Individual**",
    "📋 **Data Filter**"
])

# --- TAB 1: DASHBOARD HARIAN ---
with tab1:
    st.subheader("Ringkasan Pasar (pada tanggal terpilih)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Saham Aktif", len(df_day['Stock Code'].unique()))
    col2.metric("Saham Unusual Volume", int(df_day['Unusual Volume'].sum()))
    col3.metric("Total Nilai Transaksi (Value)", f"Rp {df_day['Value'].sum():,.0f}") # Format Koma ditambahkan

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
            column_config={ # Perbaikan untuk format desimal
                "Close": st.column_config.NumberColumn(format="Rp %d"),
                "Change %": st.column_config.NumberColumn(format="%.2f%%")
            }
        )

    with col_l:
        st.markdown("**Top 10 Losers (%)**")
        top_losers = df_day.sort_values("Change %", ascending=True).head(10)
        st.dataframe(
            top_losers[['Stock Code', 'Close', 'Change %']], 
            use_container_width=True, 
            hide_index=True,
            column_config={ # Perbaikan untuk format desimal
                "Close": st.column_config.NumberColumn(format="Rp %d"),
                "Change %": st.column_config.NumberColumn(format="%.2f%%")
            }
        )
        
    with col_v:
        st.markdown("**Top 10 by Value**")
        top_value = df_day.sort_values("Value", ascending=False).head(10)
        st.dataframe(
            top_value[['Stock Code', 'Close', 'Value']], 
            use_container_width=True, 
            hide_index=True,
             column_config={ # Perbaikan untuk format desimal
                "Close": st.column_config.NumberColumn(format="Rp %d"),
                "Value": st.column_config.NumberColumn(format="Rp %d")
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
                title="Distribusi Final Signal"
            )
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
                title="Distribusi Sektor (Hanya Unusual Volume)"
            )
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
            
            # Chart Harga (Close)
            fig_price = px.line(
                df_stock, 
                x='Last Trading Date', 
                y='Close', 
                title=f"Pergerakan Harga - {stock_to_analyze}"
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # Chart Volume vs MA20
            fig_vol = px.bar(
                df_stock, 
                x='Last Trading Date', 
                y='Volume', 
                title=f"Volume Perdagangan vs. MA20 - {stock_to_analyze}"
            )
            fig_vol.add_scatter(
                x=df_stock['Last Trading Date'], 
                y=df_stock['MA20_vol'], 
                mode='lines', 
                name='MA20 Volume',
                line=dict(color='orange', dash='dash')
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Chart Net Foreign Flow
            fig_nff = px.bar(
                df_stock, 
                x='Last Trading Date', 
                y='Net Foreign Flow', 
                title=f"Aliran Dana Asing (Net) - {stock_to_analyze}",
                color='Net Foreign Flow',
                color_continuous_scale=['red', 'green']
            )
            st.plotly_chart(fig_nff, use_container_width=True)

# --- TAB 3: DATA FILTER ---
with tab3:
    st.subheader("Data Terfilter (Sesuai Pilihan Sidebar)")
    st.markdown(f"**Menampilkan {len(df_filtered)} baris data**")
    
    columns_to_show = [
        "Stock Code", "Company Name", "Sector",
        "Close", "Change %", "Volume", "Volume Spike (x)",
        "Unusual Volume", "Net Foreign Flow", "Final Signal"
    ]
    available_columns = [col for col in columns_to_show if col in df_filtered.columns]
    
    st.dataframe(
        df_filtered[available_columns].sort_values("Volume Spike (x)", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={ # Perbaikan untuk format desimal
            "Close": st.column_config.NumberColumn(format="Rp %d"),
            "Change %": st.column_config.NumberColumn(format="%.2f%%"),
            "Volume": st.column_config.NumberColumn(format="%d"),
            "Volume Spike (x)": st.column_config.NumberColumn(format="%.2fx"),
            "Net Foreign Flow": st.column_config.NumberColumn(format="%d"),
        }
    )

st.markdown("---")
st.info("Data diambil dari file CSV lokal. Muat ulang halaman jika file diperbarui.")

