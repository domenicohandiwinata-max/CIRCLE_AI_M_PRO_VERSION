import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pycoingecko import CoinGeckoAPI

# ==================================================
# CONFIG - NO API KEY NEEDED!
# ==================================================

st.set_page_config(
    page_title="CIRCLE AI PRO | Market Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PRO UI STYLING
# ==================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1429 100%);
        color: #e8eaf6;
    }
    
    /* PRO Badge */
    .pro-badge {
        background: linear-gradient(135deg, #00f5d4 0%, #00bbf9 100%);
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.75rem;
        box-shadow: 0 4px 15px rgba(0, 245, 212, 0.4);
        animation: glow 2s infinite;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 4px 15px rgba(0, 245, 212, 0.4); }
        50% { box-shadow: 0 4px 25px rgba(0, 245, 212, 0.6); }
    }
    
    /* Disclaimer */
    .disclaimer-banner {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15), rgba(255, 152, 0, 0.1));
        border: 2px solid rgba(255, 193, 7, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 25px;
        text-align: center;
    }
    
    .disclaimer-text {
        color: #ffd93d;
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
    }
    
    .disclaimer-sub {
        color: #8892b0;
        font-size: 0.85rem;
        margin: 8px 0 0 0;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(145deg, rgba(28, 31, 38, 0.8), rgba(20, 23, 30, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }
    
    .metric-label {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #e6f1ff;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    /* Signals */
    .signal-buy {
        color: #00f5d4;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(0, 245, 212, 0.5);
        animation: pulse-buy 2s infinite;
    }
    
    .signal-sell {
        color: #ff6b6b;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(255, 107, 107, 0.5);
        animation: pulse-sell 2s infinite;
    }
    
    .signal-wait {
        color: #ffd93d;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(255, 217, 61, 0.5);
    }
    
    @keyframes pulse-buy {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    
    @keyframes pulse-sell {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(0.98); }
    }
    
    /* Confidence Bar */
    .confidence-bar {
        width: 100%;
        height: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        overflow: hidden;
        margin-top: 12px;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s ease;
        background: linear-gradient(90deg, #00f5d4, #00bbf9);
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 20px;
        margin-top: 20px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px 20px;
        color: #8892b0;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 40px;
    }
    
    .footer-brand {
        font-weight: 700;
        color: #e6f1ff;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# CORE AI ENGINE
# ==================================================

def calculate_indicators(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA7'] = df['close'].rolling(7).mean()
    df['MA30'] = df['close'].rolling(30).mean()
    df['Volatility'] = df['close'].rolling(14).std()
    
    # Advanced indicators (PRO)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_upper'] = df['close'].rolling(20).mean() + (df['close'].rolling(20).std() * 2)
    df['BB_lower'] = df['close'].rolling(20).mean() - (df['close'].rolling(20).std() * 2)
    
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df.dropna()

def run_ai_prediction(df):
    features = ['close', 'MA7', 'MA30', 'RSI', 'Volatility', 'MACD', 'Signal_Line']
    X = df[features]
    y = df['Target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    
    last_row = X.iloc[-1:]
    proba = model.predict_proba(scaler.transform(last_row))[0]
    return proba

# ==================================================
# CRYPTO FUNCTIONS
# ==================================================

@st.cache_data(ttl=300)
def get_crypto_price(crypto_id):
    cg = CoinGeckoAPI()
    try:
        data = cg.get_price(
            ids=crypto_id,
            vs_currencies='usd',
            include_market_cap=True,
            include_24hr_vol=True,
            include_24hr_change=True
        )
        return {
            'usd': data[crypto_id]['usd'],
            'usd_market_cap': data[crypto_id]['usd_market_cap'],
            'usd_24h_vol': data[crypto_id]['usd_24h_vol'],
            'usd_24h_change': data[crypto_id]['usd_24h_change']
        }
    except Exception as e:
        st.error(f"Error fetching price: {e}")
        return None

@st.cache_data(ttl=300)
def get_crypto_data(crypto_id, days):
    cg = CoinGeckoAPI()
    try:
        data = cg.get_coin_market_chart_by_id(
            id=crypto_id,
            vs_currency='usd',
            days=days
        )
        prices = data['prices']
        volumes = data['total_volumes']
        
        df = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df['volume'] = [v[1] for v in volumes]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Generate OHLC
        df['open'] = df['close'].shift(1)
        df['high'] = df['close'] * 1.002
        df['low'] = df['close'] * 0.998
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"Error fetching crypto data: {e}")
        return None

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="margin: 0; font-size: 1.8rem; letter-spacing: -1px;">
                🧠 <span style="background: linear-gradient(135deg, #00f5d4, #00bbf9); 
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">CIRCLE</span> AI
            </h2>
            <p style="margin-top: 10px;">
                <span class="pro-badge">PRO VERSION</span>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Market Selection (ALL UNLOCKED)
    st.markdown('<p style="color: #8892b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Pilih Market</p>', unsafe_allow_html=True)
    
    mode = st.selectbox("", 
        ["📈 Saham (Yahoo)", "💱 Forex (Yahoo)", "🪙 Crypto (CoinGecko)"],
        label_visibility="collapsed"
    )
    
    # Timeframe Selection (ALL UNLOCKED)
    st.markdown('<br><p style="color: #8892b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Timeframe</p>', unsafe_allow_html=True)
    
    timeframe = st.selectbox("", 
        ["📅 Daily", "⚡ H1 (Intraday)", "⚡ H4 (Intraday)"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Input based on market
    st.markdown('<p style="color: #8892b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Asset</p>', unsafe_allow_html=True)
    
    if mode == "📈 Saham (Yahoo)":
        ticker = st.text_input("", value="BBCA.JK", placeholder="e.g., BBCA.JK, AAPL").upper()
        st.caption("Gunakan .JK untuk saham Indonesia")
    elif mode == "💱 Forex (Yahoo)":
        st.markdown("**Format:** USDIDR=X, EURUSD=X, GBPUSD=X, JPY=X")
        ticker = st.text_input("", value="USDIDR=X", placeholder="USDIDR=X").upper()
        st.caption("Currency pair dari Yahoo Finance")
    else:  # Crypto
        crypto_options = {
            "Bitcoin (BTC)": "bitcoin",
            "Ethereum (ETH)": "ethereum",
            "Binance Coin (BNB)": "binancecoin",
            "Solana (SOL)": "solana",
            "Ripple (XRP)": "ripple",
            "Cardano (ADA)": "cardano",
            "Dogecoin (DOGE)": "dogecoin",
            "Polkadot (DOT)": "polkadot",
            "Avalanche (AVAX)": "avalanche-2",
            "Chainlink (LINK)": "chainlink"
        }
        selected_crypto = st.selectbox("", list(crypto_options.keys()))
        ticker = crypto_options[selected_crypto]

# ==================================================
# DISCLAIMER
# ==================================================
st.markdown("""
    <div class="disclaimer-banner">
        <p class="disclaimer-text">⚠️ PERINGATAN: Keputusan Investasi Ada di Tangan Anda</p>
        <p class="disclaimer-sub">
            Jangan simpulkan terlalu cepat dari sinyal AI ini. CIRCLE AI adalah alat bantu analisis, 
            bukan saran investasi resmi. Selalu lakukan riset mandiri dan pertimbangkan risiko Anda.
        </p>
    </div>
""", unsafe_allow_html=True)

# ==================================================
# MAIN DASHBOARD
# ==================================================
st.markdown(f"""
    <div style="margin-bottom: 30px;">
        <h1 style="margin: 0; font-size: 2.5rem;">Market Intelligence <span style="color: #00f5d4;">PRO</span></h1>
        <p style="color: #8892b0; margin: 10px 0 0 0; font-size: 1.1rem;">
            Analisis real-time untuk <span style="color: #00f5d4; font-weight: 600;">{ticker}</span>
            <span style="color: #ffd700; margin-left: 10px;">[{timeframe}]</span>
        </p>
    </div>
""", unsafe_allow_html=True)

# ==================================================
# DATA FETCHING (ALL VIA YAHOO FINANCE & COINGECKO - NO API KEY!)
# ==================================================
with st.spinner('⏳ Memuat data pasar...'):
    df_raw = None
    err = None
    
    if mode in ["📈 Saham (Yahoo)", "💱 Forex (Yahoo)"]:
        # SAHAM & FOREX sama-sama pake Yahoo Finance!
        period = "1y"
        interval = "1d"
        if "H1" in timeframe:
            period = "1mo"
            interval = "1h"
        elif "H4" in timeframe:
            period = "3mo"
            interval = "4h"
        
        data_yf = yf.download(ticker, period=period, interval=interval, progress=False)
        if not data_yf.empty:
            if isinstance(data_yf.columns, pd.MultiIndex):
                data_yf.columns = data_yf.columns.get_level_values(0)
            df_raw = data_yf.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
        else:
            err = f"Ticker {ticker} tidak ditemukan. Coba format: BBCA.JK atau USDIDR=X"
            
    else:  # Crypto
        days = "30" if ("H1" in timeframe or "H4" in timeframe) else "90"
        df_raw = get_crypto_data(ticker, days)
        if df_raw is None:
            err = "Gagal mengambil data crypto"

# ==================================================
# ANALYSIS & DISPLAY
# ==================================================
if df_raw is not None and len(df_raw) > 30:
    df_proc = calculate_indicators(df_raw)
    proba = run_ai_prediction(df_proc)
    
    prob_up = proba[1]
    conf = max(proba) * 100
    last_price = df_proc['close'].iloc[-1]
    
    if len(df_proc) > 1:
        price_change = ((last_price - df_proc['close'].iloc[-2]) / df_proc['close'].iloc[-2]) * 100
    else:
        price_change = 0
    
    # Determine Signal
    if conf < 58:
        signal, css_class, signal_desc = "HOLD", "signal-wait", "Tunggu sinyal lebih jelas"
        conf_color = "#ffd93d"
    elif prob_up > 0.5:
        signal, css_class, signal_desc = "BUY", "signal-buy", "Momentum naik terdeteksi"
        conf_color = "#00f5d4"
    else:
        signal, css_class, signal_desc = "SELL", "signal-sell", "Tren turun terprediksi"
        conf_color = "#ff6b6b"

    # METRICS
    st.markdown('<div style="margin-bottom: 30px;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        change_color = "#00f5d4" if price_change >= 0 else "#ff6b6b"
        change_icon = "▲" if price_change >= 0 else "▼"
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Harga Terakhir</div>
                <div class="metric-value">${last_price:,.2f}</div>
                <div style="color: {change_color}; font-size: 0.9rem; margin-top: 5px;">
                    {change_icon} {abs(price_change):.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Kepercayaan AI</div>
                <div class="metric-value" style="color: {conf_color};">{conf:.1f}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf}%; background: linear-gradient(90deg, {conf_color}, {conf_color}88);"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Sinyal AI</div>
                <div class="{css_class}">{signal}</div>
                <div style="color: #8892b0; font-size: 0.85rem; margin-top: 5px;">{signal_desc}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # DISCLAIMER NEAR SIGNAL
    st.markdown("""
        <div style="background: rgba(255, 107, 107, 0.08); border-left: 4px solid #ff6b6b; 
                    border-radius: 0 12px 12px 0; padding: 15px 20px; margin: 20px 0;">
            <p style="margin: 0; color: #ff9f9f; font-size: 0.9rem; font-weight: 500;">
                <span style="font-size: 1.2rem;">⚠️</span>
                Ini hanya prediksi AI, bukan rekomendasi beli/jual. Keputusan final tetap di tangan Anda.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # CHART
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08, 
        row_heights=[0.7, 0.3],
        subplot_titles=('Price Action', 'RSI & MACD')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_proc.index, 
        open=df_proc['open'], 
        high=df_proc['high'], 
        low=df_proc['low'], 
        close=df_proc['close'],
        increasing_line_color='#00f5d4',
        decreasing_line_color='#ff6b6b',
        name="OHLC"
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(
        x=df_proc.index, 
        y=df_proc['MA7'], 
        name="MA 7",
        line=dict(color='#00bbf9', width=2),
        opacity=0.9
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_proc.index, 
        y=df_proc['MA30'], 
        name="MA 30",
        line=dict(color='#f72585', width=2),
        opacity=0.9
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df_proc.index,
        y=df_proc['BB_upper'],
        name="BB Upper",
        line=dict(color='rgba(255,255,255,0.3)', width=1),
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_proc.index,
        y=df_proc['BB_lower'],
        name="BB Lower",
        line=dict(color='rgba(255,255,255,0.3)', width=1),
        fill='tonexty',
        fillcolor='rgba(255,255,255,0.05)',
        showlegend=False
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df_proc.index, 
        y=df_proc['RSI'], 
        name="RSI",
        line=dict(color='#ffd93d', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 217, 61, 0.1)'
    ), row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=df_proc.index,
        y=df_proc['MACD'],
        name="MACD",
        line=dict(color='#9d4edd', width=2)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df_proc.index,
        y=df_proc['Signal_Line'],
        name="Signal",
        line=dict(color='#c77dff', width=2, dash='dash')
    ), row=2, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="#ff6b6b", 
                  annotation_text="Overbought", row=2, col=1,
                  annotation_font_color="#ff6b6b")
    fig.add_hline(y=30, line_dash="dash", line_color="#00f5d4",
                  annotation_text="Oversold", row=2, col=1,
                  annotation_font_color="#00f5d4")
    
    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6f1ff'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.3)'
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # AI PREDICTION SECTION (PRO)
    st.markdown("""
        <div class="glass-card" style="margin-top: 30px;">
            <h3 style="margin: 0 0 20px 0; color: #ffd700;">
                🔮 AI Price Prediction (PRO Feature)
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    days_pred = st.slider("Prediksi hari ke depan", 1, 7, 3)
    
    if st.button("🚀 Generate Prediction"):
        recent_trend = (df_proc['close'].iloc[-1] - df_proc['close'].iloc[-7]) / df_proc['close'].iloc[-7]
        current_price = df_proc['close'].iloc[-1]
        predicted_price = current_price * (1 + recent_trend * days_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Harga Sekarang", f"${current_price:,.2f}")
        with col2:
            st.metric(
                f"Prediksi {days_pred} hari",
                f"${predicted_price:,.2f}",
                f"{((predicted_price-current_price)/current_price)*100:.2f}%"
            )
        
        # Chart prediction
        future_dates = [df_proc.index[-1] + timedelta(days=i) for i in range(1, days_pred+1)]
        future_prices = [current_price + (predicted_price - current_price) * i/days_pred for i in range(1, days_pred+1)]
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=df_proc.index[-14:],
            y=df_proc['close'][-14:],
            mode='lines',
            name='Historical',
            line=dict(color='#1E88E5')
        ))
        fig_pred.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices,
            mode='lines+markers',
            name='Prediction',
            line=dict(color='#FF6B6B', dash='dash')
        ))
        fig_pred.update_layout(
            title='Price Prediction',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Recommendation
        if recent_trend > 0.02:
            st.success("🟢 **SINYAL BELI** - Tren naik terdeteksi")
        elif recent_trend < -0.02:
            st.error("🔴 **SINYAL JUAL** - Tren turun terdeteksi")
        else:
            st.warning("🟡 **TUNGGU** - Pasar sideways")

    # INSIGHTS
    st.markdown("""
        <div style="margin-top: 30px;">
            <div class="glass-card">
                <h4 style="margin: 0 0 15px 0; color: #e6f1ff;">📊 Ringkasan Analisis AI PRO</h4>
                <p style="color: #8892b0; line-height: 1.6; margin: 0;">
                    Model Random Forest PRO (300 estimators) menganalisis indikator teknikal 
                    RSI, Moving Average, MACD, Bollinger Bands, dan Volatilitas untuk 
                    menghasilkan sinyal trading dengan kepercayaan tinggi.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

else:
    error_msg = err if err else "Data tidak cukup untuk analisis"
    st.markdown(f"""
        <div style="background: rgba(255,107,107,0.1); border: 1px solid rgba(255,107,107,0.3); 
                    border-radius: 16px; padding: 30px; text-align: center; margin-top: 20px;">
            <h3 style="color: #ff6b6b; margin: 0;">⚠️ Analisis Gagal</h3>
            <p style="color: #8892b0; margin: 10px 0 0 0;">{error_msg}</p>
        </div>
    """, unsafe_allow_html=True)

# ==================================================
# FOOTER DISCLAIMER
# ==================================================
st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px; padding: 25px; margin: 30px 0; text-align: center;">
        <p style="color: #ff6b6b; font-size: 1.2rem; font-weight: 700; margin: 0 0 10px 0;">
            ⚠️ PERINGATAN PENTING ⚠️
        </p>
        <p style="color: #8892b0; font-size: 0.95rem; line-height: 1.6; margin: 0;">
            <strong style="color: #ff6b6b;">Keputusan investasi sepenuhnya ada di tangan Anda.</strong><br><br>
            CIRCLE AI PRO adalah alat bantu analisis teknikal berbasis machine learning. 
            Sinyal yang ditampilkan bersifat prediktif dan tidak menjamin profit. 
            Pasar keuangan mengandung risiko tinggi. Selalu lakukan riset mandiri (DYOR) 
            dan konsultasikan dengan penasihat keuangan berlisensi.
            <strong style="color: #ffd93d;">Jangan simpulkan terlalu cepat dari sinyal AI.</strong>
        </p>
    </div>
""", unsafe_allow_html=True)

# FOOTER
st.markdown("""
    <div class="footer">
        <p class="footer-brand">CIRCLE AI PRO</p>
        <p style="margin: 10px 0;">Premium Market Intelligence Tool</p>
        <p style="font-size: 0.8rem; opacity: 0.6;">
            Built with passion by a 14-year-old developer • #BanggaBuatanIndonesia 🇮🇩
        </p>
    </div>
""", unsafe_allow_html=True)