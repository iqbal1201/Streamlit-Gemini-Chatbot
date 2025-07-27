import streamlit as st
import math
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import google.generativeai as genai

# Configure Google Gemini API (Replace with your actual API key)
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("Google API Key not found. Please set it in Streamlit secrets.")
    st.stop()

model_gemini = genai.GenerativeModel('gemini-2.0-flash')

# --- Memuat model dan scaler yang sudah dilatih ---
@st.cache_resource
def load_trained_model():
    try:
        model = load_model("bbri_lstm_model.h5")
        scaler = joblib.load("bbri_scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error memuat model atau scaler: {e}. Pastikan Anda telah melatih dan menyimpan model dengan menjalankan script pelatihan terlebih dahulu.")
        st.stop()

model_lstm, scaler_lstm = load_trained_model()

def get_stock_data(ticker, start_date, end_date):
    """Fetches stock data using yfinance, handling MultiIndex columns."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            close_col_name = [col for col in df.columns if 'Close' in col][0]
            df['Close'] = df[close_col_name]
            df.columns = df.columns.droplevel(1) # Flatten MultiIndex for easier access
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def predict_future_price(model, scaler, last_n_days_data, look_back=60):
    """Predicts the next day's closing price."""
    scaled_data = scaler.transform(last_n_days_data.reshape(-1, 1))

    X_input = []
    X_input.append(scaled_data[-look_back:, 0])
    X_input = np.array(X_input)
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    pred_price_scaled = model.predict(X_input, verbose=0)
    pred_price = scaler.inverse_transform(pred_price_scaled)
    return pred_price[0][0]

def generate_chatbot_response(prompt, stock_data_info, plot_description):
    """Generates a chatbot response using Gemini."""
    full_prompt = f"""
    Anda adalah chatbot yang berpengetahuan luas tentang data harga saham dan grafik.
    Berikut adalah informasi mengenai data saham yang sedang ditampilkan:
    {stock_data_info}
    Berikut adalah deskripsi visual dari grafik:
    {plot_description}

    Berdasarkan informasi di atas dan pengetahuan umum Anda, jawablah pertanyaan pengguna:
    "{prompt}"
    """
    try:
        response = model_gemini.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error dari Gemini API: {e}. Coba lagi.")
        return "Maaf, terjadi kesalahan saat memproses pertanyaan Anda."

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Prediksi Harga Saham BBRI dengan LSTM dan Chatbot")
st.write("Model LSTM telah dilatih sebelumnya dan dimuat di sini untuk visualisasi interaktif dan prediksi.")

# Initialize session state variables if not already present
if 'df_bbri' not in st.session_state:
    st.session_state.df_bbri = None
if 'predicted_tomorrow_price' not in st.session_state:
    st.session_state.predicted_tomorrow_price = None
if 'plot_df' not in st.session_state:
    st.session_state.plot_df = None
if 'initial_data_loaded' not in st.session_state:
    st.session_state.initial_data_loaded = False

# --- Sidebar Inputs ---
st.sidebar.header("Pengaturan Data & Prediksi")
start_date_input = st.sidebar.date_input("Tanggal Mulai Data", value=datetime(2019, 1, 1))
end_date_input = st.sidebar.date_input("Tanggal Akhir Data", value=datetime.now())

# Check for valid date range
if start_date_input >= end_date_input:
    st.sidebar.error("Tanggal mulai harus sebelum tanggal akhir.")
    st.session_state.initial_data_loaded = False
else:
    st.sidebar.success("Rentang tanggal valid.")

# --- Logic for loading initial data and running analysis ---
if st.sidebar.button("Refresh Data & Analisis") or not st.session_state.initial_data_loaded:
    with st.spinner("Mengunduh data dan menjalankan analisis..."):
        df_bbri_temp = get_stock_data('BBRI.JK', start_date_input.strftime('%Y-%m-%d'), end_date_input.strftime('%Y-%m-%d'))
        
        if df_bbri_temp is not None and not df_bbri_temp.empty:
            st.session_state.df_bbri = df_bbri_temp
            st.session_state.initial_data_loaded = True
            
            # --- Bagian Prediksi Menggunakan Model yang Sudah Dilatih ---
            look_back = 60
            if len(st.session_state.df_bbri['Close']) >= look_back:
                last_look_back_days_data = st.session_state.df_bbri['Close'].tail(look_back).values
                predicted_price = predict_future_price(model_lstm, scaler_lstm, last_look_back_days_data, look_back)
                st.session_state.predicted_tomorrow_price = predicted_price

                scaled_current_data = scaler_lstm.transform(st.session_state.df_bbri['Close'].values.reshape(-1, 1))
                
                x_pred_vis = []
                if len(scaled_current_data) >= look_back:
                    for i in range(look_back, len(scaled_current_data)):
                        x_pred_vis.append(scaled_current_data[i-look_back:i, 0])
                    
                    x_pred_vis = np.array(x_pred_vis)
                    x_pred_vis = np.reshape(x_pred_vis, (x_pred_vis.shape[0], x_pred_vis.shape[1], 1))
                    
                    predictions_vis_full = model_lstm.predict(x_pred_vis, verbose=0)
                    predictions_vis_full = scaler_lstm.inverse_transform(predictions_vis_full)
                    
                    plot_df_temp = st.session_state.df_bbri.copy()
                    plot_df_temp['Predictions'] = np.nan

                    prediction_indices_full = st.session_state.df_bbri.index[look_back:]
                    
                    if len(predictions_vis_full) == len(prediction_indices_full):
                        plot_df_temp.loc[prediction_indices_full, 'Predictions'] = predictions_vis_full.flatten()
                        st.session_state.plot_df = plot_df_temp
                    else:
                        st.error("Ada ketidakcocokan panjang antara prediksi dan indeks data. Prediksi visualisasi mungkin tidak akurat.")
                        st.session_state.plot_df = None
                        
                else:
                    st.session_state.plot_df = None
                    st.warning(f"Data yang diunduh ({len(st.session_state.df_bbri)} hari) tidak cukup untuk visualisasi prediksi karena kurang dari {look_back} hari.")

            else:
                st.warning(f"Data yang diunduh ({len(st.session_state.df_bbri)} hari) tidak cukup untuk memprediksi harga hari berikutnya. Dibutuhkan setidaknya {look_back} hari data.")
                st.session_state.predicted_tomorrow_price = None
                st.session_state.plot_df = None
        else:
            st.session_state.initial_data_loaded = False
            st.session_state.df_bbri = None
            st.session_state.predicted_tomorrow_price = None
            st.session_state.plot_df = None
            st.warning("Tidak ada data yang tersedia untuk rentang tanggal yang dipilih atau ticker tidak valid.")

# --- Display analysis results always if data is loaded ---
if st.session_state.initial_data_loaded and st.session_state.df_bbri is not None and not st.session_state.df_bbri.empty:
    st.header("1. Data Saham BBRI")
    st.write("Data Saham BBRI (Beberapa Baris Terakhir):")
    st.dataframe(st.session_state.df_bbri.tail())

    st.header("2. Grafik Harga Penutupan BBRI")
    plt.style.use('fivethirtyeight')
    fig_close, ax_close = plt.subplots(figsize=(16, 8))
    ax_close.plot(st.session_state.df_bbri.index, st.session_state.df_bbri['Close'])
    ax_close.set_title('Close Price History BBRI', fontsize=20)
    ax_close.set_xlabel('Date', fontsize=18)
    ax_close.set_ylabel('Close Price IDR (Rp)', fontsize=18)
    st.pyplot(fig_close)

    st.header("3. Prediksi Menggunakan Model LSTM yang Telah Dilatih")
    if st.session_state.predicted_tomorrow_price is not None:
        st.subheader(f"Harga Penutupan BBRI yang Diprediksi untuk Hari Berikutnya: Rp {st.session_state.predicted_tomorrow_price:.2f}")

    if st.session_state.plot_df is not None and 'Predictions' in st.session_state.plot_df.columns:
        st.subheader("Visualisasi Model Prediksi Harga Penutupan BBRI")
        
        # Matplotlib untuk Visualisasi Prediksi (sesuai contoh)
        fig_pred, ax_pred = plt.subplots(figsize=(16, 8))
        
        # Hitung ulang visual_training_data_len di sini juga untuk plotting
        visual_training_data_len = math.ceil(len(st.session_state.plot_df) * .8)

        # Pisahkan data untuk plotting Train dan Val
        train_plot = st.session_state.plot_df[:visual_training_data_len]
        valid_plot = st.session_state.plot_df[visual_training_data_len:].copy()
        
        # Pastikan kolom 'Predictions' di valid_plot sesuai
        # valid_plot['Predictions'] sudah terisi dari plot_df_temp

        ax_pred.plot(train_plot.index, train_plot['Close'], label='Data Pelatihan')
        ax_pred.plot(valid_plot.index, valid_plot['Close'], label='Harga Aktual (Validasi)')
        
        # Hanya plot prediksi jika ada nilai non-NaN di dalamnya
        valid_predictions_df_for_plot = valid_plot.dropna(subset=['Predictions'])
        if not valid_predictions_df_for_plot.empty:
            ax_pred.plot(valid_predictions_df_for_plot.index, valid_predictions_df_for_plot['Predictions'], label='Prediksi Model')
        
        ax_pred.set_title('Model Prediksi Harga Penutupan BBRI', fontsize=20)
        ax_pred.set_xlabel('Date', fontsize=18)
        ax_pred.set_ylabel('Close Price IDR (Rp)', fontsize=18)
        ax_pred.legend(loc='lower right')
        st.pyplot(fig_pred)

        st.subheader("Tabel Harga Aktual dan Prediksi (Bagian Terbaru)")
        st.dataframe(st.session_state.plot_df[['Close', 'Predictions']].tail(60)) # Tetap 60 hari terakhir untuk tabel

    # --- Chatbot Feature (always visible after initial load) ---
    st.header("4. Chatbot Gemini - Tanyakan tentang Data & Grafik")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ajukan pertanyaan tentang data saham BBRI atau grafik di atas:"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        stock_data_info = f"Data saham BBRI dari {start_date_input} hingga {end_date_input}. Jumlah baris data: {len(st.session_state.df_bbri)}."
        stock_data_info += f"Data memiliki kolom: {st.session_state.df_bbri.columns.tolist()}."
        stock_data_info += f"Harga penutupan terakhir: Rp {st.session_state.df_bbri['Close'].iloc[-1]:.2f}."
        if st.session_state.predicted_tomorrow_price is not None:
            stock_data_info += f"Harga prediksi untuk hari berikutnya: Rp {st.session_state.predicted_tomorrow_price:.2f}."
        
        # Sesuaikan deskripsi plot untuk chatbot
        plot_description = f"""
        Grafik 'Close Price History BBRI' menampilkan riwayat harga penutupan saham BBRI secara keseluruhan dari tanggal {start_date_input} hingga {end_date_input}.
        Grafik 'Harga Aktual vs Prediksi Model BBRI' menampilkan harga aktual saham dari awal data yang diunduh, dan superimposed di atasnya adalah harga yang diprediksi oleh model LSTM, namun harga prediksi hanya ditampilkan untuk sekitar 60 hari terakhir dari data yang ada.
        Sumbu X adalah Tanggal dan Sumbu Y adalah Harga Penutupan dalam IDR (Rp).
        """
        
        with st.chat_message("assistant"):
            with st.spinner("Memproses pertanyaan..."):
                chatbot_response = generate_chatbot_response(user_question, stock_data_info, plot_description)
                st.markdown(chatbot_response)
        st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

else:
    st.info("Pilih rentang tanggal di sidebar dan klik 'Refresh Data & Analisis' untuk memuat grafik dan memulai analisis.")

st.markdown("""
<style>
.reportview-container .main .block-container{
    padding-top: 2rem;
    padding-right: 2rem;
    padding-left: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)