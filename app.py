import streamlit as st
import joblib
import pandas as pd

# --- Fungsi untuk Memuat Model ---
# Menggunakan cache agar model tidak perlu dimuat ulang setiap kali ada interaksi
@st.cache_resource
def load_model(path):
    """Memuat model dari file .pkl"""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan di path: {path}")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop()

# --- Konfigurasi Halaman dan Judul ---
st.set_page_config(page_title="Prediksi Risiko Kredit", layout="wide")
st.title("Aplikasi Prediksi Risiko Kredit 💳")
st.write(
    "Aplikasi ini menggunakan model Logistic Regression untuk memprediksi "
    "risiko kredit nasabah berdasarkan fitur-fitur yang relevan. "
    "Silakan masukkan data nasabah di sidebar kiri."
)

# --- Memuat Model ---
MODEL_PATH = 'model_logistic_regression_best.pkl'
model = load_model(MODEL_PATH)

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("Input Fitur Nasabah")

# --- Mapping untuk Fitur Kategorikal ---
# Dibuat berdasarkan urutan abjad, sesuai cara kerja LabelEncoder
housing_map = {'Gratis (free)': 0, 'Milik Sendiri (own)': 1, 'Sewa (rent)': 2}
saving_map = {'Sedikit (little)': 0, 'Sedang (moderate)': 1, 'Cukup Kaya (quite rich)': 2, 'Kaya (rich)': 3, 'Tidak Diketahui (unknown)': 4}
checking_map = {'Sedikit (little)': 0, 'Sedang (moderate)': 1, 'Kaya (rich)': 2, 'Tidak Diketahui (unknown)': 3}
purpose_map = {
    'Bisnis': 0, 'Mobil': 1, 'Peralatan Rumah Tangga': 2, 'Edukasi': 3,
    'Furnitur/Peralatan': 4, 'Radio/TV': 5, 'Perbaikan': 6, 'Liburan/Lainnya': 7
}

# --- Input Widget di Sidebar ---
age = st.sidebar.slider("Usia (Tahun)", min_value=18, max_value=80, value=35, step=1)
credit_amount = st.sidebar.number_input("Jumlah Kredit", min_value=250, max_value=20000, value=3000, step=100)
duration = st.sidebar.slider("Durasi Pinjaman (Bulan)", min_value=4, max_value=72, value=24, step=1)

housing_label = st.sidebar.selectbox("Status Kepemilikan Rumah", list(housing_map.keys()))
saving_label = st.sidebar.selectbox("Status Rekening Tabungan", list(saving_map.keys()))
checking_label = st.sidebar.selectbox("Status Rekening Giro", list(checking_map.keys()))
purpose_label = st.sidebar.selectbox("Tujuan Kredit", list(purpose_map.keys()))

# Mengonversi label pilihan pengguna menjadi nilai numerik
housing = housing_map[housing_label]
saving_accounts = saving_map[saving_label]
checking_account = checking_map[checking_label]
purpose = purpose_map[purpose_label]

# --- Tombol Prediksi dan Tampilan Hasil ---
col1, col2 = st.columns([1, 4])

with col1:
    predict_button = st.button("Prediksi Risiko", use_container_width=True, type="primary")

if predict_button:
    # Membuat DataFrame dari input
    # Urutan kolom harus sama persis dengan urutan saat model dilatih
    features = pd.DataFrame({
        'Age': [age],
        'Housing': [housing],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [purpose]
    })

    # Melakukan prediksi
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    st.subheader("Hasil Prediksi")

    # Menampilkan hasil prediksi dengan visual yang jelas
    if prediction[0] == 1: # Asumsi 1 = Good, 0 = Bad
        st.success("## Risiko Rendah (Good)")
        st.progress(float(probability[0][1]))
        st.write(f"**Tingkat Kepercayaan:** {probability[0][1]*100:.2f}%")
    else:
        st.error("## Risiko Tinggi (Bad)")
        st.progress(float(probability[0][0]))
        st.write(f"**Tingkat Kepercayaan:** {probability[0][0]*100:.2f}%")

    # Menampilkan data input untuk verifikasi
    st.write("---")
    st.write("Data yang Dimasukkan:")
    st.dataframe(features)