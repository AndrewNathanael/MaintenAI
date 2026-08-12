import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Konfigurasi Halaman Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Analytics",
    page_icon="⚙️",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Fungsi Load Model & Data
# ---------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    model_path = "model/model.pkl"
    scaler_path = "model/scaler.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("⚠️ File 'model.pkl' atau 'scaler.pkl' tidak ditemukan! Silakan jalankan 'python train_model.py' terlebih dahulu di terminal.")
        st.stop()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

@st.cache_data
def load_data():
    data_path = "data/predictive_maintenance.csv"
    if os.path.exists(data_path):
        df_loaded = pd.read_csv(data_path)
        # Rename kolom asli Kaggle ke nama standar
        feature_mapping = {
            'Air temperature [K]': 'Air_temperature_K',
            'Process temperature [K]': 'Process_temperature_K',
            'Rotational speed [rpm]': 'Rotational_speed_rpm',
            'Torque [Nm]': 'Torque_Nm',
            'Tool wear [min]': 'Tool_wear_min',
            'Target': 'Machine_failure',
            'Machine failure': 'Machine_failure'
        }
        df_loaded = df_loaded.rename(columns=feature_mapping)
        return df_loaded
    return None

# ---------------------------------------------------------
# 3. Definisi Variabel Global (Wajib sebelum bagian Tab)
# ---------------------------------------------------------
model, scaler = load_ml_assets()
df = load_data()  # <--- Definisi variabel df dilakukan di sini

# ---------------------------------------------------------
# 4. Header & Sidebar Dashboard
# ---------------------------------------------------------
st.title("⚙️ Predictive Maintenance Sensor Dashboard")
st.markdown("Aplikasi Monitoring Data Sensor Industri & Prediksi Risiko Kerusakan Mesin secara Real-Time.")

st.sidebar.title("Navigasi & Info")
st.sidebar.info(
    "Dashboard ini membantu tim maintenance memprediksi kondisi "
    "anomali mesin berdasarkan indikator suhu, RPM, torsi, dan wear time."
)

# ---------------------------------------------------------
# 5. Tampilan Utama (Tabs)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard EDA & Metrik", 
    "🔮 Simulator Prediksi Real-Time", 
    "📁 Batch Prediction (Upload CSV)"
])

# =========================================================
# TAB 1: EDA & METRIK
# =========================================================
with tab1:
    st.header("Analisis Data Eksploratif (EDA) & Ringkasan Sensor")
    
    if df is not None:
        total_mesin = len(df)
        total_failure = df['Machine_failure'].sum() if 'Machine_failure' in df.columns else 0
        failure_rate = (total_failure / total_mesin) * 100 if total_mesin > 0 else 0
        
        # Ringkasan Metrik
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sampel Sensor", f"{total_mesin:,}")
        col2.metric("Total Kejadian Failure", f"{total_failure:,}")
        col3.metric("Failure Rate (%)", f"{failure_rate:.2f}%")
        col4.metric("Rata-Rata Tool Wear", f"{df['Tool_wear_min'].mean():.1f} min" if 'Tool_wear_min' in df.columns else "N/A")
        
        st.divider()
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Distribusi Risiko Kerusakan Mesin")
            if 'Machine_failure' in df.columns:
                fig_pie = px.pie(
                    df, names='Machine_failure', 
                    title='Perbandingan Status Normal (0) vs Failure (1)',
                    color='Machine_failure',
                    color_discrete_map={0: '#2ec4b6', 1: '#e71d36'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_right:
            st.subheader("Korelasi Fitur Sensor")
            numeric_df = df.select_dtypes(include=[np.number])
            fig_corr = px.imshow(
                numeric_df.corr(), text_auto=".2f", aspect="auto",
                color_continuous_scale="Blues",
                title="Heatmap Korelasi Sensor"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Data mentah belum ditemukan di folder `data/predictive_maintenance.csv`.")

# =========================================================
# TAB 2: SIMULATOR PREDIKSI
# =========================================================
with tab2:
    st.header("Simulator Prediksi Anomali Mesin")
    st.write("Atur parameter kondisi sensor di bawah untuk menghitung probabilitas risiko kegagalan mesin.")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        air_temp = st.slider("Air Temperature [K]", 290.0, 310.0, 300.0, 0.1)
        process_temp = st.slider("Process Temperature [K]", 300.0, 325.0, 310.0, 0.1)
        rot_speed = st.slider("Rotational Speed [RPM]", 1000, 3000, 1500, 10)
        
    with col_input2:
        torque = st.slider("Torque [Nm]", 10.0, 90.0, 40.0, 0.5)
        tool_wear = st.slider("Tool Wear [min]", 0, 250, 100, 1)

    st.divider()

    input_data = pd.DataFrame([{
        'Air_temperature_K': air_temp,
        'Process_temperature_K': process_temp,
        'Rotational_speed_rpm': rot_speed,
        'Torque_Nm': torque,
        'Tool_wear_min': tool_wear
    }])

    if st.button("🔍 Hitung Risiko & Prediksi Mesin", type="primary", use_container_width=True):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risiko Failure (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e71d36" if probability > 50 else "#2ec4b6"},
                    'steps': [
                        {'range': [0, 40], 'color': "#c8e6c9"},
                        {'range': [40, 70], 'color': "#fff9c4"},
                        {'range': [70, 100], 'color': "#ffcdd2"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.subheader("Hasil Diagnosis AI:")
            if prediction == 1 or probability > 50:
                st.error("⚠️ POTENSI KERUSAKAN TERDETEKSI!")
                st.write(f"Tingkat Risiko: **{probability:.1f}%**")
                st.markdown(
                    "**Rekomendasi Tindakan:**\n"
                    "- Segera jadwalkan perawatan/penggantian komponen.\n"
                    "- Periksa sistem pendingin dan beban kerja mesin."
                )
            else:
                st.success("✅ KONDISI MESIN AMAN / NORMAL")
                st.write(f"Tingkat Risiko: **{probability:.1f}%**")

# =========================================================
# TAB 3: BATCH PREDICTION
# =========================================================
with tab3:
    st.header("Analisis Massal via File CSV")
    uploaded_file = st.file_uploader("Pilih file CSV sensor", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview Data Terunggah:", batch_df.head(5))
        
        req_cols = ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
        
        if all(col in batch_df.columns for col in req_cols):
            if st.button("🚀 Prediksi Seluruh Data CSV"):
                X_batch = batch_df[req_cols]
                X_batch_scaled = scaler.transform(X_batch)
                
                predictions = model.predict(X_batch_scaled)
                probabilities = model.predict_proba(X_batch_scaled)[:, 1] * 100
                
                batch_df['Predicted_Failure'] = predictions
                batch_df['Risk_Probability_%'] = np.round(probabilities, 2)
                
                st.success("✅ Prediksi Selesai!")
                st.dataframe(batch_df)
        else:
            st.error(f"Format CSV tidak sesuai! Pastikan memiliki kolom berikut: {req_cols}")