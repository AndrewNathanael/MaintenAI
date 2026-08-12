import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

def train():
    print("🚀 Memulai Proses Pelatihan Model Predictive Maintenance...")
    
    data_path = "data/predictive_maintenance.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: File {data_path} tidak ditemukan!")
        print("💡 Harap pastikan file CSV Kaggle berada di folder 'data/predictive_maintenance.csv'")
        return

    # 1. Membaca CSV
    print(f"📁 Membaca dataset dari: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Pemetaan Nama Kolom Kaggle/UCI ke Format Internal
    column_mapping = {
        'Air temperature [K]': 'Air_temperature_K',
        'Process temperature [K]': 'Process_temperature_K',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Torque [Nm]': 'Torque_Nm',
        'Tool wear [min]': 'Tool_wear_min',
        'Target': 'Machine_failure',
        'Machine failure': 'Machine_failure'
    }

    # Mengubah nama kolom otomatis jika ada
    df = df.rename(columns=column_mapping)

    # 3. Menentukan Fitur dan Target
    feature_cols = [
        'Air_temperature_K', 
        'Process_temperature_K', 
        'Rotational_speed_rpm', 
        'Torque_Nm', 
        'Tool_wear_min'
    ]
    target_col = 'Machine_failure'

    # Validasi apakah semua kolom fitur tersedia
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Kolom berikut tidak ditemukan di CSV: {missing_cols}")
        print(f"📋 Kolom yang tersedia di CSV Anda adalah: {list(df.columns)}")
        return

    X = df[feature_cols]
    y = df[target_col]

    print(f"✅ Data berhasil diproses. Total {len(df)} baris data.")

    # 4. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Scaling Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Melatih Model Random Forest
    print("⏳ Melatih model Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # 7. Evaluasi Model
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    print("\n✅ Pelatihan Model Selesai!")
    print(f"🎯 F1-Score pada Test Set: {f1:.4f}")
    print("\n--- Laporan Klasifikasi ---")
    print(classification_report(y_test, y_pred))

    # 8. Simpan Model & Scaler
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    print("\n💾 Model (model/model.pkl) dan Scaler (model/scaler.pkl) berhasil disimpan!")

if __name__ == "__main__":
    train()