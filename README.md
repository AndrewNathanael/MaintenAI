# ⚙️ MaintenAI — Industrial IoT Predictive Maintenance Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PredictPulse AI** adalah platform analitik dan prediksi risiko kerusakan mesin industri berbasis Machine Learning & IoT. Platform ini memungkinkan tim *maintenance* melakukan pemantauan sensor secara *real-time*, mensimulasikan kondisi kegagalan, dan melakukan prediksi massal melalui antarmuka interaktif **Streamlit**.

---

## 📊 Alur Kerja Sistem (Flowchart)

Berikut adalah diagram alir (*flowchart*) arsitektur end-to-end dari pengolahan data sensor hingga prediksi pada dashboard:

```mermaid
flowchart TD
    %% Node Data & Ingestion
    A[📡 Sensor IoT Mesin / Dataset Kaggle] -->|Data Mentah CSV| B[🛠️ Data Preprocessing & Scaling]
    
    %% Node ML Pipeline
    subgraph ML_Pipeline [Pipeline Machine Learning]
        B --> C[📊 Feature Selection & Standardization]
        C --> D[🌲 Model Training: Random Forest Classifier]
        D --> E[⚡ Model Quantization: INT8 PTQ ONNX]
        E --> F[💾 Export Artefak: model.pkl & scaler.pkl]
    end
    
    %% Node Streamlit Application
    subgraph Streamlit_App [Aplikasi Streamlit Dashboard]
        F --> G[🚀 Load ML Assets ke Memory]
        
        %% User Interaction Nodes
        G --> H1[📊 Tab 1: Dashboard EDA & Heatmap]
        G --> H2[🔮 Tab 2: Live Simulator Sensor]
        G --> H3[📁 Tab 3: Batch CSV Prediction]
        
        %% Real-time Inference & Output
        H2 -->|Slider Input| I[🧮 Real-Time Inference Engine]
        H3 -->|Upload CSV| I
        
        I --> J{Status Machine Failure?}
        J -->|Risiko > 50%| K[⚠️ Alert Kerusakan & Gauge Meter Merah]
        J -->|Risiko ≤ 50%| L[✅ Status Mesin Aman & Gauge Meter Hijau]
    end

