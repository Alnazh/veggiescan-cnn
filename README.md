# VeggieLens - Klasifikasi Sayuran Berbasis CNN

Sistem klasifikasi jenis sayuran menggunakan **Convolutional Neural Network (CNN)**
berbasis arsitektur MobileNetV2 (Transfer Learning).
Dibangun dengan Flask, TensorFlow/Keras, dan Bootstrap 5.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)

---

## 🎮 Demo
🔗 **Live:** `https://alnazh-veggielens-cnn.hf.space`
📂 **GitHub:** `https://github.com/Alnazh/veggielens-cnn`

---

## ✨ Fitur Utama

- 🔍 Klasifikasi 15 jenis sayuran dari foto (upload drag & drop)
- 📊 Top-3 prediksi dengan confidence score secara real-time
- 📈 Dashboard statistik: distribusi prediksi & tren harian (Chart.js)
- 📱 Tampilan responsif desktop & mobile
- 🚀 Inference cepat dengan model MobileNetV2 pre-trained

## 🥬 15 Kelas Sayuran

| # | Nama | Nama Dataset |
|---|------|-------------|
| 00 | Buncis | Bean |
| 01 | Pare | Bitter_Gourd |
| 02 | Labu Air | Bottle_Gourd |
| 03 | Terong | Brinjal |
| 04 | Brokoli | Broccoli |
| 05 | Kubis | Cabbage |
| 06 | Paprika | Capsicum |
| 07 | Wortel | Carrot |
| 08 | Kembang Kol | Cauliflower |
| 09 | Mentimun | Cucumber |
| 10 | Pepaya Muda | Papaya |
| 11 | Kentang | Potato |
| 12 | Labu Kuning | Pumpkin |
| 13 | Lobak | Radish |
| 14 | Tomat | Tomato |

## 🛠️ Stack Teknologi

- **Backend**: Python 3.11, Flask 3.0, Gunicorn
- **ML**: TensorFlow 2.15, Keras, MobileNetV2 (Transfer Learning), NumPy, Pillow
- **Frontend**: HTML5, CSS3, Bootstrap 5.3, Chart.js
- **Deployment**: Hugging Face Spaces / Railway

## ⚡ Cara Menjalankan (Lokal)

### 1. Clone repositori
```bash
git clone https://github.com/Alnazh/veggielens-cnn.git
cd veggielens-cnn
```

### 2. Install dependensi
```bash
pip install -r requirements.txt
```

### 3. Pastikan model tersedia
Letakkan file model di folder `model/`:
```
model/
├── vegetable_model.keras
└── class_indices.json
```

### 4. Jalankan aplikasi
```bash
python app.py
```

Buka browser: `http://localhost:5000`

## 📁 Struktur Folder

```
veggielens/
├── app.py               # Aplikasi Flask utama + inference CNN
├── requirements.txt     # Dependensi Python
├── Procfile             # Konfigurasi deployment
├── README.md
├── model/
│   ├── vegetable_model.keras   # Model CNN hasil training
│   └── class_indices.json      # Mapping indeks ke nama kelas
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── dashboard.html
│   ├── about.html
│   ├── 404.html
│   └── 500.html
├── static/
│   ├── css/custom.css
│   ├── js/main.js
│   └── uploads/         # Folder sementara upload (auto-hapus)
└── prediction_log.json  # Log prediksi (dibuat otomatis)
```

## 🧠 Arsitektur Model CNN

| Komponen | Detail |
|---|---|
| Base Model | MobileNetV2 (Transfer Learning) |
| Input Size | 128 × 128 × 3 (RGB) |
| Preprocessing | `mobilenet_v2.preprocess_input` (skala -1 s/d 1) |
| Output Layer | Dense 15 + Softmax |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Akurasi Validasi | ~99% |
| Jumlah Epoch | 15 (dengan fine-tuning di epoch ke-7) |

## 📊 Dataset

**Vegetable Image Dataset** - Kaggle (Misrak Ahmed, 2021)
🔗 https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset

- 21.000+ gambar, 15 kelas, resolusi 224×224
- Distribusi seimbang ~1.400 gambar per kelas
- Lisensi: CC BY 4.0

---

## Screenshots

### Tampilan Aplikasi
**Dashboard**
![Dashboard](static/img/dashboard.png)
**Klasifikasi**
![Klasifikasi](static/img/klasifikasi.png)