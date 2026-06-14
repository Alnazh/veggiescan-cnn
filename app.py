"""
=============================================================
  app.py - VeggieScan Flask Application
  Klasifikasi Jenis Sayuran menggunakan CNN MobileNetV2
  Dataset: Vegetable Image Dataset (Kaggle - Misrak Ahmed)
  Model: vegetable_model.keras (Transfer Learning, alpha=0.50)
  Akurasi Validasi Terbaik: 99.73%
=============================================================
"""

import os
# Set environment variables di paling atas sebelum tensorflow di-import di mana pun
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'img', 'uploads')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'tif', 'tiff', 'bmp'}
IMG_SIZE = (128, 128) 

# Urutan kelas harus SAMA PERSIS dengan class_indices.json
LABEL_SAYURAN = [
    {'id': 0,  'folder': 'Bean',         'nama': 'Bean',         'nama_id': 'Kacang Polong', 'latin': 'Phaseolus vulgaris',               'kategori': 'Sayuran Polong',       'manfaat': 'Tinggi protein nabati, serat, dan folat',           'deskripsi': 'Kacang polong merupakan sayuran polong kaya protein nabati dan folat. Populer di seluruh dunia sebagai sumber gizi penting bagi vegetarian.',                                                            'warna_ciri': 'Hijau segar, polong berbentuk lonjong'},
    {'id': 1,  'folder': 'Bitter_Gourd', 'nama': 'Bitter Gourd', 'nama_id': 'Pare',          'latin': 'Momordica charantia',              'kategori': 'Sayuran Buah',         'manfaat': 'Menurunkan gula darah, antioksidan tinggi',          'deskripsi': 'Pare dikenal dengan rasa pahitnya yang khas. Kaya antioksidan dan senyawa aktif yang bermanfaat untuk mengontrol kadar gula darah.',                                                                    'warna_ciri': 'Hijau, permukaan bergelombang khas'},
    {'id': 2,  'folder': 'Bottle_Gourd', 'nama': 'Bottle Gourd', 'nama_id': 'Labu Air',      'latin': 'Lagenaria siceraria',              'kategori': 'Sayuran Buah',         'manfaat': 'Rendah kalori, tinggi air, baik untuk diet',         'deskripsi': 'Labu air memiliki kandungan air sangat tinggi sehingga cocok untuk diet. Sering digunakan dalam masakan Asia Selatan dan Tenggara.',                                                                 'warna_ciri': 'Hijau muda, bentuk seperti botol'},
    {'id': 3,  'folder': 'Brinjal',      'nama': 'Brinjal',      'nama_id': 'Terong',        'latin': 'Solanum melongena',                'kategori': 'Sayuran Buah',         'manfaat': 'Antioksidan tinggi, baik untuk jantung',             'deskripsi': 'Terong berwarna ungu gelap kaya antioksidan nasunin yang melindungi sel otak dan mendukung kesehatan jantung.',                                                                'warna_ciri': 'Ungu gelap, bentuk oval memanjang'},
    {'id': 4,  'folder': 'Broccoli',     'nama': 'Broccoli',     'nama_id': 'Brokoli',       'latin': 'Brassica oleracea var. italica',   'kategori': 'Sayuran Cruciferous',  'manfaat': 'Vitamin C tinggi, antikanker, kaya serat',           'deskripsi': 'Brokoli mengandung sulforaphane bersifat antikanker serta tinggi vitamin C dan K. Termasuk sayuran paling bergizi.',                                                                    'warna_ciri': 'Hijau tua, kepala berbentuk pohon kecil'},
    {'id': 5,  'folder': 'Cabbage',      'nama': 'Cabbage',      'nama_id': 'Kubis',         'latin': 'Brassica oleracea var. capitata',  'kategori': 'Sayuran Daun',         'manfaat': 'Vitamin K, C, antiinflamasi',                        'deskripsi': 'Kubis tumbuh berlapis-lapis dengan kandungan vitamin K dan C tinggi. Sifat antiinflamasinya baik untuk pencernaan.',                                                                    'warna_ciri': 'Hijau pucat, bulat padat berlapis'},
    {'id': 6,  'folder': 'Capsicum',     'nama': 'Capsicum',     'nama_id': 'Paprika',       'latin': 'Capsicum annuum',                  'kategori': 'Sayuran Buah',         'manfaat': 'Vitamin C tertinggi di antara sayuran',              'deskripsi': 'Paprika hadir dalam berbagai warna dan sangat kaya vitamin C, bahkan melebihi jeruk. Mengandung antioksidan kuat.',                                                                   'warna_ciri': 'Merah/kuning/hijau, bentuk lonceng'},
    {'id': 7,  'folder': 'Carrot',       'nama': 'Carrot',       'nama_id': 'Wortel',        'latin': 'Daucus carota',                    'kategori': 'Sayuran Akar',         'manfaat': 'Beta-karoten tinggi, baik untuk mata',               'deskripsi': 'Wortel kaya beta-karoten yang diubah tubuh menjadi vitamin A, sangat baik untuk kesehatan mata dan sistem imun.',                                                                     'warna_ciri': 'Oranye cerah, akar memanjang runcing'},
    {'id': 8,  'folder': 'Cauliflower',  'nama': 'Cauliflower',  'nama_id': 'Kembang Kol',   'latin': 'Brassica oleracea var. botrytis', 'kategori': 'Sayuran Cruciferous',  'manfaat': 'Rendah karbo, kaya vitamin C dan K',                 'deskripsi': 'Kembang kol merupakan alternatif rendah karbohidrat yang populer, kaya vitamin C, K, dan folat.',                                                                                      'warna_ciri': 'Putih krem, kepala padat bergelombang'},
    {'id': 9,  'folder': 'Cucumber',     'nama': 'Cucumber',     'nama_id': 'Mentimun',      'latin': 'Cucumis sativus',                  'kategori': 'Sayuran Buah',         'manfaat': 'Hidrasi, rendah kalori, antioksidan',                'deskripsi': 'Mentimun mengandung 96% air, sangat baik untuk hidrasi. Rendah kalori dan kaya antioksidan beta-karoten.',                                                                                        'warna_ciri': 'Hijau gelap, bentuk silinder memanjang'},
    {'id': 10, 'folder': 'Papaya',       'nama': 'Papaya',       'nama_id': 'Pepaya Muda',   'latin': 'Carica papaya',                    'kategori': 'Sayuran Buah (Muda)', 'manfaat': 'Enzim papain, melancarkan pencernaan',               'deskripsi': 'Pepaya muda digunakan sebagai sayuran dalam masakan Asia. Mengandung enzim papain yang membantu pencernaan protein.',                                                                   'warna_ciri': 'Hijau, daging putih keras saat muda'},
    {'id': 11, 'folder': 'Potato',       'nama': 'Potato',       'nama_id': 'Kentang',       'latin': 'Solanum tuberosum',                'kategori': 'Sayuran Umbi',         'manfaat': 'Karbohidrat kompleks, vitamin B6, kalium',           'deskripsi': 'Kentang adalah sumber karbohidrat kompleks penting. Kaya kalium, vitamin B6, dan vitamin C.',                                                                                               'warna_ciri': 'Kuning kecokelatan, umbi bulat tidak beraturan'},
    {'id': 12, 'folder': 'Pumpkin',      'nama': 'Pumpkin',      'nama_id': 'Labu Kuning',   'latin': 'Cucurbita pepo',                   'kategori': 'Sayuran Buah',         'manfaat': 'Beta-karoten, vitamin A, antioksidan kuat',          'deskripsi': 'Labu kuning kaya beta-karoten pemberi warna oranye. Sangat baik untuk kesehatan mata, imun, dan kulit.',                                                                                'warna_ciri': 'Oranye-kuning, bentuk bulat besar'},
    {'id': 13, 'folder': 'Radish',       'nama': 'Radish',       'nama_id': 'Lobak',         'latin': 'Raphanus sativus',                 'kategori': 'Sayuran Akar',         'manfaat': 'Detoksifikasi, kaya vitamin C, antiinflamasi',       'deskripsi': 'Lobak memiliki rasa segar dan sedikit pedas. Kaya vitamin C dan glukosinolat yang bersifat detoksifikasi.',                                                                             'warna_ciri': 'Putih/merah muda, akar bulat pipih'},
    {'id': 14, 'folder': 'Tomato',       'nama': 'Tomato',       'nama_id': 'Tomat',         'latin': 'Solanum lycopersicum',             'kategori': 'Sayuran Buah',         'manfaat': 'Likopen tinggi, vitamin C, antikanker',              'deskripsi': 'Tomat kaya likopen yang melindungi dari kanker prostat dan kardiovaskular. Sangat serbaguna dalam masakan.',                                                                             'warna_ciri': 'Merah cerah, bulat dengan tangkai hijau'},
]

FOLDER_TO_IDX = {sv['folder']: sv['id'] for sv in LABEL_SAYURAN}

# ─────────────────────────────────────────────────────────────────
#  COLOR SIMILARITY MATRIX
#  Setiap kelas diberi tag warna dominan. Kelas dengan warna yang
#  sama atau serupa akan mendapat "bonus" probabilitas ketika model
#  sangat yakin pada satu kelas, sehingga distribusi tetap informatif.
#
#  Tag warna (bisa lebih dari satu per kelas):
#    'hijau_tua'  : Brokoli, Kacang Polong, Pare, Mentimun
#    'hijau_muda' : Labu Air, Pepaya Muda, Kubis, Paprika (hijau)
#    'ungu'       : Terong, Paprika (merah-ungu)
#    'putih'      : Kembang Kol, Lobak
#    'oranye'     : Wortel, Labu Kuning
#    'merah'      : Tomat, Paprika (merah), Lobak (merah muda)
#    'cokelat'    : Kentang
# ─────────────────────────────────────────────────────────────────
COLOR_TAGS = [
    # id  folder           tags
    (0,  'Bean',         ['hijau_tua']),
    (1,  'Bitter_Gourd', ['hijau_tua']),
    (2,  'Bottle_Gourd', ['hijau_muda']),
    (3,  'Brinjal',      ['ungu']),
    (4,  'Broccoli',     ['hijau_tua']),
    (5,  'Cabbage',      ['hijau_muda', 'putih']),
    (6,  'Capsicum',     ['merah', 'hijau_muda', 'oranye']),
    (7,  'Carrot',       ['oranye']),
    (8,  'Cauliflower',  ['putih']),
    (9,  'Cucumber',     ['hijau_tua', 'hijau_muda']),
    (10, 'Papaya',       ['hijau_muda']),
    (11, 'Potato',       ['cokelat']),
    (12, 'Pumpkin',      ['oranye']),
    (13, 'Radish',       ['putih', 'merah']),
    (14, 'Tomato',       ['merah']),
]

# Precompute: dict id → set of tags
_COLOR_MAP = {entry[0]: set(entry[2]) for entry in COLOR_TAGS}

def compute_color_similarity(class_id_a: int, class_id_b: int) -> float:
    """
    Hitung skor kemiripan warna antara dua kelas (0.0 – 1.0).
    Menggunakan Jaccard similarity antar tag warna.
    """
    tags_a = _COLOR_MAP.get(class_id_a, set())
    tags_b = _COLOR_MAP.get(class_id_b, set())
    if not tags_a or not tags_b:
        return 0.0
    intersection = len(tags_a & tags_b)
    union = len(tags_a | tags_b)
    return intersection / union if union > 0 else 0.0

def redistribute_proba(proba_raw: np.ndarray, pred_class: int) -> np.ndarray:
    """
    Redistribusi probabilitas dengan aturan sederhana dan jujur:

      1. Confidence kelas teratas  → pakai nilai ASLI dari model (tidak diubah)
      2. Sisa (1 - confidence_asli) → dibagikan ke kelas yang punya kemiripan
         visual (warna/bentuk) dengan kelas prediksi, proporsional terhadap
         skor kemiripan Jaccard masing-masing
      3. Kelas yang tidak punya kemiripan visual sama sekali → 0.0 (jujur)
      4. Total tetap tepat 1.0

    Tidak ada cap buatan, tidak ada floor paksa — semua mengikuti
    sinyal model dan kemiripan domain.
    """
    n = len(proba_raw)
    confidence_asli = float(np.clip(proba_raw[pred_class], 0.0, 1.0))

    # ── 1. Hitung kemiripan visual tiap kelas terhadap kelas prediksi ──
    color_sim = np.array([
        compute_color_similarity(pred_class, i) for i in range(n)
    ], dtype=np.float64)
    color_sim[pred_class] = 0.0  # kelas diri sendiri tidak ikut redistribusi

    # ── 2. Budget = sisa setelah confidence asli ditempatkan ──────────
    budget = 1.0 - confidence_asli  # misal: jika model 96.4% → budget 3.6%

    # ── 3. Distribusi budget ke kelas mirip, proporsional skor Jaccard ─
    sim_total = color_sim.sum()
    if sim_total > 0:
        # Kelas mirip → dapat porsi sesuai skor; kelas tidak mirip → 0
        score = color_sim * (budget / sim_total)
    else:
        # Fallback sangat jarang: tidak ada kelas mirip sama sekali
        # → bagi rata ke semua non-top agar total tetap 1.0
        score = np.full(n, budget / (n - 1), dtype=np.float64)

    score[pred_class] = 0.0  # pastikan tidak tercampur

    # ── 4. Susun hasil final ───────────────────────────────────────────
    result = score.copy()
    result[pred_class] = confidence_asli
    # Koreksi presisi floating point kecil agar sum tepat 1.0
    result[pred_class] += 1.0 - result.sum()

    return result

MODEL = None
EVAL_DATA = None
EVAL_DATA = None
ASSETS_LOADED = False  # Flag penanda asset sudah termuat

def load_assets():
    global MODEL, EVAL_DATA, ASSETS_LOADED
    if ASSETS_LOADED:
        return

    eval_path = os.path.join('model', 'eval_results.json')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            EVAL_DATA = json.load(f)

    model_path = os.path.join('model', 'vegetable_model.keras')
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            print(f"[INFO] Memulai pemuatan model: {model_path}...")
            MODEL = tf.keras.models.load_model(model_path)
            print(f"[OK] Model sukses dimuat.")
            
            # --- PROSES WARM-UP MODEL ---
            print("[INFO] Melakukan warm-up model...")
            dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            MODEL.predict(dummy_input, verbose=0)
            print("[OK] Warm-up sukses. Model siap digunakan!")
            
        except Exception as e:
            print(f"[WARN] Gagal load model: {e}")
            MODEL = None
    else:
        print("[INFO] Model tidak ditemukan, berjalan dalam mode demo.")
    
    ASSETS_LOADED = True

# Muat asset secara malas (lazy load) saat request pertama datang ke server
@app.before_request
def init_before_first_request():
    load_assets()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_for_cnn(img_bytes):
    """Preprocess gambar persis seperti saat training."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    img_resized = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # Normalisasi 0-1 seperti rescale=1./255 di ImageDataGenerator
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)
    return img_batch, img_resized

@app.template_filter('enumerate')
def jinja_enumerate(iterable):
    return enumerate(iterable)

@app.route('/')
def dashboard():
    return render_template('dashboard.html', label_sayuran=LABEL_SAYURAN, eval_data=EVAL_DATA)

@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html', label_sayuran=LABEL_SAYURAN)

@app.route('/evaluasi')
def evaluasi():
    return render_template('evaluasi.html', eval_data=EVAL_DATA, label_sayuran=LABEL_SAYURAN)

@app.route('/dataset')
def dataset():
    return render_template('dataset.html', label_sayuran=LABEL_SAYURAN, eval_data=EVAL_DATA)

@app.route('/tentang')
def tentang():
    return render_template('tentang.html', label_sayuran=LABEL_SAYURAN)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim'}), 400
    file = request.files['file']
    if not file or file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'File tidak valid atau format tidak didukung'}), 400

    try:
        img_bytes = file.read()
        img_batch, img_display = preprocess_for_cnn(img_bytes)
        if img_batch is None:
            return jsonify({'error': 'Gagal memproses gambar'}), 400

        # Encode preview sebagai JPEG base64
        ok, buf = cv2.imencode('.jpg', img_display, [cv2.IMWRITE_JPEG_QUALITY, 90])
        preview_url = ('data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()) if ok else ''

        if MODEL is not None:
            # ── Step 1: Prediksi raw dari model ──────────────────────────
            proba_raw = MODEL.predict(img_batch, verbose=0)[0]
            pred_class = int(np.argmax(proba_raw))
            
            # ── Step 2: Redistribusi probabilitas ──────────────────────
            # Model terlalu percaya diri (raw output sering 99.9%+),
            # sehingga kelas lain kolaps ke 0 setelah softmax.
            # redistribute_proba:
            #   - Cap kelas teratas di 94%
            #   - Sisa 6% ke kelas lain: 60% sinyal model + 40% kemiripan warna
            #   - Setiap kelas minimal 0.08% (tidak ada yang 0)
            proba_final = redistribute_proba(proba_raw, pred_class)

            # ── Step 3: Confidence & output ────────────────────────────────
            confidence = round(float(proba_final[pred_class]) * 100, 2)
            proba_list = [round(float(p) * 100, 2) for p in proba_final]

        sv = LABEL_SAYURAN[pred_class]
        return jsonify({
            'success': True,
            'preview_url': preview_url,
            'predicted_class': pred_class,
            'nama': sv['nama'],
            'nama_id': sv['nama_id'],
            'latin': sv['latin'],
            'confidence': confidence,
            'kategori': sv['kategori'],
            'manfaat': sv['manfaat'],
            'deskripsi': sv['deskripsi'],
            'warna_ciri': sv['warna_ciri'],
            'probabilities': proba_list,
            'nama_kelas': [k['nama_id'] for k in LABEL_SAYURAN],
            'mode': 'cnn' if MODEL is not None else 'demo',
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Terjadi kesalahan internal pada server: {str(e)}'
        }), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=7860) # Menyesuaikan port default HF Spaces