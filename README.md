# Deteksi Atribut Seragam Sekolah (School Uniform Attribute Detection)

Aplikasi computer vision yang mendeteksi kelengkapan atribut seragam sekolah secara real-time menggunakan webcam. Menggunakan HOG (Histogram of Oriented Gradients) sebagai ekstraksi fitur dan SVM (Support Vector Machine) sebagai classifier, dengan pendekatan sliding window.

## Atribut yang Dideteksi
- **Dasi** (Tie)
- **Logo Sekolah** (School Badge)
- **Topi** (Hat)
- **Ikat Pinggang** (Belt)

---

## Cara Kerja

1. **Preprocessing** — setiap crop objek di-resize ke 96×96, lalu dinormalisasi pencahayaan dengan CLAHE sebelum diekstrak fitur HOG-nya.
2. **Training** — SVM kernel linear ditraining dengan fitur HOG dari crop objek + background patch (grid-based sampling). Data augmentasi 6 jenis diterapkan 5× per crop untuk memperkaya variasi data. Hard Negative Mining dijalankan sebagai pass kedua untuk mengurangi false positive.
3. **Hyperparameter Tuning** — nilai `C` terbaik dipilih berdasarkan macro F1-score di folder `valid/` (bukan `test/`) agar evaluasi final tetap jujur.
4. **Deteksi** — frame webcam di-resize ke resolusi 320×240 lalu di-scan dengan sliding window 96×96 (step 64); tiap window diklasifikasi SVM, difilter confidence threshold dan zona vertikal, lalu dibersihkan dengan NMS (threshold 0.3) sebelum ditampilkan.

---

## Hasil Evaluasi (Test Set)

| Kelas | Precision | Recall | F1-Score | AP |
|-------|-----------|--------|----------|----|
| Dasi | 0.72 | 0.75 | 0.73 | 82.63% |
| Logo Sekolah | 0.83 | 0.86 | 0.84 | 90.09% |
| Topi | 0.70 | 0.95 | 0.81 | 93.00% |
| Ikat Pinggang | 0.35 | 0.50 | 0.41 | 58.31% |
| Background | 0.92 | 0.73 | 0.81 | 94.50% |
| **Macro avg** | **0.70** | **0.76** | **0.72** | — |
| **Weighted avg** | **0.81** | **0.78** | **0.78** | — |
| **mAP** | | | | **83.71%** |

**Average Precision (AP) per kelas:**
- Dasi            : 82.63%
- Logo Sekolah    : 90.09%
- Topi            : 93.00%
- Ikat Pinggang   : 58.31%
- Background      : 94.50%

**=> mAP (Mean Average Precision) : 83.71%**

> **Catatan:** AP (Average Precision) dihitung dari area di bawah kurva precision-recall di semua threshold confidence, sehingga nilainya bisa lebih tinggi dari F1-score yang dihitung pada satu titik threshold saja. mAP adalah rata-rata AP dari kelima kelas.

---

## Requirements

- Python 3.9–3.11
- Webcam
- Jupyter Notebook / JupyterLab / VS Code dengan ekstensi Jupyter
- Package:
  ```
  opencv-python
  opencv-contrib-python
  numpy
  scikit-image
  scikit-learn
  matplotlib
  ```

> Semua package lain yang dipakai (`pickle`, `os`, `random`, `time`, `xml.etree`, `json`, `datetime`) sudah tersedia sebagai bagian dari Python standard library.

---

## Installation

1. Clone repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Buat dan aktifkan virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install opencv-python opencv-contrib-python numpy scikit-image scikit-learn matplotlib
   ```

---

## Dataset

Menggunakan dataset **Deteksi Seragam** dari Roboflow dalam format Pascal VOC XML.

**Dataset Link:** [Deteksi Seragam Dataset on Roboflow](https://universe.roboflow.com/skripsi-gztjx/deteksi-seragam/dataset/1)

### Cara Download

1. Buka link dataset di atas, login/daftar Roboflow
2. Klik **Download Dataset** → pilih format **Pascal VOC**
3. Extract zip yang didownload
4. Pindahkan folder `train/`, `valid/`, dan `test/` ke root folder projek ini

### Struktur Dataset

| Folder | Jumlah Gambar | Keterangan |
|--------|--------------|------------|
| `train/` | 495 gambar | Untuk training model |
| `valid/` | 50 gambar | Untuk hyperparameter tuning |
| `test/` | 20 gambar | Untuk evaluasi final |

Semua gambar berukuran 640×640px, format JPG, annotasi Pascal VOC XML.

---

## Project Structure

```
.
├── main.ipynb                 # Notebook utama: training, evaluasi, dan deteksi webcam
├── svm_seragam_model.pkl      # Model SVM tersimpan (auto-generated, git-ignored)
├── confusion_matrix.png       # Confusion matrix hasil evaluasi (auto-generated)
├── evaluation_results.json    # Hasil evaluasi dalam format JSON (auto-generated)
├── train/                     # Dataset training (git-ignored)
├── valid/                     # Dataset validasi (git-ignored)
├── test/                      # Dataset test (git-ignored)
├── .gitignore
└── README.md
```

---

## Struktur Notebook (main.ipynb)

| Cell | Isi |
|------|-----|
| 1 | Import Library |
| 2 | Konfigurasi Global |
| 3 | Preprocessing & Feature Extraction (CLAHE + HOG) |
| 4 | Augmentasi Data (6 jenis, 5× per crop) |
| 5 | Dataset Loader (Grid-based Background Sampling) |
| 6 | Training + Hyperparameter Tuning via `valid/` |
| 7 | Hard Negative Mining (2-pass training) |
| 8 | Evaluasi Model pada Test Set |
| 9 | Fungsi Deteksi (Sliding Window + NMS) |
| 10 | Real-time Webcam Detection |

---

## Usage

Buka dan jalankan `main.ipynb` secara berurutan dari cell 1 hingga cell 10.

### Yang Terjadi Saat Run

- **Cell 1–5** — definisi fungsi dan konfigurasi (< 1 menit)
- **Cell 6** — training model + hyperparameter tuning (~45–60 menit, hanya jika `.pkl` belum ada)
- **Cell 7** — Hard Negative Mining + retrain (~15–20 menit, hanya jika training baru)
- **Cell 8** — evaluasi di `test/`, tampilkan confusion matrix, simpan `evaluation_results.json`
- **Cell 9** — definisi fungsi deteksi
- **Cell 10** — webcam aktif, tekan **`q`** untuk keluar

> Jika `svm_seragam_model.pkl` sudah ada, cell 6 dan 7 otomatis di-skip langsung load model.

---

## Tips Penggunaan Webcam

- Jarak optimal dari kamera: **1–1.5 meter**
- Pastikan pencahayaan cukup (CLAHE membantu tapi tidak bisa mengatasi gelap total)
- Posisikan tubuh sehingga kepala ada di bagian atas frame untuk deteksi Topi

---

## Notes & Limitations

- Untuk retrain dari awal: hapus `svm_seragam_model.pkl` lalu jalankan ulang dari cell 1
- Target FPS: ~10 FPS (HOG+SVM tidak bisa memanfaatkan GPU)
- Ikat Pinggang paling sulit dideteksi karena sering tertutup dan datanya paling sedikit
- Pendekatan HOG+SVM lebih lambat dan kurang akurat dibanding deep learning (YOLO/MobileNet), tapi lebih mudah dijelaskan secara akademis
