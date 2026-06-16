# Plant Disease Classification + Grad-CAM IoU Evaluation

Proyek ini membandingkan performa **GoogleNet** dan **DenseNet201** untuk deteksi penyakit tanaman, mencakup evaluasi akurasi klasifikasi dan keandalan spasial penjelasan visual Grad-CAM menggunakan metrik **Intersection over Union (IoU)** terhadap ground-truth bounding box dari dataset lapangan.

## Overview

Pipeline terdiri dari dua fase:

- **Fase 1** - Fine-tuning GoogleNet dan DenseNet201 pada 38 kelas dari dataset PlantVillage. Evaluasi mencakup akurasi, presisi, recall, dan F1-score.
- **Fase 2** - Ekstraksi heatmap Grad-CAM dari kedua model dan pengukuran keandalan lokalisasi spasial menggunakan IoU terhadap anotasi bounding box format YOLO dari dataset lapangan Roboflow.

## Requirements

```
torch>=2.6.0
torchvision
tensorflow>=2.0
opencv-python
grad-cam
scikit-learn
matplotlib
seaborn
numpy
Pillow
```

Install dependencies:

```bash
pip install grad-cam opencv-python scikit-learn seaborn matplotlib
```

## Dataset

### PlantVillage (Fase 1)

Download dari [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease). Dataset berisi 54.305 gambar dari 38 kelas penyakit tanaman tomat, kentang, dan paprika.

### Roboflow Bounding Box Dataset (Fase 2)

Download dataset beranotasi dalam format YOLOv8 dari Roboflow. Dibutuhkan 1.500 gambar dengan file label `.txt` berisi koordinat bounding box yang sudah dinormalisasi.

## Cara Penggunaan

Jalankan notebook secara berurutan:

```
plant_disease_full_pipeline.ipynb
```

**Fase 1** - Preprocessing, dataset split, training dan evaluasi GoogleNet dan DenseNet201 menggunakan PyTorch dengan backbone frozen dan custom classifier head.

**Fase 2** - Ekstraksi Grad-CAM menggunakan forward/backward hook manual. IoU dihitung antara heatmap yang dibinarisasi (threshold = 0.5) dengan ground-truth bounding box dari Roboflow pada 300 gambar evaluasi.

## Hyperparameter

| Parameter | Nilai |
|---|---|
| Ukuran gambar | 224 x 224 |
| Batch size | 50 |
| Epochs | 10 |
| Learning rate | 0.001 |
| LR scheduler | Step decay |
| Dropout | 0.25 |
| Split data | 70% / 20% / 10% |

## Preprocessing

1. Konversi RGB ke ruang warna LAB
2. Otsu's thresholding pada L-channel
3. Morphological Closing dan Opening dengan kernel elips 5x5 piksel
4. Bitwise AND mask terhadap gambar asli
5. Resize ke 224x224 piksel

## Hardware

Diuji menggunakan PyTorch 2.6.0+cu118 dengan CUDA. Waktu training sekitar 5.400-5.700 detik per model untuk 10 epoch.

#
