import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# ============================================================
# KONFIGURASI - Dioptimalkan untuk training CPU (PC kampus)
# ============================================================
DATASET_PATH = "Vegetable Images"
IMG_SIZE     = (128, 128)   # diperkecil dari 224x224, cukup untuk akurasi tinggi
BATCH_SIZE   = 32
EPOCHS       = 10           # fase 1 (base frozen)
FINE_TUNE_EPOCHS = 8         # fase 2 (fine-tuning)
# Tidak ada steps_per_epoch -> seluruh dataset dipakai tiap epoch (penting untuk akurasi >=90%)
NUM_CLASSES  = 15
MODEL_OUTPUT = "vegetable_model.keras"
HISTORY_OUTPUT = "training_history.json"

CLASS_NAMES = [
    'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
]

print("=" * 60)
print("  VEGETABLE IMAGE CLASSIFICATION - CNN TRAINING (CPU, akurasi >=90%)")
print("=" * 60)
print(f"TensorFlow version : {tf.__version__}")
print(f"Dataset path       : {DATASET_PATH}")
print(f"Image size         : {IMG_SIZE}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Epochs (fase 1)    : {EPOCHS}")
print(f"Epochs (fase 2)    : {FINE_TUNE_EPOCHS}")
print(f"Num classes        : {NUM_CLASSES}")
print("=" * 60)

# ============================================================
# CEK DATASET
# ============================================================
train_dir = os.path.join(DATASET_PATH, "train")
val_dir   = os.path.join(DATASET_PATH, "validation")
test_dir  = os.path.join(DATASET_PATH, "test")

for d in [train_dir, val_dir, test_dir]:
    if not os.path.exists(d):
        raise FileNotFoundError(
            f"\n[ERROR] Folder tidak ditemukan: {d}\n"
            f"Pastikan dataset sudah diekstrak dan DATASET_PATH sudah benar.\n"
            f"Struktur yang diharapkan:\n"
            f"  {DATASET_PATH}/\n"
            f"    train/\n"
            f"    validation/\n"
            f"    test/\n"
        )

print("\n[OK] Semua folder dataset ditemukan.")

# ============================================================
# DATA GENERATOR (augmentasi ringan agar CPU tidak terbebani)
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\n[INFO] Train samples      : {train_generator.samples}")
print(f"[INFO] Validation samples : {val_generator.samples}")
print(f"[INFO] Test samples       : {test_generator.samples}")
print(f"[INFO] Class indices      : {train_generator.class_indices}")

with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f, indent=2)
print("\n[OK] class_indices.json disimpan.")

# ============================================================
# ARSITEKTUR MODEL CNN (Transfer Learning - MobileNetV2 alpha=0.50)
# ============================================================
print("\n[INFO] Membangun model dengan MobileNetV2 alpha=0.50 (ringan tapi akurat)...")

base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    alpha=0.50,              # cukup ringan untuk CPU, akurasi jauh lebih baik dari 0.35
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    ModelCheckpoint(
        MODEL_OUTPUT,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================
# FASE 1: TRAINING (base frozen)
# ============================================================
print("\n" + "=" * 60)
print("  FASE 1: Training dengan base model frozen")
print("=" * 60)

history1 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# FASE 2: FINE-TUNING (unfreeze sebagian layer atas)
# ============================================================
print("\n" + "=" * 60)
print("  FASE 2: Fine-tuning (unfreeze 20 layer terakhir)")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# EVALUASI PADA DATA TEST
# ============================================================
print("\n" + "=" * 60)
print("  EVALUASI MODEL PADA DATA TEST")
print("=" * 60)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\n  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc * 100:.2f}%")

# ============================================================
# SIMPAN HISTORY & GRAFIK
# ============================================================
acc  = history1.history['accuracy']  + history2.history['accuracy']
val_acc  = history1.history['val_accuracy']  + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

with open(HISTORY_OUTPUT, "w") as f:
    json.dump({
        "accuracy": acc, "val_accuracy": val_acc,
        "loss": loss,    "val_loss": val_loss
    }, f, indent=2)
print(f"\n[OK] History disimpan ke {HISTORY_OUTPUT}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(acc,     label='Train Accuracy')
ax1.plot(val_acc, label='Val Accuracy')
ax1.axvline(len(history1.history['accuracy']) - 1,
            color='gray', linestyle='--', label='Fine-tune start')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True)

ax2.plot(loss,     label='Train Loss')
ax2.plot(val_loss, label='Val Loss')
ax2.axvline(len(history1.history['loss']) - 1,
            color='gray', linestyle='--', label='Fine-tune start')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True)

plt.suptitle('Vegetable CNN - Training History (CPU Mode)', fontsize=14)
plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("[OK] Grafik disimpan ke training_history.png")
plt.show()

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print("\n" + "=" * 60)
print("  TRAINING SELESAI!")
print("=" * 60)
print(f"  Model tersimpan   : {MODEL_OUTPUT}")
print(f"  Class indices     : class_indices.json")
print(f"  Grafik history    : training_history.png")
print(f"  Test Accuracy     : {test_acc * 100:.2f}%")
print("=" * 60)
print("\nSalin file berikut ke laptop untuk deploy Flask:")
print(f"  1. {MODEL_OUTPUT}")
print(f"  2. class_indices.json")
print("=" * 60)
