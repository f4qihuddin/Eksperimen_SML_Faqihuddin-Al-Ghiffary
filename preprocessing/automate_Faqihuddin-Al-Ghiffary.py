import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def preprocess_image(TRAIN_DIR, VAL_DIR):
    checkpoint_dir_train = os.path.join(TRAIN_DIR, '.ipynb_checkpoints')
    checkpoint_dir_val = os.path.join(VAL_DIR, '.ipynb_checkpoints')

    if os.path.exists(checkpoint_dir_train):
        shutil.rmtree(checkpoint_dir_train)
        print(f"Removed: {checkpoint_dir_train}")

    if os.path.exists(checkpoint_dir_val):
        shutil.rmtree(checkpoint_dir_val)
        print(f"Removed: {checkpoint_dir_val}")

    # Memastikan direktori ada sebelum digunakan oleh generator
    # Catatan: Kode augmentasi sebelumnya harus sudah memindahkan file ke folder kategori di sini
    if not os.path.exists(TRAIN_DIR):
        print(f"Peringatan: Folder {TRAIN_DIR} tidak ditemukan. Pastikan proses augmentasi di sel sebelumnya sudah selesai.")

    # ImageDataGenerator dengan augmentasi sederhana
    datagen = ImageDataGenerator(
        rescale=1/255.,
        validation_split=0.3
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(150, 150),
            batch_size=128,
            color_mode='grayscale',
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        validation_generator = datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(150, 150),
            batch_size=128,
            color_mode='grayscale',
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        test_generator = test_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(150, 150),
            batch_size=128,
            color_mode='grayscale', # Disamakan dengan training (grayscale)
            class_mode='categorical',
            shuffle=False
        )

        # Dapatkan direktori tempat script dijalankan
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Buat path untuk folder preprocessing
        output_base_dir = os.path.join(script_dir, 'coffeebeans_preprocessing')
        os.makedirs(output_base_dir, exist_ok=True)

        # Fungsi untuk mengkonversi generator ke numpy array
        def generator_to_numpy(generator):
            X, y = [], []
            for images, labels in generator:
                X.append(images)
                y.append(labels)
                if len(X) >= len(generator):
                    break
            return np.concatenate(X), np.concatenate(y)

        # Konversi train generator
        print("Mengkonversi train generator...")
        X_train, y_train = generator_to_numpy(train_generator)
        np.save(os.path.join(output_base_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_base_dir, 'y_train.npy'), y_train)
        print(f"Train: {X_train.shape}, {y_train.shape}")

        # Konversi validation generator
        print("Mengkonversi validation generator...")
        X_val, y_val = generator_to_numpy(validation_generator)
        np.save(os.path.join(output_base_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_base_dir, 'y_val.npy'), y_val)
        print(f"Validation: {X_val.shape}, {y_val.shape}")

        # Konversi test generator
        print("Mengkonversi test generator...")
        X_test, y_test = generator_to_numpy(test_generator)
        np.save(os.path.join(output_base_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_base_dir, 'y_test.npy'), y_test)
        print(f"Test: {X_test.shape}, {y_test.shape}")

        print(f"\nSemua data tersimpan di folder: {output_base_dir}")

    except Exception as e:
        print(f"Error saat memuat generator: {e}\nPastikan data augmentasi sudah tersusun dalam folder kategori (misal: train_augmented/cats/)")
        
preprocess_image('coffeebeans_raw/train', 'coffeebeans_raw/test')