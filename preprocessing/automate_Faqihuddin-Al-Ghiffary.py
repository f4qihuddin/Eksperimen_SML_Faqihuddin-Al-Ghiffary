import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

def save_preprocessed_images_from_generator(generator, output_base_dir, max_images_per_class=None):
    """
    Saves images and their labels from a Keras ImageDataGenerator to specified
    output directories, maintaining the class structure.
    Images are denormalized (multiplied by 255) and converted to uint8.

    Args:
        generator: An instance of ImageDataGenerator (e.g., train_generator).
        output_base_dir: The root directory where preprocessed images will be saved.
                         e.g., 'preprocessed_data/train'
        max_images_per_class: Optional. The maximum number of images to save per class.
                              If None, all images from the generator are saved.
    """
    print(f"Preparing to save images to: {output_base_dir}")

    # Get class labels mapping from the generator
    class_indices = generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}

    # Create base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Create class subdirectories
    for class_name in class_indices.keys():
        os.makedirs(os.path.join(output_base_dir, class_name), exist_ok=True)

    saved_counts = {class_name: 0 for class_name in class_indices.keys()}
    total_saved_images = 0

    print(f"Saving images from generator to {output_base_dir}...")

    # Iterate through the generator for one full pass (or until max_images_per_class is met).
    num_batches = len(generator)
    for i in range(num_batches):
        batch_images, batch_labels = next(generator) # Get the next batch

        for img_array, label_one_hot in zip(batch_images, batch_labels):
            class_idx = np.argmax(label_one_hot)
            class_name = idx_to_class[class_idx]

            if max_images_per_class is not None and saved_counts[class_name] >= max_images_per_class:
                continue # Skip if we've saved enough for this class

            # Denormalize (if scaled 0-1) and convert to uint8
            img_array_denorm = (img_array * 255).astype(np.uint8)

            # Handle grayscale vs. RGB
            if img_array_denorm.shape[-1] == 1: # Grayscale
                img_array_denorm = np.squeeze(img_array_denorm, axis=-1)
                img = Image.fromarray(img_array_denorm, mode='L')
            else: # RGB
                img = Image.fromarray(img_array_denorm)

            filename = f"{class_name}_{saved_counts[class_name]}.png"
            filepath = os.path.join(output_base_dir, class_name, filename)
            img.save(filepath)

            saved_counts[class_name] += 1
            total_saved_images += 1

        # Check if all classes have reached their max_images_per_class (if specified)
        if max_images_per_class is not None and all(count >= max_images_per_class for count in saved_counts.values()):
            break

    print(f"Finished saving. Total images saved: {total_saved_images}")
    for class_name, count in saved_counts.items():
        print(f"  {class_name}: {count} images saved.")

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

        # Define output directories for the preprocessed images
        output_base_dir = 'coffeebeans_preprocessing'
        output_train_dir = os.path.join(output_base_dir, 'train')
        output_validation_dir = os.path.join(output_base_dir, 'validation')
        output_test_dir = os.path.join(output_base_dir, 'test')

        # Save images from the train generator
        save_preprocessed_images_from_generator(train_generator, output_train_dir)

        # Save images from the validation generator
        save_preprocessed_images_from_generator(validation_generator, output_validation_dir)

        # Save images from the test generator
        save_preprocessed_images_from_generator(test_generator, output_test_dir)

    except Exception as e:
        print(f"Error saat memuat generator: {e}\nPastikan data augmentasi sudah tersusun dalam folder kategori (misal: train_augmented/cats/)")

    return train_generator, validation_generator, test_generator

preprocess_image('coffeebeans_raw/train', 'coffeebeans_raw/test')