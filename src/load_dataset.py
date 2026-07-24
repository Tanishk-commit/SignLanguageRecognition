import tensorflow as tf

DATASET_PATH = "dataset"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

print("\nDataset loaded successfully!")

print("\nClasses:")
print(train_dataset.class_names)

print("\nNumber of classes:", len(train_dataset.class_names))