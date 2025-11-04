import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


base_dir = "D:\Dataset\Celeb Frames"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")


def get_image_paths(folder, label):
    paths = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            paths.append([os.path.join(folder, filename), label])
    return paths


train_real = get_image_paths(os.path.join(train_dir, "real"), "REAL")
train_fake = get_image_paths(os.path.join(train_dir, "fake"), "FAKE")
val_real = get_image_paths(os.path.join(val_dir, "real"), "REAL")
val_fake = get_image_paths(os.path.join(val_dir, "fake"), "FAKE")

train_df = pd.DataFrame(train_real + train_fake, columns=["filename", "class"])
test_df = pd.DataFrame(val_real + val_fake, columns=["filename", "class"])

print(f"✅ Loaded {len(train_df)} training images")
print(f"✅ Loaded {len(test_df)} validation images")


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    train_df,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_dataframe(
    train_df,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    subset='validation'
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)


densenet = DenseNet121(
    weights='imagenet',  # using pretrained ImageNet weights
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in densenet.layers:
    layer.trainable = False


def build_model(base_model):
    model = Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0005),
        metrics=['accuracy']
    )
    return model

model = build_model(densenet)
model.summary()

checkpoint = ModelCheckpoint('model_best.h5', save_best_only=True, monitor='val_accuracy', mode='max')

print("\n🔹 Training Step 1: With frozen DenseNet layers...")
train_history_step1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=7,
    callbacks=[checkpoint]
)

print("\n🔹 Training Step 2: Unfreezing DenseNet layers...")
model.load_weights('model_best.h5')

for layer in model.layers:
    layer.trainable = True

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

train_history_step2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3,
    callbacks=[checkpoint]
)


pd.DataFrame(train_history_step1.history).to_csv('history_step1.csv', index=False)
pd.DataFrame(train_history_step2.history).to_csv('history_step2.csv', index=False)

print("\n✅ Training complete! Model and histories saved.")
