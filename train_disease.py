import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = 'data/plantvillage'
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 5

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.15,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE,
    batch_size=BATCH, subset='training'
)
val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE,
    batch_size=BATCH, subset='validation'
)

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training disease model...")
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

model.save('model/disease_model.h5')
print("Saved → model/disease_model.h5")

# Save class names
import json
with open('model/class_names.json', 'w') as f:
    json.dump(train_gen.class_indices, f)
print("Saved → model/class_names.json")