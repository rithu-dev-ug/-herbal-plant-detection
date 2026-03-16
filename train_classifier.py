import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, MobileNetV2
from keras.layers import Dense, Flatten, Concatenate, Input, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 16
DATASET_DIR = "dataset"

# ─────────────────────────────────────────────
# SEPARATE GENERATORS
# Training gets augmentation. Validation gets NONE.
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=35,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode="nearest"
)

# Validation must be clean — rescale only
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print(f"\nClasses found: {train.class_indices}")
print(f"Training samples:   {train.samples}")
print(f"Validation samples: {val.samples}\n")

# ─────────────────────────────────────────────
# CLASS WEIGHTS (handles imbalance)
# ─────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train.classes),
    y=train.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}\n")

# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Load backbones — fully frozen to start
vgg    = VGG16(weights="imagenet",     include_top=False, input_tensor=input_layer)
mobile = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_layer)

for layer in vgg.layers:
    layer.trainable = False

for layer in mobile.layers:
    layer.trainable = False

# Use GlobalAveragePooling instead of Flatten
# Flatten produces huge vectors (~100k) that overfit fast
vgg_features    = GlobalAveragePooling2D()(vgg.output)
mobile_features = GlobalAveragePooling2D()(mobile.output)

merged = Concatenate()([vgg_features, mobile_features])

# Deeper classifier head with BatchNorm for stability
x = Dense(512, activation="relu")(merged)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(train.num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output)

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
callbacks_phase1 = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        "best_phase1.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

callbacks_phase2 = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        "best_phase2.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# ─────────────────────────────────────────────
# PHASE 1 — Train classifier head only (frozen backbones)
# Let the new Dense layers converge before touching CNN weights
# ─────────────────────────────────────────────
print("=" * 50)
print("PHASE 1 — Training classifier head (backbones frozen)")
print("=" * 50)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train,
    validation_data=val,
    epochs=15,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase1
)

# ─────────────────────────────────────────────
# PHASE 2 — Fine-tune: unfreeze last 4 layers of each backbone
# Use a MUCH lower learning rate to avoid destroying pretrained weights
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("PHASE 2 — Fine-tuning (last 4 layers unfrozen)")
print("=" * 50)

for layer in vgg.layers[-4:]:
    layer.trainable = True

for layer in mobile.layers[-4:]:
    layer.trainable = True

# Recompile with low LR — critical after unfreezing
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train,
    validation_data=val,
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase2
)

# ─────────────────────────────────────────────
# SAVE FINAL MODEL
# ─────────────────────────────────────────────
model.save("plant_classifier.h5")
print("\nModel saved as plant_classifier.h5")
print("Best checkpoints also saved as best_phase1.h5 and best_phase2.h5")

# ─────────────────────────────────────────────
# SAVE CLASS INDICES
# Saves label mapping so inference_engine never
# relies on a hardcoded class list
# ─────────────────────────────────────────────
import json
with open("class_indices.json", "w") as f:
    json.dump(train.class_indices, f, indent=4)
print("Class indices saved as class_indices.json")
print(f"Classes: {train.class_indices}")

print("\nTraining complete.")