import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================
from parse_skeleton import load_dataset

X, y = load_dataset("../data/ntu_skeletons")

print("Dataset shape:", X.shape, y.shape)

# Convert labels to categorical
y_cat = to_categorical(y, num_classes=2)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 5)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(2, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# =========================
# SAVE MODEL
# =========================
model.save("../models/silent_signals_intent.h5")
print("✅ Model saved!")
