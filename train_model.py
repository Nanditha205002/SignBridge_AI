import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

DATA_PATH = os.path.join('data')
ACTIONS = np.array(['hello', 'hospital', 'doctor', 'yesterday', 'go'])
SEQUENCE_LENGTH = 30

label_map = {label: num for num, label in enumerate(ACTIONS)}

sequences, labels = [], []
for action in ACTIONS:
    for sequence in range(30):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Build LSTM Model Architecture
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    Dropout(0.2),
    LSTM(128, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(ACTIONS.shape[0], activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

os.makedirs('models', exist_ok=True)
model.save('models/signbridge_lstm.h5')
print("Model saved to models/signbridge_lstm.h5")