import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras_tuner as kt

# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of actions (signs) labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Define the number of sequences and frames
sequences = 40
frames = 50

# Create a label map to map each action label to a numeric value
label_map = {label:num for num, label in enumerate(actions)}

# Initialize empty lists to store landmarks and labels
landmarks, labels = [], []

# Iterate over actions and sequences to load landmarks and corresponding labels
for action in actions:
    for sequence in range(sequences):
        file_list = sorted(glob.glob(os.path.join(PATH, action, str(sequence), '*.npy')))
        temp = []

        if len(file_list) < frames:
            continue

        for file_path in file_list[:frames]:
            npy = np.load(file_path)
            temp.append(npy)
        landmarks.append(temp)
        labels.append(label_map[action])

# Convert landmarks and labels to numpy arrays
X = np.array(landmarks)
Y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

# Define a model building function for Keras Tuner
def build_model(hp):
    model = Sequential()

    # Add the first LSTM layer
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(frames, X.shape[2])))

    # Add the second LSTM layer
    model.add(LSTM(128, return_sequences=True, activation='relu'))

    # Add the third LSTM layer
    model.add(LSTM(128, return_sequences=True, activation='relu'))  # New LSTM layer added

    # Add the fourth LSTM layer
    model.add(LSTM(64, return_sequences=False, activation='relu'))

    # Add the first Dense layer
    model.add(Dense(64, activation='relu'))

    # Add a second Dense layer to increase depth
    model.add(Dense(64, activation='relu'))  # New Dense layer added

    # Add the output layer
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Tune the learning rate for the optimizer
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', 'accuracy'])

    return model

# Define the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_categorical_accuracy',  # Optimize for validation accuracy
    max_trials=5,  # Try up to 5 different models
    executions_per_trial=1,  # Train each model once
    directory='tuner_dir',  # Save the tuner results here
    project_name='tune_learning_rate'
)

# Run the tuner search
tuner.search(X_train, Y_train, epochs=100, validation_split=0.1)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"The best learning rate is: {best_hps.get('learning_rate')}")

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Define a checkpoint callback to save the best model
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',        # Monitor the validation loss
    patience=150,              # Wait for 150 epochs for improvement
    verbose=1,                 # Print a message when stopping
    mode='min',                # Stop when the quantity monitored has stopped decreasing
    restore_best_weights=True  # Restore the weights of the best epoch
)

# Train the model with the best hyperparameters
history = model.fit(X_train, Y_train, epochs=10000, validation_split=0.1, callbacks=[checkpoint, early_stopping])

# Plot training & validation accuracy values
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Make predictions on the test set
predictions = np.argmax(model.predict(X_test), axis=1)

# Get the true labels from the test set
test_labels = np.argmax(Y_test, axis=1)

# Calculate the accuracy of the predictions
accuracy = metrics.accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")
