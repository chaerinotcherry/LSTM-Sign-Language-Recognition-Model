# %%

# Import necessary libraries
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn import metrics
import matplotlib.pyplot as plt  # 그래프를 그리기 위해 필요한 라이브러리

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set the path pattern to the data directories
PATH_PATTERN = 'data_*'

# Get all relevant directories
directories = [d for d in glob.glob(PATH_PATTERN) if os.path.isdir(d)]

# Create an array of actions (signs) labels by listing the contents of the data directories
actions = ['water', 'fire','down','flow','hell','walk','fail']

actions = np.array(list(actions))

# Define the number of sequences and frames
sequences = 40
frames = 30

# Create a label map to map each action label to a numeric value
label_map = {label: num for num, label in enumerate(actions)}  # 예: {'water': 0, 'down': 1}

# Initialize empty lists to store landmarks and labels
landmarks, labels = [], []  # 각각 초기화

# Iterate over directories, actions, and sequences to load landmarks and corresponding labels
for directory in directories:
    for action in actions:
        action_path = os.path.join(directory, action)
        if os.path.exists(action_path):
            for sequence in range(sequences):
                # 파일 리스트를 가져와서 정렬합니다. (타임스탬프가 포함된 파일 이름을 정렬하면 순서가 보장됩니다.)
                file_list = sorted(glob.glob(os.path.join(action_path, str(sequence), '*.npy')))
                temp = []
                for file_path in file_list[:frames]:  # 필요한 프레임 수만큼 데이터 로드
                    npy = np.load(file_path)
                    temp.append(npy)  # temp에 추가하기
                if len(temp) == frames:  # 필요한 프레임 수만큼 데이터가 있는 경우에만 추가
                    landmarks.append(temp)  # landmark에 data_collection.npy에 있는 값을 landmark에 추가
                    labels.append(label_map[action])  # label에 각각 값을 적어내기 예를 들어 water이면 0 down이면 1로 적어내기.

# Convert landmarks and labels to numpy arrays
X, Y = np.array(landmarks), to_categorical(labels).astype(int)  # X에 landmark에 있는 애들을 numpy 배열로 담기, Y에 label을 one-hot 코딩으로 담기

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

# Define the model architecture
model = Sequential()
model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(frames, X.shape[2])))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model and save the training history
history = model.fit(X_train, Y_train, epochs=140, validation_split=0.1)

# Save the trained model
model.save('my_model.keras')

# Make predictions on the test set
predictions = np.argmax(model.predict(X_test), axis=1)  # 가장 높은 확률의 인덱스를 선택한다. 예를 들어 array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]]) 이렇게 되어 있으면 --> [0,1,0] 으로 된다.

# Get the true labels from the test set
test_labels = np.argmax(Y_test, axis=1)
'''array([[1, 0],
       [0, 1],
       [1, 0]])      이러한 느낌의 Y_test가 있다면 이를 [0,1,0] 으로 만든다.
'''

# Calculate the accuracy of the predictions
accuracy = metrics.accuracy_score(test_labels, predictions)  # 따라서 이를 test_label과 prediction을 비교해서 정확도를 산출해낸다.

# Plotting training history
fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['categorical_accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_categorical_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper right')

plt.show()