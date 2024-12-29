# LSTM Sign Language Recognition Model for VR devices

## 1. Development Background
The project aims to move beyond the use of VR controllers that rely on monotonous actions and simple button inputs. Instead, it seeks to implement an interface that recognizes a sequence of actions on VR devices by leveraging hand gesture/sign language recognition.

## 2. Project Description
In the initial stages, webcam was utilized as a starting point, with some features forked from https://github.com/dgovor/Sign-Language-Translator. These features were later expanded and refined to meet the project's objectives.

To create a sign language recognition model for Oculus devices, we developed a Unity package to capture hand keypoint movements using the Oculus Quest 2. We also created a method to preprocess the data and developed a model training process to achieve optimal results and enabled real-time testing using the device.

### 2.2 Key Features
- Data generation
- Data preprocessing
- Model training
- Model deployment/real-time testing

## 3. File Structure
The primary file structure of the project is as follows:

```
📂 LSTM-Sign-Language-Recognition-Model
│   │
├── 📂 oculus                            # Recognition model for an Oculus device
│   ├── 📂 python                        
│   │    ├── convert.py                  # Data Preprocessing
│   │    ├── getrange.py                 # Script for calculating data ranges (can be used for data preprocessing)
│   │    ├── main.py                     # Main script for processing
│   │    └── model.py                    # LSTM model implementation
│   ├── 📂 unity                         
│   │    └── handtracking.unitypackage   # Package for collecting data and testing model with Onculus
│   │
├── 📂 webcam                            # Recognition model for webcam
│   ├── data_collection.py               # Script for collecting data
│   ├── delete.py                        # Script for deleting unwanted data
│   ├── main.py                          # Main script for webcam data processing
│   ├── model.py                         # LSTM model implementation
│   └── my_functions.py                  # Helper functions
│   
└── README.md                           
```

## 4. Usage Instructions

### 4.1 Webcam
Collect data
```bash
python webcam/data_collection.py
```

Train a Model based on Collected Data
```bash
python webcam/model.py
```

Test the Trained Model
```bash
python webcam/main.py
```

### 4.2 Oculus

Collect data with the handtracking unity package (oculus/unity/handtracking.unitypackage)
1. Turn on "Timer" script
2. Select "Data Collect" script and fill the sign name
3. Run the package and use sign language with oculus on

Preprocess Data
```bash
python oculus/convert.py
```
\* Decide Threshold Value with this:
```bash
python oculus/getrange.py
```

Train a Model based on Collected Data
```bash
python oculus/model.py
```

Test the Trained Model
1. Run following command:
```bash
python oculus/main.py
```
2. Send real-time data through the unity package (oculus/unity/handtracking.unitypackage); Select "Test Collect" Script


## 5. Applications and Performance
This project has been successfully utilized in developing a Motion Recognition-Based Adventure Game in VR,
which was awarded the **2024 Metaverse Developer Contest Meta Representative Award**.
Learn more about the model and the game on https://www.modoogallery.online/studioon

## 6. Collaborators and Helpers

- [Chaeri Kim](https://github.com/chaerinotcherry)
- [Jaewon Zhang](https://github.com/silverstick393)
- Developed with advice from professor **Heesun Park**
