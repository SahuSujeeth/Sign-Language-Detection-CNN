# Sign Language Detection using CNN

A real-time Sign Language Detection system built using **Convolutional Neural Networks (CNN)**, **OpenCV**, and **Streamlit**. This project recognizes hand gestures for **Alphabets (A–Z)** and **Numbers (0–9)** using a webcam and displays the predicted result with confidence.

---

## 🚀 Features

- Real-time hand sign detection
- Supports Alphabets and Numbers
- CNN-based image classification
- Live webcam prediction
- ROI (Region of Interest) for better accuracy
- Confidence score display
- Streamlit Web UI
- Training accuracy graph
- Confusion matrix
- Precision / Recall report

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- Streamlit

---

## 📂 Project Structure

```text
SignLanguage_CNN/
│── dataset/
│   ├── A/
│   ├── B/
│   ├── ...
│   └── 0/
│── collect.py
│── train_model.py
│── predict_live.py
│── streamlit_app.py
│── model.h5
│── labels.pkl
│── README.md

```
## ⚙️ Installation

1️⃣ Clone Repository

```
git clone https://github.com/sahusujeeth/Sign-Language-Detection-CNN.git
cd Sign-Language-Detection-CNN
```
2️⃣ Install Required Libraries

```
pip install tensorflow opencv-python numpy scikit-learn matplotlib seaborn pandas streamlit
```
## 📸 Dataset Collection

Use webcam to collect hand gesture images.

Run:
```
python collect.py
```
How to Use:
Enter label (Example: A, B, 1, 2)
Camera opens
Show hand inside green box
Press:
s → Save image
q → Quit
Repeat for all classes:
Alphabets: A to Z
Numbers: 0 to 9
## 🧠 Train the Model

After collecting dataset, train CNN model.

Run:
```
python train_model.py
```
Output Files:
model.h5 → Trained CNN model
labels.pkl → Label encoder
accuracy_graph.png
confusion_matrix.png
classification_report.csv
## 🎥 Real-Time Prediction (OpenCV)

Run live detection using webcam.
```
python predict_live.py
```
Controls:
A → Alphabet mode
N → Number mode
Q → Quit
## 🌐 Web App (Streamlit)

Run project in browser.
```
streamlit run streamlit_app.py
```
Then open the browser URL shown in terminal.

## 📊 Model Evaluation

This project includes:

Accuracy Graph
Confusion Matrix
Precision
Recall
F1-Score
💡 How It Works
Webcam captures hand image
ROI selects hand area
Image resized to model input size
CNN extracts features
Model predicts sign
Output displayed with confidence
📌 Future Improvements
Better UI design
Voice output
Sentence formation
Mobile app version
Higher accuracy with larger dataset

## 👨‍💻 Author

Developed as a Final Year Project for real-time Sign Language Recognition using CNN.