# 🌿 Smart Herbal Plant Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ObjectDetection-green)
![Flask](https://img.shields.io/badge/Flask-WebApp-lightgrey)

A computer vision web application that detects **medicinal plants from images** and provides **scientifically documented medicinal uses and precautions**.

The system uses a **hybrid deep learning pipeline** combining:

* YOLO-based plant detection
* CNN-based plant classification
* Structured medicinal knowledge retrieval

---

# 📌 Project Overview

The goal of this system is to assist users in identifying **common medicinal plants** using computer vision.

The workflow:

1. User uploads an image of a plant.
2. YOLO detects plant regions in the image.
3. The CNN classifier predicts the plant species.
4. The system displays:

   * Scientific name
   * Medicinal uses
   * Safety precautions

The application runs locally through a **Flask web interface**.

---

# 🧠 Model Pipeline

```
Input Image
     │
     ▼
YOLO Plant Detector
(detect plant regions)
     │
     ▼
Crop Detected Regions
     │
     ▼
CNN Classifier
(plant species prediction)
     │
     ▼
Medicinal Knowledge Retrieval
(CSV database)
     │
     ▼
Web Interface Output
```

---

# 🌱 Supported Plant Species

The current model can recognize the following medicinal plants:

• Aloe Vera
• Brahmi
• Centella (Gotu Kola)
• Turmeric

An **Unknown class** is included to reject:

• weeds
• dry leaves
• irrelevant vegetation

---

# 📂 Project Structure

```
herbal-plant-detection
│
├── app.py
├── inference_engine.py
├── plant_detector.py
├── dataset_report.py
│
├── medicinal_data.csv
├── class_indices.json
├── requirements.txt
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/rithu-dev-ug/-herbal-plant-detection.git
cd -herbal-plant-detection
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📥 Download Trained Models

The trained models are stored separately due to their size.

Download them from Google Drive:

**Model Download Link**

```
https://drive.google.com/drive/folders/14jrrCB16sliZ0_FShO9J9-cIqo1vb7hZ?usp=drive_link
```

Download the following files:

```
plant_classifier.h5
best_phase1.h5
best_phase2.h5
```

Place them in the project root:

```
herbal-plant-detection
│
├── app.py
├── plant_classifier.h5
├── best_phase1.h5
├── best_phase2.h5
```

---

# 🚀 Running the Application

Start the server:

```bash
python app.py
```

Open the browser:

```
http://127.0.0.1:5000
```

Upload an image and the system will detect the medicinal plant.

---

# 📊 Dataset

Images were collected and curated from:

• iNaturalist
• Botanical image references
• Manually collected datasets

Dataset cleaning steps included removing:

• non-plant objects
• dry leaves
• weeds
• noisy backgrounds

An **Unknown category** was introduced to reduce false predictions.

---

# ⚠️ Key Challenges

During development several challenges were encountered:

• visually similar leaf structures
• cluttered backgrounds with weeds
• class imbalance in the dataset
• false detections from background vegetation

These were addressed through dataset refinement and improved detection filtering.

---

# 🛠 Technologies Used

Python
TensorFlow / Keras
Ultralytics YOLOv8
OpenCV
Flask
Pandas
NumPy

---

# 🎓 Academic Context

Developed as a **Computer Science Engineering mini-project** exploring practical applications of **deep learning and computer vision for herbal plant recognition**.

---

# 📜 License

This project is intended for **educational and research purposes only**.
