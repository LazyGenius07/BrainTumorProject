Brain Tumor Detection & Segmentation Using Deep Learning
Abstract

Brain tumors are among the most critical neurological disorders, where early and precise diagnosis plays a vital role in patient survival. Manual examination of MRI scans is time-consuming and highly dependent on radiological expertise. To address this challenge, this project presents a deep learning–based system for automated brain tumor classification and segmentation using Magnetic Resonance Imaging (MRI).

The proposed system consists of two major components:

    Tumor Classification Model – A transfer learning approach using the VGG16 architecture is implemented to classify MRI images into four categories: Glioma, Meningioma, Pituitary tumor, and No Tumor. The model is trained using augmented image data and optimized using the Adam optimizer with sparse categorical cross-entropy loss.

    Tumor Segmentation Model – A U-Net based convolutional neural network is designed to accurately segment tumor regions from MRI images. The segmentation model uses a combination of Binary Cross-Entropy and Dice Loss to improve boundary detection and region accuracy.

The classification model performance is evaluated using:

    Accuracy
    Confusion Matrix
    Classification Report
    ROC Curve and AUC
    The segmentation model performance is evaluated using:
    Dice Coefficient
    Validation Accuracy
    Tumor Area Percentage Estimation

Additionally, a Flask-based web application integrates both models, allowing users to upload MRI images and obtain tumor classification results along with segmented tumor visualization and tumor percentage estimation.This project demonstrates the practical application of deep learning techniques in medical image analysis and highlights the potential of AI-assisted diagnostic systems in healthcare.

# Features

 Brain tumor classification using deep learning

 Tumor region segmentation

 Web-based interface for image upload

 Real-time prediction results

 Clean and user-friendly UI

 Model training notebooks included

 Flask-based deployment

# Tech Stack
 > Programming Language

    Python

    Deep Learning & ML

    TensorFlow / Keras

    NumPy

    OpenCV

    Matplotlib
    
    Web Framework

    Flask

    Frontend

    HTML

    CSS

 > Development Tools

    Jupyter Notebook

    Git & GitHub

    VS Code

# Project Structure
    BrainTumorProject/
    │
    ├── Dateset
        ├── https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
        ├── https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── Models/
    │   ├── Brain Tumor Detection.ipynb
    │   └── Brain_tumor_segmentation.ipynb
    ├── templates/
    │   ├── index.html
    │   └── result.html
    ├── static/
    └── .gitignore

# Installation Steps
1️ Clone the Repository
git clone https://github.com/LazyGenius07/BrainTumorProject.git
cd BrainTumorProject

2️ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows

3️ Install Dependencies
pip install -r requirements.txt

4️ Add Trained Model Files

Place your trained model files inside the Models/ folder:

classification_model.h5
segmentation_model.h5


(These are not included in the repo due to size limitations.)

▶ How to Run
python app.py


Then open your browser and go to:

http://127.0.0.1:5000/


Upload an MRI image to see: <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/3d7a32fa-ed01-4718-97c9-c57fdaf30908" />


Classification result (Tumor / No Tumor)
<img width="389" height="411" alt="image" src="https://github.com/user-attachments/assets/7175d5a5-f425-403e-989c-679e40a03a4d" />

Segmented tumor output
<img width="484" height="253" alt="image" src="https://github.com/user-attachments/assets/b3bce6ba-e8de-4e90-843b-95627ba32755" />

📸 Screenshots
🏠 Home Page

(Add screenshot here)

![Home Page](screenshots/home.png)

Prediction Result

(Add screenshot here)

![Result Page](screenshots/result.png)


(Create a screenshots/ folder and add images for better presentation.)

Model Details

CNN-based classification model

Image preprocessing: resizing, normalization

Binary classification output

Segmentation model for tumor boundary extraction

# Future Improvements

> Improve model accuracy using transfer learning

> Add multi-class tumor classification

> Deploy application on cloud (Render)

> Add user authentication system

> Integrate real-time MRI dataset support

> Convert into full medical diagnostic dashboard

# Author's

    1>  D M Abdul Razzaq
        B.E – Artificial Intelligence and Machine Learing
        Mini Project – 2026
    2>  Jeevan Ravindra Kunter
        B.E – Artificial Intelligence and Machine Learing
        Mini Project – 2026
    
GitHub: https://github.com/LazyGenius07

📜 License

This project is for academic and educational purposes only.


