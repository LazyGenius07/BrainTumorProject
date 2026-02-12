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

<img width="1919" height="1013" alt="image" src="https://github.com/user-attachments/assets/a3643082-9d7f-4c12-ba58-1b5451ae3982" />


Prediction Result

<img width="1835" height="1550" alt="image" src="https://github.com/user-attachments/assets/15ed0687-feed-4c4f-bd8c-8c038cd88c56" />

# Model Details
1. CNN-Based Tumor Classification Model


        The tumor classification system is implemented using Transfer Learning with VGG16, a deep convolutional neural network pre-trained on the ImageNet dataset. Transfer learning enables the model to leverage learned low-level and mid-level image features, improving performance on medical imaging tasks with limited data.

        🔹 Architecture Configuration

            Base Model: VGG16 (include_top=False, pretrained on ImageNet)    
            Input Size: 128 × 128 × 3    
            Frozen Layers: All convolutional layers initially frozen    
            Fine-Tuning: Last 3 convolutional layers unfrozen for domain adaptation
            Custom Fully Connected Layers Added:    
            Flatten Layer    
            Dropout (0.3)    
            Dense Layer (128 neurons, ReLU activation)
            Dropout (0.2)
            Output Layer (Softmax activation)

        🔹 Classification Output

            The model performs multi-class classification and predicts one of the following categories:
            No Tumor
            Glioma Tumor
            Meningioma Tumor
            Pituitary Tumor

            The final output layer uses Softmax activation, producing probability scores for each class.

        🔹 Training Configuration

        Optimizer: Adam (Learning Rate = 0.0001)
        Loss Function: Sparse Categorical Crossentropy
        Evaluation Metrics:
            Sparse Categorical Accuracy
            Confusion Matrix
            Classification Report (Precision, Recall, F1-score)
            ROC Curve and AUC Score
2. Image Preprocessing Pipeline
   Proper preprocessing is essential for improving convergence and model generalization.

        🔹 Image Resizing

            All MRI images are resized to:
            128 × 128 × 3
            This ensures consistent input dimensions across the dataset.

        🔹 Normalization

            Pixel intensity values are scaled from:
            0–255 → 0–1
            This stabilizes gradient updates and improves training efficiency.

        🔹 Data Augmentation
            
            To reduce overfitting and improve robustness, the following augmentations are applied:
            Random brightness adjustment (0.8–1.2 range)
            Random contrast variation (0.8–1.2 range)
            Data shuffling at each epoch
            Custom batch data generator to manage memory efficiently

3. Tumor Segmentation Model (Boundary Extraction)

    For precise tumor localization, a U-Net architecture is implemented to perform pixel-wise segmentation.

        🔹 Architecture Overview
            
            The U-Net model consists of:
                Encoder (Contracting Path):
                    Convolution Blocks
                    ReLU Activation
                    MaxPooling Layers
                    Bottleneck Layer

                Decoder (Expanding Path):
                    Transposed Convolutions
                    Skip Connections (to retain spatial information)
                    Convolution Blocks
                    Output Layer:
                        1×1 Convolution
                        Sigmoid Activation
                🔹 Input Size
                        128 × 128 × 3
                🔹 Output
                        Binary segmentation mask:
                            1 → Tumor Region
                            0 → Background

4. Segmentation Loss Function

To improve segmentation accuracy, a hybrid loss function is used:
Binary Cross Entropy (BCE) + Dice Loss

    🔹 Dice Coefficient Formula
        Dice = (2 × |Prediction ∩ Ground Truth|) / (|Prediction| + |Ground Truth|)
        This ensures:
            Pixel-wise classification accuracy (BCE)
            Maximized overlap between predicted and actual tumor regions (Dice Loss)

5. Training Optimization Techniques

To ensure stable and efficient training, the following techniques are applied:
    
    Early Stopping (prevents overfitting)
    Reduce Learning Rate on Plateau
    Model Checkpointing (saves best performing model)
    Fixed random seed for reproducibility

6. Integrated System Capabilities

The final system integrates classification and segmentation to provide:

    Automatic tumor detection
    Tumor type classification (4 classes)
    Tumor boundary extraction
    Tumor region mask visualization
    Tumor coverage percentage estimation
    Prediction confidence scores

# Future Improvements

    > Improve model accuracy using transfer learning
    > Add multi-class tumor classification
    > Deploy application on cloud (Render)
    > Add user authentication system
    > Integrate real-time MRI dataset support
    > Convert into full medical diagnostic dashboard

# Author's

    1>  D M Abdul Razzaq
        B.E – Artificial Intelligence and Machine Learing at BMS Collage of Engineering
        USN 1BM24AI407
        Mini Project – 2026
        
    2>  Jeevan Ravindra Kunter
        B.E – Artificial Intelligence and Machine Learing at BMS Collage of Engineering
        USN 1BM24AI409
        Mini Project – 2026
    
GitHub: https://github.com/LazyGenius07

📜 License

This project is for academic and educational purposes only.




