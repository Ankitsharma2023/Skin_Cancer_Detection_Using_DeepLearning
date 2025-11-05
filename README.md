# Skin Cancer Detection Using Deep Learning

A deep learning-based application for skin cancer detection using Convolutional Neural Networks (CNN). This project includes model training and multiple Streamlit interfaces for interactive predictions.

## 📋 Overview

This repository contains a complete pipeline for skin cancer detection, from model training to deployment through user-friendly Streamlit interfaces.

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow/Keras
- Streamlit
- Required dependencies (install via `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ankitsharma2023/Skin_Cancer_Detection_Using_DeepLearning.git
cd Skin_Cancer_Detection_Using_DeepLearning

# Install dependencies
pip install -r requirements.txt
```

## 📝 Usage Instructions

### Step 1: Train the Model

First, run the Jupyter notebook to train and save the model:

1. Open `notebooka66b915adb.ipynb` in Jupyter Notebook or JupyterLab
2. Execute all cells to train the CNN model
3. The trained model will be saved automatically

### Step 2: Run the Streamlit Application

After training and saving the model, you can use any of the three Streamlit interfaces:

```bash
# Option 1: Run main.py
streamlit run main.py

# Option 2: Run main2.py
streamlit run main2.py

# Option 3: Run main3.py
streamlit run main3.py
```

Each interface (`main.py`, `main2.py`, `main3.py`) provides different UI/UX experiences for skin cancer detection. Try them all to see which one you prefer!

## 🤗 Pre-trained Model

If you want to skip the training process, you can download the pre-trained model directly from Hugging Face:

**Model Link**: [https://huggingface.co/ankit87086/skin-cancer-cnn/resolve/main/skin_cancer_cnn.h5](https://huggingface.co/ankit87086/skin-cancer-cnn/resolve/main/skin_cancer_cnn.h5)

Download the model and place it in the project root directory before running the Streamlit applications.

## 📁 Project Structure

```
├── README.md
├── main.py                          # Streamlit interface variant 1
├── main2.py                         # Streamlit interface variant 2
├── main3.py                         # Streamlit interface variant 3
├── notebooka66b915adb.ipynb        # Model training notebook
└── skin_cancer_cnn.h5              # Trained model (generated after training)
```

## 🔬 Model Details

The project uses a Convolutional Neural Network (CNN) architecture trained on skin lesion images to classify different types of skin cancer.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Ankitsharma2023**

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: Make sure to have all required dependencies installed before running the application. The model requires adequate computational resources for training.
