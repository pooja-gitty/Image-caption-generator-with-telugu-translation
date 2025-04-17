# 🖼️ Image Caption Generator with Telugu Translation

This project generates captions for images and translates them into **Telugu** using deep learning and NLP techniques.

---

## 🚀 Project Overview

- 📷 **Image Captioning** using a deep learning model trained on the Flickr8k dataset.
- 🌐 **Telugu Translation** of generated captions.
- 🧠 Combines CNN + LSTM for feature extraction and sequence prediction.
- 🖥️ Interactive user interface built with **Streamlit**.

---

## 📦 Dataset

- Download the Flickr8k dataset from Kaggle:  
  👉 [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

## 💻 Steps to Run This Project

### ✅ Step 1: Set up the Jupyter Notebook
- Open `NLPProject.ipynb` in **Jupyter Notebook**.
- Replace the file paths to point to:
  - The images directory.
  - The `captions.txt` file from the Flickr8k dataset.

---

### ✅ Step 2: Generate Required Files
- Run all cells in the notebook.
- This will save the following files in your directory:
  - `model.keras`
  - `feature_extractor.keras`
  - `tokenizer.pkl`

---

### ✅ Step 3: Prepare Project Directory
- Your project folder should now contain:
  - `NLPProject.ipynb`
  - `model.keras`
  - `feature_extractor.keras`
  - `tokenizer.pkl`
  - `app.py`

---

### ✅ Step 4: Launch the Application
- In your terminal, navigate to the project directory.
- Run the Streamlit app using:

```bash
streamlit run app.py
