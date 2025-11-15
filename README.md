# Resume-Role-Classifier

## Project Overview
The Resume Classification & Suitability Checker is a web-based application that classifies resumes into predefined job categories using machine learning. This tool allows users to upload resumes and receive accurate job role predictions. 

For this project, we experimented with two different machine learning models to classify resumes into relevant job roles: LSTM (Long Short-Term Memory) and BERT (Bidirectional Encoder Representations from Transformers). Each model was evaluated based on key metrics, such as accuracy, precision, recall, and F1-score, to determine which one performed best for resume classification tasks.

### Data Source
The dataset was obtained from Kaggle and can be downloaded using the following link:
- [Dataset 1](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data)
- [Dataset 2](https://github.com/noran-mohamed/Resume-Classification-Dataset)

## Setup
Follow the instructions below to set up the environment on your local machine.

### 1. *Clone the repository*

### 2. *Create and activate a virtual environment*
#### For Windows:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```
#### For Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. *Install dependencies*

```bash
pip install -r requirements.txt
```

### 4. *Run the project*
#### To obtain the trained BERT model for resume classification, users are required to run the BERT.ipynb notebook

### 5. *Run the Webpage*

```bash
streamlit run app.py
```
