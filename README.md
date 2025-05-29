# Spam Detection API 
![flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) 
![pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) 
![sklearn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

A simple and fast Flask-based REST API that classifies incoming text messages as **spam** or **ham** (not spam) using a trained Machine Learning model built with `scikit-learn`.

---

## Features

- RESTful API built with Flask
- Spam detection using `Multinomial Naive Bayes` and `CountVectorizer`
- Language restriction (supports only English)
- Model training script included
- Uses `.venv` for dependency isolation

---

## File Structure

```
spam-detection-api/
├── app/
│   ├── __init__.py         # Flask app
│   ├── model.py            # App routes
│   ├── model.py            # Trained model function
├── models/
│   └── model.pkl           # Saved trained model
├── data/
│   └── spam.csv            # Dataset
├── train_model.py          # Train the model
├── run.py                  # Runs the Flask App
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

---
## Setup Instructions

*You can use Docker to build and run this project or follow the steps to setup the project on your machine.*

### 1. Clone the repo

```bash
git clone https://github.com/supr1yo/spam-detection-api.git
cd spam-detection-api
```

### 2. Set up the virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Add dataset (if missing)

Download the `spam.csv` dataset (e.g., from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)) and place it in the `data` directory if missing.

---

## Train the Model

```bash
python train_model.py
```

This script reads `spam.csv`, trains the model, and saves `model.pkl` inside the `models/` directory.

---

## Run the API

```bash
python run.py
```

The API will start at `http://127.0.0.1:5000/`

---

## API Endpoint
### `POST /predict`

Classifies a message as spam or not.

#### Request

```json
{
    "message": "Free $50 steam points"
}
```

#### Response

```json
{
    "label": "spam",
    "message": "Free $50 steam points"
}
```

Returns `400 Bad Request` if:
- Message is missing from request body
---


## Model Info

- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** CountVectorizer
---



## License

MIT License. See [`LICENSE`](https://github.com/supr1yo/spam-detection-api/blob/main/LICENSE) file for details.

---

## Author

Built with 💙 by [supr1yo](https://github.com/supr1yo)
