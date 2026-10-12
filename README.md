
# Airline Satisfaction Prediction

A machine learning project that predicts airline passenger satisfaction using passenger information, flight details, and service ratings.

## Features

- Predicts whether a passenger is satisfied or neutral/dissatisfied.
- Uses CatBoost and LightGBM models.
- Combines both models using probability blending.
- Provides a web interface built with Flask.
- Displays the predicted class and satisfaction probability.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- CatBoost
- LightGBM
- Flask
- HTML and CSS

## Project Structure

```text
airline-satisfaction-prediction/
├── artifacts/
├── notebook/
├── src/
├── static/
├── templates/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/airline-satisfaction-prediction.git
cd airline-satisfaction-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Open this URL in your browser:

http://127.0.0.1:5000

## Model Performance

Validation ROC-AUC:

| Model | ROC-AUC |
|---|---:|
| CatBoost | 0.957720 |
| LightGBM | 0.957672 |
| Blended Model | 0.957895 |

These results are from the local validation experiment. They are not a guarantee of performance on unseen data.

## Dataset

The project uses the Kaggle Playground Series competition dataset for predicting airline satisfaction. The dataset is excluded from this repository.

## License

This project is available for viewing and educational purposes. Contact the author for permission before redistributing or reusing the code.