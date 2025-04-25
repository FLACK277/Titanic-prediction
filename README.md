# Titanic Survival Prediction
A machine learning project to predict passenger survival on the Titanic.

## Project Objectives

This project aims to develop a machine learning model that predicts whether a passenger survived the Titanic disaster based on features like age, gender, ticket class, fare, and cabin information. The project demonstrates effective data science techniques including:

- Exploratory data analysis
- Data preprocessing and feature engineering
- Model selection, training, and evaluation
- Hyperparameter tuning
- Feature importance analysis

## Dataset Description

The Titanic dataset includes the following features:

- **PassengerId**: Unique identifier for each passenger
- **Survived**: Target variable (0 = did not survive, 1 = survived)
- **Pclass**: Passenger class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
- **Name**: Passenger name
- **Sex**: Gender of passenger
- **Age**: Age of passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Project Structure

```
titanic-survival-prediction/
├── README.md                   # Project documentation
├── titanic_survival.py         # Main Python script
├── requirements.txt            # Dependencies
├── data/
│   └── train.csv               # Training dataset
├── notebooks/
│   └── titanic_analysis.ipynb  # Jupyter notebook with analysis
└── models/
    └── tuned_model.pkl         # Saved trained model
```

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the Titanic dataset from Kaggle and place it in the `data/` directory.

## Usage

### Running the Main Script

To train the model and see the results:

```bash
python titanic_survival.py
```

The script will:
1. Load and explore the dataset
2. Preprocess the data and engineer features
3. Train and compare multiple models
4. Tune hyperparameters of the best model
5. Analyze feature importance
6. Provide a project summary and usage instructions

## Approach

### 1. Data Preprocessing

- **Missing Value Handling**: Used median imputation for numerical features and mode imputation for categorical features
- **Feature Engineering**:
  - Extracted titles from passenger names
  - Created family size and IsAlone features
  - Generated fare per person feature
  - Extracted cabin information where available
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: Standard scaling for numerical features

### 2. Model Development

The project evaluates multiple classification algorithms:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Each model is evaluated using metrics including:
- Accuracy
- Precision
- Recall
- F1-score

### 3. Hyperparameter Tuning

The best performing base model undergoes hyperparameter tuning using GridSearchCV with 5-fold cross-validation to optimize performance.

### 4. Feature Importance Analysis

The project analyzes which features are most influential in predicting survival, providing insights into the factors that affected survival rates on the Titanic.

## Results

The model achieves strong predictive performance with key insights:
- Gender was the strongest predictor (women had higher survival rates)
- Passenger class was significant (higher classes had better survival chances)
- Age was a factor (children were prioritized)
- Family size affected survival rates
- Cabin information provided insights about passenger location on the ship

## Future Improvements

- Implement more advanced feature engineering techniques
- Explore ensemble methods like stacking
- Try more advanced models like XGBoost
- Deploy the model with a simple web interface
