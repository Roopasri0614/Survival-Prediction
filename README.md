# Survival-Prediction
To build a model that predicts whether a passenger on the Titanic survived or not.
# Titanic Survival Prediction

## Overview
This project builds a machine learning model to predict whether a passenger on the Titanic survived or not, using historical passenger data. The dataset includes features such as age, gender, ticket class, fare, cabin, and more.

## Dataset
The dataset used for this project is the Titanic dataset, which can be found on [Kaggle](https://www.kaggle.com/competitions/titanic). It contains the following key columns:
- **PassengerId**: Unique identifier for each passenger
- **Survived**: Target variable (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Name**: Passenger's name (dropped in preprocessing)
- **Sex**: Gender
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number (dropped in preprocessing)
- **Fare**: Passenger fare
- **Cabin**: Cabin number (dropped due to missing data)
- **Embarked**: Port of Embarkation

## Installation
To run this project, you need Python 3 and the following dependencies:

```sh
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Steps to Run the Model

1. **Load the dataset**: Read the Titanic CSV file into a Pandas DataFrame.
2. **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical variables (Sex, Embarked).
   - Drop irrelevant columns (Name, Ticket, Cabin).
3. **Split the Data**:
   - Separate features (X) and target (y).
   - Split into training and testing sets.
4. **Train a Model**:
   - Use a classifier like RandomForest, Logistic Regression, or XGBoost.
5. **Evaluate the Model**:
   - Use accuracy, precision, recall, and F1-score for evaluation.

## Example Code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("titanic.csv")
df.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Results
- The model achieves an accuracy of around **80-85%** depending on hyperparameters and preprocessing.
- Feature importance analysis shows that **Pclass, Sex, and Age** are significant predictors.

## Future Improvements
- Use hyperparameter tuning (GridSearchCV) to optimize the model.
- Try different algorithms like XGBoost and Neural Networks.
- Perform feature engineering to create new meaningful features.

## License
This project is open-source and free to use.

---

Feel free to modify and expand this README based on your specific implementation!
