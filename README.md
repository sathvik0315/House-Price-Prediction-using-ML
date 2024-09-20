# House-Price-Prediction-using-ML

**Objective**
- I have built a Model using Random Forest Regressor of California Housing Prices Dataset to predict the price of the Houses in California.

## Code
- import sys
- import os
- import tarfile
- import urllib.request
- import numpy as np
- import pandas as pd
- from sklearn.model_selection import train_test_split
- from sklearn.model_selection import cross_val_score
- from sklearn.model_selection import GridSearchCV
- from sklearn.model_selection import StratifiedShuffleSplit
- from pandas.plotting import scatter_matrix
- from sklearn.impute import SimpleImputer
- from sklearn.preprocessing import OrdinalEncoder
- from sklearn.preprocessing import OneHotEncoder
- from sklearn.pipeline import Pipeline
- from sklearn.pipeline import FeatureUnion
- from sklearn.preprocessing import StandardScaler
- from sklearn.compose import ColumnTransformer
- from sklearn.linear_model import LinearRegression
- from sklearn.metrics import mean_squared_error
- from sklearn.metrics import mean_absolute_error
- from sklearn.tree import DecisionTreeRegressor
- from sklearn.ensemble import RandomForestRegressor
- from sklearn.model_selection import RandomizedSearchCV



# Model Training Overview

## Training the Model
- **Feature Scaling**: Essential for machine learning algorithms to perform well, as they struggle with input numerical attributes on different scales. Common techniques include:
  - Min-Max Scaling
  - Standardization
- **Pipeline Class**: Utilized in Scikit-Learn to manage the sequence of transformations.
- **Column Handling**: Categorical and numerical columns are processed separately using the `ColumnTransformer` class to apply appropriate transformations to each column.

### Decision Trees
- Initial performance with Linear Regression indicates underfitting, where the model lacks sufficient power or features.
- **Solution**: Train with Decision Trees, a more powerful model.

### Cross Validation
- Utilizes Scikit-Learn’s Cross Validation for model evaluation, producing an array of evaluation scores.
- Scoring functions must be configured for greater-is-better utility rather than lower-is-better cost functions (i.e., use the negative of MSE).

## Random Forest Regressor
- **Mechanism**: Trains multiple Decision Trees on random subsets of features and averages predictions.
- **Ensemble Learning**: Combines multiple models to enhance machine learning performance.

### Hyperparameter Optimization
- **Grid Search**: Evaluates all possible combinations of hyperparameter values using Cross Validation, suitable for smaller search spaces.
- **Randomized Search**: Preferred for larger search spaces, evaluating a specified number of random combinations by selecting random values for hyperparameters at each iteration.

