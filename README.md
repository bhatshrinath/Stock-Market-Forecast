# Stock-Market-Forecast

This project focuses on forecasting the Indian stock market using various machine learning models and techniques.

## Models and Techniques Used

The following models and techniques from `scikit-learn` and `xgboost` are used in this project:

- Linear Models:
  - `LinearRegression`
  - `Lasso`
  - `Ridge`
  - `ElasticNet`
  - `HuberRegressor`
- Neighbors-based Regression:
  - `KNeighborsRegressor`
- Tree-based Regression:
  - `DecisionTreeRegressor`
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
  - `StackingRegressor`
  - `VotingRegressor`
  - `AdaBoostRegressor`
  - `ExtraTreesRegressor`
  - `HistGradientBoostingRegressor`
  - `BaggingRegressor`
- Preprocessing:
  - `StandardScaler`
  - `RobustScaler`
  - `OneHotEncoder`
- Feature Selection:
  - `RFECV`
- Pipelines:
  - `Pipeline`
- Model Selection:
  - `train_test_split`
  - `GridSearchCV`
  - `TimeSeriesSplit`
  - `cross_val_score`
  - `KFold`
- Metrics:
  - `make_scorer`
- Gradient Boosting:
  - `XGBRegressor` from `xgboost`

## Getting Started

To get started with the project, clone the repository and install the required dependencies.

```sh
git clone <repository-url>
cd Stock-Market-Forecast
pip install -r requirements.txt