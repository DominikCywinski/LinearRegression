import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import PredictionErrorDisplay


def plot_prediction_validation(test, pred, file_name):
    plt.scatter(test, pred)
    plt.xlabel('Validation')
    plt.ylabel('Prediction')
    plt.savefig(file_name)


def plot_cross_validated_predictions(test, pred, file_name):
    fig, axs = plt.subplots(ncols = 2, figsize = (8, 4))
    PredictionErrorDisplay.from_predictions(
        test,
        y_pred = pred,
        kind = "actual_vs_predicted",
        subsample = 100,
        ax = axs[0],
        random_state = 0,
    )
    axs[0].set_title("Actual vs. Predicted values")

    PredictionErrorDisplay.from_predictions(
        test,
        y_pred = pred,
        kind = "residual_vs_predicted",
        subsample = 100,
        ax = axs[1],
        random_state = 0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    plt.savefig(file_name)


data_path = './input_data/medical_insurance.csv'
medical_insurance = pd.read_csv(data_path)

X = medical_insurance.drop(['charges'], axis = 1)
y = medical_insurance['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

X_train = X_train.dropna()
X_test = X_test.dropna()

categorical_cols = [col_name for col_name in X_train.columns if X_train[col_name].dtype == object]
numerical_cols = [col_name for col_name in X_train.columns if X_train[col_name].dtype in ['int64', 'float64']]

numerical_transformer = Pipeline(steps = [('poly', PolynomialFeatures(degree = 2)),
                                          ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps = [
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(transformers = [
    ('numerical', numerical_transformer, numerical_cols),
    ('categorical', categorical_transformer, categorical_cols)
])

pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('linear_model', LinearRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

scores = -1 * cross_val_score(pipeline, X_test, y_test, cv = 5, scoring = 'neg_mean_absolute_error')
print('MAE scores:\n', scores)
print('average MAE score:\n', scores.mean())

plot_prediction_validation(y_test, predictions, 'predictions_validation.png')
plot_cross_validated_predictions(y_test, predictions, 'cross_validated_predictions.png')
