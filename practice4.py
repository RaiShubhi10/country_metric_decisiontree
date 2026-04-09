import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text,DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['figure.figsize'] = (2.0, 1.0)
matplotlib.rcParams['font.size'] = 2

raw_df = pd.read_csv('countries_metric - Sheet1.csv')

def clean_number(value):
    """
    Converts strings like:
    '1,42,86,27,663'
    '$4.187 trillion'
    '$467.22 billion'
    '485 million'
    into float values
    """
    if pd.isna(value):
        return np.nan

    value = str(value).lower().replace(',', '').replace('$', '').strip()

    if 'trillion' in value:
        return float(value.replace('trillion', '').strip()) * 1e12
    if 'billion' in value:
        return float(value.replace('billion', '').strip()) * 1e9
    if 'million' in value:
        return float(value.replace('million', '').strip()) * 1e6

    return float(value)

cols_to_clean = [
    'Population (in millions)',
    'Nominal Gross Domestic Product (in USD)',
    'Nominal GDP Per capita (in USD)'
]

for col in cols_to_clean:
    raw_df[col] = raw_df[col].apply(clean_number)

input_cols = [
    'Population (in millions)',
    'Nominal GDP Per capita (in USD)'
]

target_col = 'Nominal Gross Domestic Product (in USD)'

X = raw_df[input_cols]
y = raw_df[target_col]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

imputer = SimpleImputer(strategy='median')

X_train = imputer.fit_transform(X_train)
X_val   = imputer.transform(X_val)
X_test  = imputer.transform(X_test)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)

model.fit(X_train, y_train)

train_preds = model.predict(X_train)
val_preds   = model.predict(X_val)
test_preds  = model.predict(X_test)

print("TRAIN R²:", r2_score(y_train, train_preds))
print("VAL R²  :", r2_score(y_val, val_preds))
print("TEST R² :", r2_score(y_test, test_preds))

print("\nTRAIN RMSE:", np.sqrt(mean_squared_error(y_train, train_preds)))
print("VAL RMSE  :", np.sqrt(mean_squared_error(y_val, val_preds)))
print("TEST RMSE :", np.sqrt(mean_squared_error(y_test, test_preds)))

plt.figure(figsize=(14, 7))
plot_tree(
    model,
    feature_names=input_cols,
    filled=True,
    max_depth=3
)
plt.title("Decision Tree Regression (GDP Prediction)")
plt.show()

feature_importance = pd.Series(
    model.feature_importances_,
    index=input_cols
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

