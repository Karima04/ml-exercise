
# Imports
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pickle

# Input
data_filename = "data/auto-mpg.csv"

# Read csv-data
data = pd.read_csv(data_filename, sep=';')

# Split data
x = data.drop(["mpg"], axis=1)  # = data.drop(data.columns[0], axis=1)
y = data["mpg"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Normalize train data
mmscaler = MinMaxScaler()
x_train_norm = mmscaler.fit_transform(x_train)
x_test_norm = mmscaler.transform(x_test)

# Load the model
model_path = "data/models/"
model_filename = 'trained_lreg_model.sav'
loaded_lreg_model = pickle.load(open(model_path + model_filename, 'rb'))

# Predict
result = loaded_lreg_model.score(x_test_norm, y_test)*100

# Output
stern = '*'*100
print("\n {}\n y-Achsen-Abschnitt: {:.2f}".format(stern, loaded_lreg_model.intercept_))
print(" Steigung:", loaded_lreg_model.coef_)
print(f" Regressor score R²: {result:.2f}% \n {stern}\n")

# Plot PS and MPG test data
plt.scatter(x_test_norm[:,2:3], y_test)
plt.xlabel('PS')
plt.ylabel('mpg')
plt.show()
