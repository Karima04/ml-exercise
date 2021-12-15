# Imports
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
x_train, x_test, y_train, y_test = train_test_split (x,y, test_size=0.3, random_state = 42)

# Normalize train data
mmscaler = MinMaxScaler()
x_train_norm = mmscaler.fit_transform(x_train)
x_test_norm = mmscaler.transform(x_test)

# Fit and Predict the model
lreg = LinearRegression().fit(x_train_norm, y_train)
y_pred= lreg.predict(x_test_norm)
print(y_pred)

# save the model to disk
model_path = "data/models/"
model_filename = 'trained_lreg_model.sav'
pickle.dump(lreg, open(model_path + model_filename, 'wb'))

# Output
model = model_filename.split('.sav')
stern = '*'*100
italic = '\033[3m'
bold = '\033[1m'
end = '\033[0m'
print(
    f'\n{stern}\n Fertig! Das trainierte Modell {italic} {bold}{model[0]}{end} ist erstellt und im Ordner {italic}{bold}{model_path}{end} zu finden \n{stern}\n')

