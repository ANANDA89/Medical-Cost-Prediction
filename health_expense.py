import pandas as pd
import numpy as np
import pickle
data= pd.read_csv('med-insurance.csv')
data['sex'] = data['sex'].replace(('male','female'), (2, 1))
data['smoker'] = data['smoker'].replace(('yes','no'), (2, 1))
data['region'] = data['region'].replace(('southeast','southwest','northeast','northwest'),(4, 1, 3, 2))
data['children'] = data['children'].replace((4, 5), (3, 3))
#data['expenses'] = np.log1p(data['expenses'])
# Model Building
y = data['expenses']
x = data.drop(['expenses','bmi'], axis = 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(x_train, y_train)
#y_pred = model.predict(x_test)
file = open('medical_expense.pkl', 'wb')
pickle.dump(model, file)
