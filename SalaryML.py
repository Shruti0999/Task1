import pandas
import joblib
ds = pandas.read_csv('SalaryData.csv')
ds.info()
x = ds['YearsExperience'].values.reshape(-1, 1)
type(x)
y = ds['Salary']
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)
model.copy_X
model.coef_
model.intercept_

'''Predicts the output using this file'''
print("|||||||||| Welcome ||||||||||")

print(" ")

print(" ")

exp = float(input("Enter your exprerience : "))

result = model.predict([[exp]])
print(" ")

print("|||||||||| The predicted Salary is about ||||||||||")
print(result)
print(" ")

joblib.dump( model ,"SalaryModel.pkl")
