import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Чтение данных из файла
df = pd.read_csv("CarPrice_Assignment.csv")

# Выбор нужных признаков и целевой переменной
X = df[['horsepower']]
y = df['price']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Обучение модели
model = RandomForestRegressor().fit(X_train, y_train)

# Расчет предсказаний
prediction = model.predict(X_test)

# Вывод точности модели
print(f"Точность обучающей сборки: {model.score(X_train, y_train)}")
print(f"Точность тестовой сборки: {model.score(X_test, y_test)}")

# Построение графиков и диаграмм
plt.figure(figsize=(10, 6))

# График фактических значений и предсказаний
plt.scatter(X_test, y_test, color='b', label='Actual')
plt.scatter(X_test, prediction, color='r', label='Predicted')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

# График рассеяния фактических значений и остатков
residuals = y_test - prediction
plt.figure(figsize=(10, 6))
plt.scatter(X_test, residuals, color='g')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Horsepower')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

# Гистограмма остатков
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='m', bins=20)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Histogram')
plt.show()