# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle


# Preprocess the data
# Load the dataset
df = pd.read_csv('data/rental_1000.csv')

# Feature engineering and selection of  Features and Label 
X = df[['rooms','area']].values
y = df['price'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
lr = LinearRegression()
model = lr.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Actual Prices Vs Predicted Prices ", y_test[:5], predictions[:5])

score = model.score(X_test, y_test)
rmse = np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2))
print(f"Model Score: {score}")
print(f"RMSE: {rmse}")

# Save the model
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Predict using the model
def predict_price(rooms, area):
    # Load the model
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Make a prediction
    prediction = model.predict([[rooms, area]])
    return prediction[0]

print(predict_price(3, 120))  # Example prediction for 3 rooms and 120 area