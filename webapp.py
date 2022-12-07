import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

st.write("""
# Red-Wine pH Prediction App
""")
st.write('---')

dataset = pd.read_csv('winequality-red.csv')
X = dataset.drop('pH', axis=1)
Y = dataset['pH']


st.sidebar.header('Input Params')

def create_object_field_and_slider_value(column_name):
  return { 
    column_name: st.sidebar.slider(column_name, 
    float(int(X[column_name].min())), 
    float(round(X[column_name].max())), 
    float(round(X[column_name].mean())), 0.01) 
    }

def user_input_features():
  field_elements = map(create_object_field_and_slider_value, X.columns)
  data = {}

  for element in field_elements:
    data.update(element)

  features = pd.DataFrame(data, index=[0])

  return features

df = user_input_features()


# Main


st.header('Specified Input parameters')
st.write(df)
st.write('---') 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

regression = LinearRegression()
regression.fit(X_train, y_train)

prediction = regression.predict(df)
y_pred = regression.predict(X_test)

st.header('Prediction of ph')
st.write(prediction)
st.write('---')

st.header('Mean Squared Error:')
st.write(mean_squared_error(y_test, y_pred))
st.write('---')

