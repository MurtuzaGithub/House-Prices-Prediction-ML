from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import os
from django.conf import settings

def home(request):
    return render(request,  "home.html")

def about(request):
    return render(request,  "about.html")

def predict(request):
    return render(request,  "predict.html")

def result(request):
    file_path = os.path.join(settings.BASE_DIR, 'USA_Housing.csv')
    data = pd.read_csv(file_path)

    data = data.drop(['Address'], axis=1)

    X = data.drop('Price', axis=1)
    Y = data['Price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])

    # Reshape the input into a 2D array with a single row
    input_data = np.array([[var1, var2, var3, var4, var5]])

    pred = model.predict(input_data)
    pred = round(pred[0])
    price = "The Predicted Price is $" + str(pred)

    return render(request, "predict.html", {"result2": price})



