# small streamlit demo. run with: streamlit run streamlit_demo.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title('iris predictor')

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=50).fit(X, y)

sl = st.slider('sepal length', 4.0, 8.0, 5.0)
sw = st.slider('sepal width', 2.0, 5.0, 3.0)
pl = st.slider('petal length', 1.0, 7.0, 4.0)
pw = st.slider('petal width', 0.1, 3.0, 1.2)

pred = clf.predict([[sl, sw, pl, pw]])[0]
names = ['setosa', 'versicolor', 'virginica']
st.write('prediction:', names[pred])
