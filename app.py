
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")

st.title("Earthquake Analysis and Magnitude Prediction")

st.markdown("""
This dashboard provides an in-depth analysis of global earthquakes and a machine learning model to predict their magnitudes.
""")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('all_month.csv')
    df = df.drop(columns=['nst', 'gap', 'dmin', 'rms', 'net', 'id', 'updated', 'type', 'horizontalError', 'depthError', 'magError', 'magNst', 'status', 'locationSource', 'magSource'])
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['dayofweek'] = df['time'].dt.dayofweek
    df['country'] = df['place'].str.split(', ').str[-1]
    return df

df = load_data()

st.header("Earthquake Data")
st.write(df.head())

st.header("Geographical Distribution of Earthquakes")
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=2)
for i in range(0,len(df)):
    folium.Marker(
        location=[df.iloc[i]['latitude'], df.iloc[i]['longitude']],
        popup=df.iloc[i]['place'],
    ).add_to(m)
st_folium(m, width=1200, height=500)

st.header("Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Earthquake Magnitudes")
    fig, ax = plt.subplots()
    sns.histplot(df['mag'], bins=30, kde=True, ax=ax , color='green',)
    st.pyplot(fig)

with col2:
    st.subheader("Distribution of Earthquake Depths")
    fig, ax = plt.subplots()
    sns.histplot(df['depth'], bins=30, kde=True, ax=ax , color='red')
    st.pyplot(fig)

st.header("Magnitude Prediction Model")

features = ['latitude', 'longitude', 'depth', 'year', 'month', 'day', 'dayofweek']
target = 'mag'

df_ml = df.dropna(subset=features + [target])

X = df_ml[features]
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Mean Squared Error:** {mse}")
st.write(f"**R-squared:** {r2}")

st.subheader("Actual vs. Predicted Magnitude")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
ax.set_xlabel("Actual Magnitude")
ax.set_ylabel("Predicted Magnitude")
st.pyplot(fig)
