import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

@st.cache_data
def load_model_and_pipeline():
    model = joblib.load("final_decision_tree_model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    return model, pipeline

@st.cache_data
def load_data_and_mlb():
    df = pd.read_csv(r"D:\summer 2_nd\Datasets\car_mrsp.csv")
    
    # Build the MultiLabelBinarizer dynamically from Market Category
    market_cat = df["Market Category"].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    mlb.fit(market_cat)
    
    return df, mlb

model, pipeline = load_model_and_pipeline()
df, mlb = load_data_and_mlb()

st.title("Car Price Predictor")
st.write("Enter car details below to predict **MSRP (Manufacturer's Suggested Retail Price)**.")

make = st.selectbox("Make", df["Make"].dropna().unique())
model_name = st.selectbox("Model", df[df["Make"] == make]["Model"].dropna().unique())
year = st.number_input("Year", min_value=1990, max_value=2025, value=2020)
engine_hp = st.number_input("Engine HP", min_value=50, max_value=1500, value=200)
fuel = st.selectbox("Fuel Type", df["Engine Fuel Type"].dropna().unique())
cylinder = st.selectbox("Engine Cylinders", sorted(df["Engine Cylinders"].dropna().unique()))
transmission = st.selectbox("Transmission", df["Transmission Type"].dropna().unique())
wheels = st.selectbox("Driven Wheels", df["Driven_Wheels"].dropna().unique())
vehicle_size = st.selectbox("Vehicle Size", df["Vehicle Size"].dropna().unique())
vehicle_style = st.selectbox("Vehicle Style", df["Vehicle Style"].dropna().unique())
highway_mpg = st.number_input("Highway MPG", min_value=10, max_value=100, value=30)
city_mpg = st.number_input("City MPG", min_value=5, max_value=100, value=20)
popularity = st.number_input("Popularity", min_value=0, max_value=10000, value=1000)


all_market_cats = sorted(set(', '.join(df["Market Category"].dropna()).split(', ')))
market_category = st.multiselect("Market Category", all_market_cats)

if st.button("Predict MSRP"):

  
    input_dict = {
        "Make": make,
        "Model": model_name,
        "Year": year,
        "Engine HP": engine_hp,
        "Engine Fuel Type": fuel,
        "Engine Cylinders": cylinder,
        "Transmission Type": transmission,
        "Driven_Wheels": wheels,
        "Vehicle Size": vehicle_size,
        "Vehicle Style": vehicle_style,
        "highway MPG": highway_mpg,
        "city mpg": city_mpg,
        "Popularity": popularity,
    }
    input_df = pd.DataFrame([input_dict])

    
    transformed = pipeline.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

 
    encoded_market_cat = mlb.transform([market_category])

    final_input = np.hstack([transformed, encoded_market_cat])

    prediction = model.predict(final_input)[0]
    st.success(f"💰 Predicted MSRP: **${int(prediction):,}**")
