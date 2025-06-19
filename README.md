#  Car MSRP Predictor Dataset

This dataset is designed for building a machine learning model that predicts the **Manufacturer's Suggested Retail Price (MSRP)** of cars based on various features such as make, model, fuel type, drivetrain, engine specifications, and more.

---

## 📁 Dataset Overview

Data set link ("https://www.kaggle.com/datasets/CooperUnion/cardataset/data")


---

## 📊 Features

| Column Name            | Description                                        | Data Type    |
|------------------------|----------------------------------------------------|--------------|
| `Make`                 | Car manufacturer                                   | Categorical  |
| `Model`                | Specific model name                                | Categorical  |
| `Year`                 | Year of manufacture                                | Numerical    |
| `Engine Fuel Type`     | Type of fuel used                                  | Categorical  |
| `Engine HP`            | Horsepower of the engine                           | Numerical    |
| `Engine Cylinders`     | Number of cylinders in the engine                  | Numerical    |
| `Transmission Type`    | Type of transmission                               | Categorical  |
| `Driven_Wheels`        | Drivetrain (FWD, RWD, AWD)                         | Categorical  |
| `Number of Doors`      | Door count                                         | Numerical    |
| `Vehicle Size`         | Vehicle classification (Compact, Midsize, etc.)   | Categorical  |
| `Vehicle Style`        | Body style (Sedan, SUV, Coupe, etc.)              | Categorical  |
| `highway MPG`          | Fuel efficiency on highways                        | Numerical    |
| `city MPG`             | Fuel efficiency in city                            | Numerical    |
| `Popularity`           | Popularity score based on market demand           | Numerical    |
| `Market Category`      | Marketing tags (e.g., Luxury, Performance)         | Multi-label  |
| `MSRP`                 | Manufacturer’s Suggested Retail Price (Target)     | Numerical    |

---

##  Use Case

This dataset is primarily used to:

- Train regression models (e.g., Decision Trees, Linear Regression)
- Explore relationships between car features and pricing
- Perform feature engineering (e.g., label encoding, one-hot encoding)
- Build a web app (e.g., with Streamlit) for MSRP prediction

---

## Libraries

- `pandas` for data preprocessing  
- `scikit-learn` for modeling and pipelines  
- `joblib` for saving the model  
- `Streamlit` for creating an interactive dashboard  

---

## File Info

- Format: `.csv`
- Rows: ~10,000+
- Source: Aggregated from public automotive listings

---

## How to use?

- Download the files from github
- place the model and pipeline in same directory of main.py or manually set the path 
- Make sure to install streamlit , pandas, numpy, sklearn beforehand
- run the main.py through Streamlit run main.py command in terminal

---
## Libraries
![Image](https://github.com/user-attachments/assets/7e813c7c-8263-4e77-a1ab-88c9149d4a95)

