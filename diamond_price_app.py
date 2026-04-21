import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("💎 Diamond Price Prediction")

# Load dataset
df = pd.read_csv("diamonds.csv")

# Remove invalid values
df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

# Feature engineering
df["volume"] = df["x"] * df["y"] * df["z"]
df["dimension_ratio"] = (df["x"] + df["y"]) / (2 * df["z"])
df["price_inr"] = df["price"] * 83

# Select useful columns
df_model = df[[
    "carat", "depth", "table", "x", "y", "z",
    "cut", "color", "clarity",
    "volume", "dimension_ratio", "price_inr"
]]

# One-hot encoding
df_encoded = pd.get_dummies(df_model, drop_first=True)

# Split features and target
X = df_encoded.drop("price_inr", axis=1)
y = df_encoded["price_inr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# UI inputs
st.subheader("Enter Diamond Details")

carat = st.number_input("Carat", min_value=0.1, max_value=5.0, value=1.0)
depth = st.number_input("Depth", min_value=40.0, max_value=80.0, value=60.0)
table = st.number_input("Table", min_value=40.0, max_value=80.0, value=57.0)
x_val = st.number_input("Length (x)", min_value=0.1, max_value=10.0, value=5.0)
y_val = st.number_input("Width (y)", min_value=0.1, max_value=10.0, value=5.0)
z_val = st.number_input("Height (z)", min_value=0.1, max_value=10.0, value=3.0)

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])

if st.button("Predict Price"):
    volume = x_val * y_val * z_val
    dimension_ratio = (x_val + y_val) / (2 * z_val)

    input_data = pd.DataFrame({
        "carat": [carat],
        "depth": [depth],
        "table": [table],
        "x": [x_val],
        "y": [y_val],
        "z": [z_val],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "volume": [volume],
        "dimension_ratio": [dimension_ratio]
    })

    # Encode input same as training
    input_encoded = pd.get_dummies(input_data, drop_first=True)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    prediction = max(0, prediction)

    st.success(f"💰 Predicted Price: ₹{int(prediction)}")