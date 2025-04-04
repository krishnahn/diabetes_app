import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import numpy as np

# App Title
st.title("Personalized Diabetes Management Platform")

# Introduction
st.markdown("""
Type 2 Diabetes Mellitus (T2DM) affects millions globally, but many are unaware of the potential for remission through lifestyle changes. 
This platform aims to empower individuals to manage and potentially reverse their diabetes through personalized recommendations and progress tracking.
""")

# Blood Glucose Input
st.header("Blood Glucose Input")
blood_glucose_level = st.number_input("Enter your current blood glucose level (mg/dL):", min_value=0)

# Personalized Recommendations
st.header("Personalized Recommendations")

if blood_glucose_level > 200:
    st.warning("Your blood glucose level is high. Consider consulting with your doctor.")
    st.subheader("Recommended Dietary Changes:")
    st.markdown("""
    - **Focus on whole, unprocessed foods:** Choose fruits, vegetables, whole grains, lean proteins, and healthy fats.
    - **Limit processed foods, sugary drinks, and unhealthy fats:** These can contribute to high blood sugar levels.
    - **Control portion sizes:** Be mindful of how much you're eating to manage calorie intake.
    - **Choose foods with a low glycemic index (GI):** These foods release glucose into your bloodstream more slowly.
    - **Drink plenty of water:** Water helps to flush out excess sugar and keeps you hydrated.
    """)
    # Add more specific recommendations or resources as needed
elif blood_glucose_level > 140:
    st.warning("Your blood glucose level is slightly elevated. Monitor it closely.")
    st.markdown("Consider making healthy dietary choices and increasing physical activity.")
else:
    st.info("Your blood glucose level is within a healthy range. Maintain a healthy lifestyle.")




# Personalized Recommendations
st.header("Personalized Recommendations")

if blood_glucose_level >= 200:
    st.warning("Your blood glucose level is high. Consider consulting with your doctor.")
    st.subheader("Recommended Dietary Changes:")
    st.markdown("""
    - **Focus on whole, unprocessed foods:** Choose fruits, vegetables, whole grains, lean proteins, and healthy fats.
    - **Limit processed foods, sugary drinks, and unhealthy fats:** These can contribute to high blood sugar levels.
    - **Control portion sizes:** Be mindful of how much you're eating to manage calorie intake.
    - **Choose foods with a low glycemic index (GI):** These foods release glucose into your bloodstream more slowly.
    - **Drink plenty of water:** Water helps to flush out excess sugar and keeps you hydrated.
    """)
    
    # --- Diet Management ---
    st.subheader("Sample Meal Plan (Consult with a dietitian for personalized plans):")
    st.markdown("""
    **Breakfast:** Oatmeal with berries and nuts, Greek yogurt with fruit, or eggs with vegetables.
    **Lunch:** Salad with grilled chicken or fish, lentil soup, or a whole-wheat sandwich with lean protein and vegetables.
    **Dinner:** Grilled salmon with roasted vegetables, chicken stir-fry with brown rice, or lentil stew.
    **Snacks:** Fruits, vegetables with hummus, nuts, or plain yogurt.
    """)


# Progress Tracking
st.header("Progress Tracking")
st.subheader("Enter your blood glucose levels for the past four months:")

months = ["Month 1", "Month 2", "Month 3", "Month 4"]
glucose_data = []

for month in months:
    glucose_level = st.number_input(f"Enter average blood glucose level for {month} (mg/dL):", min_value=0)
    glucose_data.append(glucose_level)

blood_glucose_data = pd.DataFrame({"Month": months, "Blood Glucose Level (mg/dL)": glucose_data})

fig = px.line(blood_glucose_data, x="Month", y="Blood Glucose Level (mg/dL)",
              title="Blood Glucose Level Trends Over Four Months", markers=True)
st.plotly_chart(fig)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Sample Dataset (Replace with your actual data if available)
data = pd.DataFrame({
    'age': [45, 50, 35, 60, 25, 30, 55, 40, 65, 38],
    'bmi': [28, 32, 25, 35, 22, 27, 30, 29, 33, 26],
    'blood_glucose_level': [150, 180, 120, 200, 110, 130, 170, 140, 190, 125]
})

# Preprocessing
X = data[['age', 'bmi']]
y = data['blood_glucose_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Machine Learning Model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("Diabetes Prediction")

st.header("User Input")
age = st.number_input("Enter your age:", min_value=18, max_value=100, value=40)
bmi = st.number_input("Enter your BMI:", min_value=15, max_value=50, value=25)

user_input = [[age, bmi]]
user_input = scaler.transform(user_input)
prediction = model.predict(user_input)[0]

st.header("Prediction")
st.write(f"Predicted Blood Glucose Level: {prediction:.2f} mg/dL")


st.title("Diabetes Prediction Tool")

# Enhanced Linear Regression Graph


# Create a meshgrid of points for visualization
age_range = np.linspace(data['age'].min() - 5, data['age'].max() + 5, 100)  # Extended range
bmi_range = np.linspace(data['bmi'].min() - 5, data['bmi'].max() + 5, 100)  # Extended range
age_grid, bmi_grid = np.meshgrid(age_range, bmi_range)
features_grid = np.column_stack((age_grid.ravel(), bmi_grid.ravel()))
features_grid_scaled = scaler.transform(features_grid)

# Predict blood glucose levels for the meshgrid points
predictions_grid = model.predict(features_grid_scaled)

# Plot the data and the linear regression plane
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(data['age'], data['bmi'], c=data['blood_glucose_level'], cmap='viridis', s=100, edgecolors='k')
contour = ax.contour(age_grid, bmi_grid, predictions_grid.reshape(age_grid.shape), levels=10, colors='black')
ax.clabel(contour, inline=True, fontsize=10, fmt='%.0f')  # Add contour labels
ax.set_xlabel("Age", fontsize=14)
ax.set_ylabel("BMI", fontsize=14)
ax.set_title("Linear Regression: Age vs. BMI vs. Blood Glucose Level", fontsize=16)
cbar = fig.colorbar(scatter, ax=ax, label="Blood Glucose Level (mg/dL)")
cbar.ax.tick_params(labelsize=12)
st.pyplot(fig)

# Streamlit App
st.title("Diabetes Prediction")

# Creative Linear Regression Graph with Plotly
st.header("Interactive Prediction Visualization")

# ... (meshgrid creation, prediction on meshgrid) ...


# Ensure data types are compatible with Plotly
age_grid = age_grid.astype(np.float64)
bmi_grid = bmi_grid.astype(np.float64)
predictions_grid = predictions_grid.astype(np.float64)
data['age'] = data['age'].astype(np.float64)
data['bmi'] = data['bmi'].astype(np.float64)
data['blood_glucose_level'] = data['blood_glucose_level'].astype(np.float64)

# Create Plotly figure
fig = go.Figure(data=[
    go.Scatter3d(x=data['age'], y=data['bmi'], z=data['blood_glucose_level'],
                mode='markers', marker=dict(size=8, color=data['blood_glucose_level'], 
                                           colorscale='Viridis', opacity=0.8)),
    go.Surface(x=age_grid, y=bmi_grid, z=predictions_grid.reshape(age_grid.shape),
                colorscale='Blues', opacity=0.5)
])


st.plotly_chart(fig)
