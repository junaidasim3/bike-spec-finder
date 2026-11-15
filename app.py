import streamlit as st
import pandas as pd
import joblib
import glob
import os

# Page Configuration
st.set_page_config(page_title="Bike Spec & ML Finder", page_icon="🏍️", layout="wide")

st.title("🏍️ Bike Specifications & Decision Tree App")
st.write("Search for your favorite bike models or use the trained Decision Tree model to make predictions!")

# --- 1. Auto-discover Dataset (.csv) ---
@st.cache_data
def load_data():
    files = glob.glob("**/*.csv", recursive=True) + glob.glob("/kaggle/input/**/*.csv", recursive=True)
    if files:
        return pd.read_csv(files[0])
    return None

df = load_data()

# --- 2. Load Trained Model (.pkl) ---
@st.cache_resource
def load_model():
    model_files = glob.glob("**/*.pkl", recursive=True)
    if model_files:
        return joblib.load(model_files[0])
    return None

model = load_model()

if df is not None:
    # Create Tabs for Clean Layout
    tab1, tab2 = st.tabs(["🔍 Search Bike Specs", "🤖 Decision Tree Predictor"])
    
    with tab1:
        st.header("Bike Model Specification Search")
        search_query = st.text_input("Enter bike model name (e.g., Yamaha, Honda, Ducati, Kawasaki):", "")
        
        if search_query:
            # Dynamically find the model/name column in your dataset
            name_col = next((col for col in df.columns if 'model' in col.lower() or 'name' in col.lower() or 'bike' in col.lower()), df.columns[0])
            results = df[df[name_col].astype(str).str.contains(search_query, case=False, na=False)]
            
            if not results.empty:
                st.success(f"Found {len(results)} matching bike(s):")
                for idx, row in results.iterrows():
                    with st.expander(f"🚲 {row[name_col]}"):
                        st.json(row.to_dict()) # Displays all specs (HP, torque, wheelbase, piston size, etc.)
            else:
                st.warning("No matching bike found. Try a different keyword.")
        else:
            st.info("Type a bike name above to look up its full specifications.")
            
    with tab2:
        st.header("Decision Tree Classification / Prediction")
        if model is not None:
            st.write("Provide input specifications below to see what category or target your Decision Tree predicts:")
            
            # Input fields for prediction features (adjust names/defaults to match your dataset features)
            col1, col2 = st.columns(2)
            with col1:
                hp = st.number_input("Horsepower (HP)", min_value=1.0, max_value=300.0, value=50.0)
                torque = st.number_input("Torque (Nm)", min_value=1.0, max_value=300.0, value=45.0)
            with col2:
                weight = st.number_input("Weight (kg)", min_value=50.0, max_value=500.0, value=150.0)
                displacement = st.number_input("Displacement (cc)", min_value=50.0, max_value=2500.0, value=600.0)
            
            if st.button("Run Prediction"):
                try:
                    # Note: Ensure the column order matches what you used during training!
                    input_df = pd.DataFrame([[hp, torque, weight, displacement]], columns=['HP', 'Torque', 'Weight', 'Displacement'])
                    prediction = model.predict(input_df)
                    st.success(f"🎯 Predicted Result / Category: **{prediction[0]}**")
                except Exception as e:
                    st.error(f"Prediction Error: {e}. Make sure the input feature names match your trained model's features.")
        else:
            st.warning("⚠️ No `.pkl` model file found in your repository. Please upload your trained model file.")
else:
    st.error("❌ Dataset CSV file not found. Please upload your dataset CSV into your GitHub repository.")