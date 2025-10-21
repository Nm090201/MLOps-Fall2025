import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

# FastAPI backend endpoint
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

# California housing model location
FASTAPI_HOUSING_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'FastAPI_Labs' / 'model' / 'california_model.pkl'

# Streamlit logger
LOGGER = get_logger(__name__)

def run():
    # Set dashboard config
    st.set_page_config(
        page_title="California Housing Prediction Demo",
        page_icon="🏡",
    )

    # Sidebar
    with st.sidebar:
        # Check backend status
        try:
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT)
            if backend_request.status_code == 200:
                st.success("Backend online ✅")
            else:
                st.warning("Problem connecting 😭")
        except requests.ConnectionError as ce:
            LOGGER.error(ce)
            st.error("Backend offline 😱")

        st.info("Configure housing features")

        # Manual inputs for 8 features
        MedInc = st.number_input("Median Income (in tens of thousands)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        HouseAge = st.slider("House Age", 1, 100, 20)
        AveRooms = st.number_input("Average Rooms", min_value=1.0, max_value=20.0, value=6.0, step=0.1)
        AveBedrms = st.number_input("Average Bedrooms", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
        Population = st.number_input("Population", min_value=1.0, max_value=10000.0, value=500.0, step=1.0)
        AveOccup = st.number_input("Average Occupancy", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
        Latitude = st.slider("Latitude", 32.0, 42.0, 35.0, step=0.01)
        Longitude = st.slider("Longitude", -125.0, -114.0, -120.0, step=0.01)

        # Predict button
        predict_button = st.button('Predict')

    # Main Dashboard
    st.write("# 🏡 California Housing Price Prediction")

    if predict_button:
        if FASTAPI_HOUSING_MODEL_LOCATION.is_file():
            # Build input dictionary
            client_input = json.dumps({
                "MedInc": MedInc,
                "HouseAge": HouseAge,
                "AveRooms": AveRooms,
                "AveBedrms": AveBedrms,
                "Population": Population,
                "AveOccup": AveOccup,
                "Latitude": Latitude,
                "Longitude": Longitude
            })

            try:
                result_container = st.empty()
                with st.spinner('Predicting...'):
                    predict_response = requests.post(f'{FASTAPI_BACKEND_ENDPOINT}/predict', data=client_input)

                if predict_response.status_code == 200:
                    housing_content = json.loads(predict_response.content)
                    predicted_value = housing_content["response"]
                    result_container.success(f"🏠 Predicted Median House Value: **${predicted_value * 100000:.2f}**")
                else:
                    st.toast(f':red[Status from server: {predict_response.status_code}. Refresh page and check backend status]', icon="🔴")
            except Exception as e:
                st.toast(':red[Problem with backend. Refresh page and check backend status]', icon="🔴")
                LOGGER.error(e)
        else:
            LOGGER.warning('california_model.pkl not found. Please train and save the model first.')
            st.toast(':red[Model california_model.pkl not found. Please run train.py]', icon="🔥")

if __name__ == "__main__":
    run()
