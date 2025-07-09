import streamlit as st
import requests
import json

# Set your API Gateway URL here
API_URL = "https://da3x22s1ch.execute-api.eu-west-2.amazonaws.com/dev/invoke-model"

st.title("DistilBERT-Powered News Headline Classification Pipeline")

# Define the class names
class_names = ["Business", "Science", "Entertainment", "Health"]

headline = st.text_input("Enter a news headline")

if st.button("Classify"):
    if headline:
        # Prepare the payload
        payload = {
            "query": {
                "headline": headline
            }
        }

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                raw_result = response.json()
                
                # First element is a JSON string, so parse it
                result_json = json.loads(raw_result[0])
                
                predicted_label = result_json.get("predicted_label", "Unknown")
                probabilities = result_json.get("probabilities", [[]])[0]

                st.success(f"🔮 Predicted Category: **{predicted_label}**")

                # Display class probabilities as a table or bar chart
                st.subheader("📊 Class Probabilities:")
                for name, prob in zip(class_names, probabilities):
                    st.write(f"- **{name}**: {round(prob * 100, 2)}%")

            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"⚠️ Failed to call API: {e}")
    else:
        st.warning("Please enter a headline.")