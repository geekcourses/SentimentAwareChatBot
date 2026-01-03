import streamlit as st
import requests
from transformers import pipeline

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Hybrid Sentiment Bot", layout="wide")

# Hugging Face API URL for Microsoft Phi-3 Mini
API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
EMO_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# --- 2. HELPER FUNCTIONS ---

@st.cache_resource
def get_analyzer():
    """Loads the emotion detection model once and caches it."""
    # This runs locally on the cloud server (CPU)
    return pipeline("text-classification", model=EMO_MODEL)

def get_system_instruction(emotion):
    """Returns the personality instructions based on emotion."""
    behaviors = {
        "anger":   "The user is ANGRY. Be calm, empathetic, and try to de-escalate.",
        "joy":     "The user is HAPPY. Be energetic and share their excitement!",
        "sadness": "The user is SAD. Be supportive, gentle, and very comforting.",
        "default": "Be a helpful and professional AI assistant."
    }
    return behaviors.get(emotion, behaviors["default"])

def query_phi3_api(prompt_text):
    """Sends the prompt to Hugging Face Cloud and returns the text."""
    # Check if token exists in secrets
    if "HF_TOKEN" not in st.secrets:
        st.error("⚠️ Missing Hugging Face Token. Please add it to Streamlit Secrets.")
        return "Error: No API Token found."

    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

    # Payload for Phi-3
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": 500,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.text}"

# --- 3. STATE & UI ---

st.title("🤖 Emotional Support Bot (Cloud Version)")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Analytics
with st.sidebar:
    st.header("Analytics")
    metric_placeholder = st.empty() # Placeholder for immediate updates

    # Initialize the metric with a default value so it's not invisible
    metric_placeholder.metric("Detected Emotion", "Waiting...")

    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# Draw Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 4. MAIN CHAT LOOP ---

if user_input := st.chat_input("How are you feeling?"):

    # Step A: Show User Input
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Step B: Analyze Emotion (Locally)
    analyzer = get_analyzer()
    result = analyzer(user_input)[0]
    emotion_label = result['label']

    # Update the sidebar state immediately
    metric_placeholder.metric("Detected Emotion", emotion_label.upper())

    # Step C: Prepare Prompt for Cloud API
    instruction = get_system_instruction(emotion_label)

    # We must format the string manually for the API since it's not a chat object
    # Phi-3 Format: <|user|> Question <|end|> <|assistant|>
    # We prepend the system instruction to the user's input
    full_prompt = f"<|user|>\nSystem Instruction: {instruction}\n\nUser Message: {user_input}<|end|>\n<|assistant|>"

    # Step D: Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Phi-3 is thinking..."):
            ai_response = query_phi3_api(full_prompt)
            st.write(ai_response)

    # Step E: Save and Refresh
    st.session_state.messages.append({"role": "assistant", "content": ai_response})