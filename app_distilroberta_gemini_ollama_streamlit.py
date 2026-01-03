import streamlit as st
import ollama
from transformers import pipeline

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Hybrid Sentiment Bot", layout="wide")
LLM_MODEL = "phi3:latest"
EMO_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# --- 2. HELPER FUNCTIONS (The Logic) ---

@st.cache_resource
def get_analyzer():
    """Loads the emotion detection model once and caches it."""
    return pipeline("text-classification", model=EMO_MODEL)

def get_system_instruction(emotion):
    """Returns the personality instructions based on emotion."""
    # Define how the bot should behave for each emotion
    behaviors = {
        "anger":   "The user is ANGRY. Be calm, empathetic, and try to de-escalate.",
        "joy":     "The user is HAPPY. Be energetic and share their excitement!",
        "sadness": "The user is SAD. Be supportive, gentle, and very comforting.",
        "default": "Be a helpful and professional AI assistant."
    }
    return behaviors.get(emotion, behaviors["default"])

# --- 3. STATE & UI ---

st.title("🤖 Emotional Support Bot")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "Neutral"

# 1. Create an EMPTY slot in the sidebar at the top
with st.sidebar:
    st.header("Analytics")
    # Give it a variable name so we can talk to it later
    metric_placeholder = st.empty()

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

    # Step B: Analyze Emotion
    analyzer = get_analyzer()
    result = analyzer(user_input)[0]
    emotion_label = result['label']

    # Update the sidebar state immediately
    metric_placeholder.metric("Detected Emotion", emotion_label.upper())

    # Step C: Generate AI Response
    system_prompt = get_system_instruction(emotion_label)

    with st.chat_message("assistant"):
        response_box = st.empty()
        full_text = ""

        # Connect to Ollama with the specific system prompt
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages,
            stream=True,
        )

        # Stream the text to the screen
        for chunk in stream:
            full_text += chunk['message']['content']
            response_box.markdown(full_text + "▌") # Adds a cursor effect

        response_box.markdown(full_text)

    # Step D: Save and Refresh
    st.session_state.messages.append({"role": "assistant", "content": full_text})
