import streamlit as st
from google import genai
from google.genai import types
from transformers import pipeline

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Hybrid Sentiment Bot", layout="wide")

MODEL_ID = "gemini-2.5-flash-lite"
EMO_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# --- 2. HELPER FUNCTIONS ---

@st.cache_resource
def get_analyzer():
    """Loads the emotion detection model locally."""
    return pipeline("text-classification", model=EMO_MODEL)

@st.cache_resource
def get_genai_client():
    """
    Creates the Gemini Client only once and caches it.
    This prevents the 'Client closed' error by keeping the connection alive.
    """
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("⚠️ Missing GOOGLE_API_KEY in .streamlit/secrets.toml")
        return None
    return genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

def get_system_instruction(emotion):
    behaviors = {
        "anger":   "The user is ANGRY. Be calm, empathetic, and try to de-escalate.",
        "joy":     "The user is HAPPY. Be energetic and share their excitement!",
        "sadness": "The user is SAD. Be supportive, gentle, and very comforting.",
        "default": "Be a helpful and professional AI assistant."
    }
    return behaviors.get(emotion, behaviors["default"])

def get_gemini_response_stream(history, instruction):
    """Sends chat to Google via the cached Client."""

    # 1. Get the cached client (prevents it from closing prematurely)
    client = get_genai_client()
    if not client:
        return None

    # 2. Convert Streamlit history to Gemini format
    gemini_contents = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    try:
        # 3. Generate Stream
        stream = client.models.generate_content_stream(
            model=MODEL_ID,
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                max_output_tokens=500,
                temperature=0.7
            )
        )
        return stream

    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

# --- 3. STATE & UI ---

st.title("🤖 Emotional Support Bot (Gemini Cloud)")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Analytics")
    metric_placeholder = st.empty()
    metric_placeholder.metric("Detected Emotion", "Waiting...")
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 4. MAIN CHAT LOOP ---

if user_input := st.chat_input("How are you feeling?"):
    # A. Show User Input
    st.chat_message("user").write(user_input)

    # B. Analyze Emotion
    analyzer = get_analyzer()
    result = analyzer(user_input)[0]
    emotion_label = result['label']
    metric_placeholder.metric("Detected Emotion", emotion_label.upper())

    # C. Prepare System Prompt
    instruction = get_system_instruction(emotion_label)

    # Add the NEW user message to the history temporarily for the API call
    current_chat_history = st.session_state.messages + [{"role": "user", "content": user_input}]

    # D. Generate Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        stream = get_gemini_response_stream(current_chat_history, instruction)

        if stream:
            for chunk in stream:
                # In the new SDK, accessing .text is safe
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.write(full_response + "▌")

            response_placeholder.write(full_response)

            # Save final state
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": full_response})