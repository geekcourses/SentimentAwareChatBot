import os
from google import genai
from google.genai import types
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file. Ensures GOOGLE_API_KEY is there.
load_dotenv()

class SentimentAnalyzer:
    """Encapsulates the local Hugging Face emotion pipeline."""
    def __init__(self, model_name:str="j-hartmann/emotion-english-distilroberta-base", device:int=-1):
        print(f"Loading local sentiment model '{model_name}' (device={device})...")
        self._pipeline = pipeline("text-classification", model=model_name, device=device)

    def analyze(self, text: str) -> str:
        """Return the top label for the input text."""
        result = self._pipeline(text)[0]
        return str(result["label"])

class GeminiLLM:
    """Encapsulates Gemini API setup and response generation."""
    def __init__(self, model_name:str="gemini-2.5-flash", api_key:str|None=None):
        print(f"Setting up Gemini API '{model_name}'...")
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def list_available_models(self) -> list:
        """List all available models from the Gemini API."""
        try:
            models = self.client.models.list()
            return [m.name for m in models]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def generate_response(self, system_msg: str, user_input: str) -> str|None:
        """Generate a response using the Gemini API with system and user messages."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=system_msg,
                    temperature=0.7
                )
            )
            return response.text
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                print(f"\n❌ Model '{self.model_name}' not found.")
                print("Available models:")
                models = self.list_available_models()
                if models:
                    for model in models:
                        print(f"  - {model}")
                else:
                    print("  Could not retrieve model list.")
                raise
            else:
                raise


class ChatApp:
    """Orchestrates sentiment analysis, LLM response generation, and CLI interaction."""
    def __init__(self,
                 emo_model="j-hartmann/emotion-english-distilroberta-base",
                 emo_device=-1,
                 llm_model="gemini-2.5-flash",
                 api_key=None):
        self.sentiment_analyzer = SentimentAnalyzer(model_name=emo_model, device=emo_device)
        self.llm = GeminiLLM(model_name=llm_model, api_key=api_key)

    def get_system_message(self, sentiment: str) -> str:
        """Return context-aware system message based on detected sentiment."""
        if sentiment == "anger":
            return "The user is ANGRY. Be extremely empathetic, de-escalate, and keep it brief."
        elif sentiment == "joy":
            return "The user is HAPPY. Be enthusiastic and match their high energy!"
        elif sentiment == "sadness":
            return "The user is SAD. Provide a supportive, gentle, and comforting response."
        else:
            return "The user is neutral. Provide a professional and helpful response."

    def run_chat(self):
        print("\n--- Emotion-Aware Bot Started (Local Sentiment + Gemini API) ---")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit']:
                    break

                # Detect sentiment
                detected = self.sentiment_analyzer.analyze(user_input)
                system_msg = self.get_system_message(detected)

                # Generate response with Gemini API
                response = self.llm.generate_response(system_msg, user_input)

                print(f"Bot [{detected}]: {response}\n")
            except Exception as e:
                print(f"Error: {e}\n")
                break


if __name__ == "__main__":
    # Allow simple overrides via environment variables (keeps original defaults)
    EMO_MODEL = os.getenv("EMO_MODEL", "j-hartmann/emotion-english-distilroberta-base")
    EMO_DEVICE = int(os.getenv("EMO_DEVICE", "-1"))
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3-flash-preview")

    app = ChatApp(emo_model=EMO_MODEL, emo_device=EMO_DEVICE, llm_model=LLM_MODEL)
    app.run_chat()