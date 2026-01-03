import os
import joblib
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    """Uses a locally saved joblib binary classifier (0/1)."""
    def __init__(self, model_path: str = "classifier.joblib", vectorizer_path: str = "vectorizer.joblib"):
        print(f"Loading local model and vectorizer...")
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def analyze(self, text: str) -> tuple[str, float]:
        """
        Predicts sentiment and returns a label and the confidence score.
        1 = Positive, 0 = Negative
        """
        # Vectorize the user input
        vectorized = self.vectorizer.transform([text])

        # Predict class and probabilities
        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]

        # Map the numeric prediction to your specific emoji labels
        if prediction == 1:
            sentiment_label = "Positive"
            confidence = probabilities[1]
        else:
            sentiment_label = "Negative"
            confidence = probabilities[0]

        return sentiment_label, confidence

class GeminiLLM:
    """Encapsulates Gemini API setup."""
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, system_msg: str, user_input: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=system_msg,
                temperature=0.7
            )
        )
        return response.text

class ChatApp:
    def __init__(self, model_file, vectorizer_file, llm_model):
        self.analyzer = SentimentAnalyzer(model_file, vectorizer_file)
        self.llm = GeminiLLM(model_name=llm_model)

    def get_system_message(self, sentiment: str) -> str:
        # We map your model's labels to the desired AI behavior
        messages = {
            "Positive": "The user is HAPPY. Be enthusiastic and match their high energy!",
            "Negative": "The user is SAD or NEGATIVE. Be supportive, gentle, and comforting."
        }
        return messages.get(sentiment, "Provide a professional response.")

    def run_chat(self):
        print("\n--- Sentiment Chat Active (Binary Model + Gemini) ---")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']: break

            # Get label and confidence
            label, confidence = self.analyzer.analyze(user_input)

            # Prepare instructions for Gemini
            system_msg = self.get_system_message(label)
            ai_response = self.llm.generate_response(system_msg, user_input)

            print(f"Bot [{label} ({confidence:.2%})]: {ai_response}\n")

if __name__ == "__main__":
    SENTIMENT_ANALYZER_MODEL_PATH = "./models/classifier.joblib"
    SENTIMENT_ANALYZER_VECTORIZER_PATH = "./models/vectorizer.joblib"
    GEMINI_LLM_MODEL_NAME = "gemini-2.5-flash-lite"

    app = ChatApp(
        SENTIMENT_ANALYZER_MODEL_PATH,
        SENTIMENT_ANALYZER_VECTORIZER_PATH,
        GEMINI_LLM_MODEL_NAME
    )
    app.run_chat()