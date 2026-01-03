import os
from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

# Load environment variables from .env file. Ensures GOOGLE_API_KEY is there.
load_dotenv()

class SentimentAnalyzer:
    """Encapsulates the local Hugging Face emotion pipeline."""
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base", device=-1):
        print(f"Loading local sentiment model '{model_name}' (device={device})...")
        self._pipeline = pipeline("text-classification", model=model_name, device=device)

    def analyze(self, text: str) -> str:
        """Return the top label for the input text."""
        result = self._pipeline(text)[0]
        return result["label"]


class LLMChain:
    """Encapsulates LLM setup and a factory method that builds the prompt -> llm chain."""
    def __init__(self, model_name="gemini-2.5-flash"):
        print(f"Setting up Gemini LLM '{model_name}'...")
        self.llm = ChatGoogleGenerativeAI(model=model_name)

    def create_chain(self, input_dict):
        """Inspects sentiment via provided input_dict and returns a chain runnable."""
        user_input = input_dict.get("user_input", "")
        sentiment = input_dict.get("sentiment", "neutral")

        if sentiment == "anger":
            system_msg = "The user is ANGRY. Be extremely empathetic, de-escalate, and keep it brief."
        elif sentiment == "joy":
            system_msg = "The user is HAPPY. Be enthusiastic and match their high energy!"
        elif sentiment == "sadness":
            system_msg = "The user is SAD. Provide a supportive, gentle, and comforting response."
        else:
            system_msg = "The user is neutral. Provide a professional and helpful response."

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{user_input}")
        ])

        return prompt | self.llm | StrOutputParser()


class ChatApp:
    """Orchestrates sentiment analysis, LLM chain creation, and CLI interaction."""
    def __init__(self,
                 emo_model="j-hartmann/emotion-english-distilroberta-base",
                 emo_device=-1,
                 llm_model="gemini-2.5-flash"):
        self.sentiment_analyzer = SentimentAnalyzer(model_name=emo_model, device=emo_device)
        self.llm_chain = LLMChain(model_name=llm_model)

        # The Runnable expects a callable that accepts a single input dict
        def runnable_adapter(input_dict):
            user_input = input_dict.get("user_input", "")
            # Reuse provided sentiment if available to avoid duplicate analysis
            detected = input_dict.get("sentiment")
            if detected is None:
                detected = self.sentiment_analyzer.analyze(user_input)
            input_with_sentiment = {**input_dict, "sentiment": detected}
            return self.llm_chain.create_chain(input_with_sentiment)

        self.full_chain = RunnableLambda(runnable_adapter)

    def run_chat(self):
        print("\n--- Emotion-Aware Bot Started (Local Sentiment + Gemini) ---")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break

            # Detect sentiment once and pass it into the runnable to avoid duplicate work
            detected = self.sentiment_analyzer.analyze(user_input)
            response = self.full_chain.invoke({"user_input": user_input, "sentiment": detected})

            print(f"Bot [{detected}]: {response}\n")


if __name__ == "__main__":
    # Allow simple overrides via environment variables (keeps original defaults)
    EMO_MODEL = os.getenv("EMO_MODEL", "j-hartmann/emotion-english-distilroberta-base")
    EMO_DEVICE = int(os.getenv("EMO_DEVICE", "-1"))
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

    app = ChatApp(emo_model=EMO_MODEL, emo_device=EMO_DEVICE, llm_model=LLM_MODEL)
    app.run_chat()