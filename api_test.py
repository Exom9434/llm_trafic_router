import os
from dotenv import load_dotenv
import openai
from anthropic import Anthropic
import google.generativeai as genai

load_dotenv()

oa_client  = openai.OpenAI()
ant_client = Anthropic()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

TEST_PROMPT = "What is 1+1? Answer with the number only."
MODELS = {
    "openai":    "gpt-5-nano-2025-08-07",
    "anthropic": "claude-haiku-4-5-20251001",
    "google":    "gemini-3.1-flash-lite-preview",
}

def test_openai():
    response = oa_client.chat.completions.create(
        model=MODELS["openai"],
        messages=[{"role": "user", "content": TEST_PROMPT}],
        temperature=0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()

def test_anthropic():
    response = ant_client.messages.create(
        model=MODELS["anthropic"],
        max_tokens=10,
        temperature=0,
        messages=[{"role": "user", "content": TEST_PROMPT}]
    )
    return response.content[0].text.strip()

def test_google():
    model    = genai.GenerativeModel(MODELS["google"])
    response = model.generate_content(
        TEST_PROMPT,
        generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=10)
    )
    return response.text.strip()

TESTS = {
    "openai":    test_openai,
    "anthropic": test_anthropic,
    "google":    test_google,
}

if __name__ == "__main__":
    for provider, fn in TESTS.items():
        model = MODELS[provider]
        try:
            answer = fn()
            print(f"[OK] {provider} ({model}) → '{answer}'")
        except Exception as e:
            print(f"[FAIL] {provider} ({model}) → {e}")