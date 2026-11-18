import google.generativeai as genai
from pathlib import Path
import os

from dotenv import load_dotenv

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="google.generativeai",
)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Set GEMINI_API_KEY in your .env file.")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-3.5-flash-lite')
context = Path('context.txt').read_text(encoding='utf-8')

def chat_with_bot(prompt):
    response = model.generate_content(f'Use the following context to answer:\n{context}\n\nQuestion: {prompt}')
    return response.text.strip()


print("Welcome to Sentinel's Chatbot! Type 'exit' to quit.\n")
while True:
    user = input("You: ")
    if user.strip().lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    bot = chat_with_bot(user)
    print("Chatbot:", bot)