from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from groq import Groq
from dotenv import load_dotenv
import json, os

# ====== Load environment variables ======
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ====== Initialize Groq client ======
client = Groq(api_key=GROQ_API_KEY)

# ====== FastAPI setup ======
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== Load emotion model from Hugging Face ======
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/bhavana2075/emotion_detection"
HF_HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"
}



# ====== Chat history ======
HISTORY_FILE = "chat_history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


def detect_emotion(text):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if any(word in text.lower() for word in greetings):
        return "neutral"

    payload = {"inputs": text}
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)

    if response.status_code != 200:
        return "neutral"

    result = response.json()

    try:
        return max(result[0], key=lambda x: x["score"])["label"]
    except:
        return "neutral"


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    user_text = data["message"]

    history = load_history()
    user_history = history.get(user_id, [])

    emotion = detect_emotion(user_text)

    # recent 5 messages as context
    context = "\n".join([f"{m['role']}: {m['text']}" for m in user_history[-5:]])

    # --- Refined short, natural prompt ---
    prompt = f"""
    You are MindEase — a kind, calm, and emotionally aware mental wellness chatbot.
    Your goal is to comfort the user briefly (max 2 short sentences), using the emotion detected.

    User’s emotion: {emotion}
    Chat so far:
    {context}

    User: "{user_text}"
    Respond warmly, simply, and encouragingly — like a caring friend.
    Avoid repeating similar phrases or analyzing emotions too deeply.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=80,  # limit response length
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        print("Groq API Error:", e)
        reply = "I'm sorry, I’m having trouble responding right now. Please try again later."

    user_history.append({"role": "user", "text": user_text, "emotion": emotion})
    user_history.append({"role": "bot", "text": reply})
    history[user_id] = user_history
    save_history(history)

    return {"reply": reply}
