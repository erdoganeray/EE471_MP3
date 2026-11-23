"""Dream Canvas: Flask server for the image generation chatbot (EE471 Mini Project 3).

Run:  python app.py   ->  http://127.0.0.1:5000

  GET  /                 the page (templates/index.html + static/)
  GET  /api/status       which models are loaded
  POST /api/image        {"prompt": "..."}                    -> {"image": "data:image/png;base64,..."}
  POST /api/chat         {"messages": [{"role", "content"}]}  -> {"reply": "..."}
  POST /api/transcribe   multipart form, field "audio"        -> {"text": "..."}
"""
import os
import tempfile
import time

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from models import AIModels

app = Flask(__name__)
ai = AIModels()


def error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify(ai.status())


@app.post("/api/image")
def generate_image():
    prompt = (request.get_json(silent=True) or {}).get("prompt", "").strip()
    if not prompt:
        return error("Write a prompt first.")
    start = time.time()
    image = ai.image.generate(prompt)
    return jsonify({"image": image, "seconds": round(time.time() - start, 1)})


@app.post("/api/chat")
def chat():
    messages = (request.get_json(silent=True) or {}).get("messages", [])
    messages = [{"role": m.get("role"), "content": str(m.get("content", "")).strip()}
                for m in messages if m.get("role") in ("user", "assistant")]
    if not messages or messages[-1]["role"] != "user" or not messages[-1]["content"]:
        return error("The last message must be a non-empty user message.")
    start = time.time()
    reply = ai.chat.reply(messages)
    return jsonify({"reply": reply, "seconds": round(time.time() - start, 1)})


@app.post("/api/transcribe")
def transcribe():
    audio = request.files.get("audio")
    if audio is None:
        return error("No audio received.")
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    # Windows cannot reopen an open NamedTemporaryFile, so close it before ffmpeg reads it
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        audio.save(tmp)
        tmp.close()
        start = time.time()
        text = ai.speech.transcribe(tmp.name)
    finally:
        os.unlink(tmp.name)
    return jsonify({"text": text, "seconds": round(time.time() - start, 1)})


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):  # 404, 405 ... keep their own status
        return error(e.description, e.code)
    app.logger.exception(e)
    return error(f"{type(e).__name__}: {e}", 500)


if __name__ == "__main__":
    ai.warm_up()
    # threaded: the page stays responsive while a model is busy; debug reloader off so models load once
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), threaded=True, debug=False)
