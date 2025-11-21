from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from jinja2 import Environment, DictLoader
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from PIL import Image
import imageio.v2 as imageio
import pyttsx3
import matplotlib.pyplot as plt
import torch
import os

app = FastAPI()

# Cartelle statiche
for folder in ["immagini", "video", "musica", "voce", "grafici"]:
    os.makedirs(f"static/{folder}", exist_ok=True)

# Modelli
image_gen = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float32
).to("cpu")

music_gen = pipeline("text-to-audio", model="facebook/musicgen-small")
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

# HTML template
templates = {
    "index.html": """
    <!DOCTYPE html>
    <html lang="it">
    <head>
      <meta charset="UTF-8">
      <title>Giorgio's Intelligence</title>
      <style>
        body { font-family: 'Segoe UI'; background: #1f1c2c; color: white; text-align: center; padding: 40px; }
        h1 { font-size: 3em; background: linear-gradient(to right, #00aaff, #ff0055); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 30px; }
        input[type="text"] { padding: 15px; width: 60%; border-radius: 10px; border: none; font-size: 1em; margin-bottom: 20px; }
        .icons { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-bottom: 30px; }
        button { background-color: white; color: #1f1c2c; border: none; border-radius: 10px; padding: 15px 20px; font-size: 1em; font-weight: bold; cursor: pointer; display: flex; align-items: center; gap: 10px; }
        img, video, audio, p { margin-top: 30px; max-width: 80%; border-radius: 10px; }
        a { color: #00ffff; text-decoration: underline; }
      </style>
    </head>
    <body>
      <h1>GIORGIO’S INTELLIGENCE</h1>
      <form method="post">
        <input type="text" name="prompt" placeholder="Scrivi il tuo prompt" required />
        <div class="icons">
          <button name="action" value="immagine">🖼️ immagine</button>
          <button name="action" value="video">🎥 video</button>
          <button name="action" value="musica">🎵 musica</button>
          <button name="action" value="voce">🗣️ voce</button>
          <button name="action" value="grafico">📊 grafico</button>
          <button name="action" value="chat">💬 chatbot</button>
        </div>
      </form>
      {% if file_path %}
        <div>
          {{ preview|safe }}
          <p><a href="{{ file_path }}" download>📥 Scarica file</a></p>
        </div>
      {% endif %}
    </body>
    </html>
    """
}

env = Environment(loader=DictLoader(templates))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    html = env.get_template("index.html").render(file_path=None, preview=None)
    return HTMLResponse(content=html)

@app.post("/", response_class=HTMLResponse)
async def generate(request: Request, prompt: str = Form(...), action: str = Form(...)):
    safe_name = prompt.replace(" ", "_").replace("/", "_")
    file_path, preview = None, ""

    if action == "immagine":
        image = image_gen(prompt).images[0]
        file_path = f"static/immagini/{safe_name}.png"
        image.save(file_path)
        preview = f'<img src="/{file_path}" />'

    elif action == "video":
        frames = []
        for i in range(5):
            img = image_gen(f"{prompt}, scena {i+1}").images[0]
            img_path = f"static/video/{safe_name}_frame{i}.png"
            img.save(img_path)
            frames.append(img_path)

        file_path = f"static/video/{safe_name}.mp4"
        with imageio.get_writer(file_path, fps=1) as writer:
            for frame_path in frames:
                frame = imageio.imread(frame_path)
                writer.append_data(frame)

        preview = f'<video controls src="/{file_path}"></video>'

    elif action == "musica":
        audio = music_gen(prompt)[0]["audio"]
        file_path = f"static/musica/{safe_name}.wav"
        with open(file_path, "wb") as f:
            f.write(audio)
        preview = f'<audio controls src="/{file_path}"></audio>'

    elif action == "voce":
        file_path = f"static/voce/{safe_name}.mp3"
        engine = pyttsx3.init()
        engine.save_to_file(prompt, file_path)
        engine.runAndWait()
        preview = f'<audio controls src="/{file_path}"></audio>'

    elif action == "grafico":
        file_path = f"static/grafici/{safe_name}.png"
        plt.figure()
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title(prompt)
        plt.savefig(file_path)
        preview = f'<img src="/{file_path}" />'

    elif action == "chat":
        risposta = chatbot(prompt, max_new_tokens=100)[0]["generated_text"]
        preview = f"<p><strong>Risposta:</strong> {risposta}</p>"

    html = env.get_template("index.html").render(file_path=file_path, preview=preview)
    return HTMLResponse(content=html)

@app.get("/static/{folder}/{filename}", response_class=FileResponse)
async def serve_file(folder: str, filename: str):
    return FileResponse(path=f"static/{folder}/{filename}")
