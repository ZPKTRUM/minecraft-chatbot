import os
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app = Flask(__name__)
CORS(app)

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.3-70b-instruct")

SYSTEM_PROMPT = """
Eres un experto guía de Minecraft para principiantes. Tu misión es ayudar al jugador desde el primer día (cortar un árbol) hasta derrotar a la Enderdragón. Proporcionas información clara sobre:
- Mobs (zombies, creepers, enderman, etc.) y cómo enfrentarlos.
- Biomas (características, recursos, peligros).
- Objetos y crafteos (recetas paso a paso, utilidad).
- Progresión: herramientas, armaduras, nether, fortalezas, ojo de ender, combate final.

Además, analizas SITUACIONES que el jugador te describa (ej: "Estoy perdido en una cueva con poca comida", "Un creeper explotó mi casa", "No encuentro diamantes") y das consejos prácticos y accionables.

Siempre responde en el mismo idioma que el usuario (español o inglés). Usa un tono amigable, motivador y apto para novatos. Incluye recetas de crafteo cuando sea relevante, las respuestas deben ser breves, sin realizar tanto relleno.
"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "messages" not in data:
            return jsonify({"error": "Mensaje no válido"}), 400

        history = data["messages"]
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=full_messages,
            temperature=0.6,
            max_tokens=500
        )
        reply = completion.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        print("ERROR interno:")
        traceback.print_exc()
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)