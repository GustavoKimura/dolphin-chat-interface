from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama
from jinja2 import Template
from typing import cast, Dict, Any

app = Flask(__name__, template_folder="templates", static_folder="static")

with open("models/templates/custom.jinja", "r", encoding="utf-8") as f:
    jinja_template = Template(f.read())

llm = Llama(
    model_path="models/dolphin/dolphin.gguf",
    n_ctx=512,
    n_threads=4,
    n_batch=64,
    n_threads_batch=4,
    mlock=True,
    repeat_penalty=1.1,
    temperature=0.7,
    top_k=20,
    top_p=0.9,
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("prompt", "")

    messages = [{"role": "user", "content": user_input}]

    formatted_prompt = jinja_template.render(messages=messages)

    response_text = ""
    for chunk in llm(prompt=formatted_prompt, stop=["</s>"], echo=False, stream=True):
        chunk_dict = cast(Dict[str, Any], chunk)
        response_text += chunk_dict["choices"][0]["text"]

    text = response_text.strip()

    return jsonify({"response": text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
