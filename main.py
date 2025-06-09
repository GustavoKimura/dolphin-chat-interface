from flask import Flask, request, render_template, Response, stream_with_context
from llama_cpp import Llama
from jinja2 import Template
from typing import cast, Dict, Any
import os
import json

app = Flask(__name__, template_folder="templates", static_folder="static")

with open("models/templates/custom.jinja", "r", encoding="utf-8") as f:
    jinja_template = Template(f.read())

if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    llm = Llama(
        model_path="models/dolphin/dolphin.gguf",
        n_ctx=2048,
        n_threads=4,
        n_batch=64,
        n_threads_batch=32,
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

    def generate():
        for chunk in llm(
            prompt=formatted_prompt,
            echo=False,
            stream=True,
            max_tokens=512,
        ):
            chunk_dict = cast(Dict[str, Any], chunk)
            token = chunk_dict["choices"][0]["text"]
            yield f"data: {json.dumps({'token': token})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
