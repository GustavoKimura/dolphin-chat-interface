import os, json, time
from typing import cast, Dict, Any
from flask import Flask, request, render_template, Response, stream_with_context
from llama_cpp import Llama
from jinja2 import Template

app = Flask(__name__, template_folder="templates", static_folder="static")

with open("models/templates/custom.jinja", "r", encoding="utf-8") as f:
    jinja_template = Template(f.read())

llm = Llama(
    model_path="models/dolphin/dolphin.gguf",
    n_ctx=4096,
    n_threads=os.cpu_count(),
    n_threads_batch=os.cpu_count(),
    n_batch=512,
    n_ubatch=128,
    mlock=True,
    repeat_penalty=1.15,
    temperature=0.5,
    top_k=50,
    top_p=0.85,
    chat_format="chatml",
    stop=[],
    verbose=False,
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("prompt", "")
    messages = [{"role": "user", "content": user_input}]
    formatted_prompt = jinja_template.render(messages=messages).strip()

    def generate():
        start_time = time.time()
        token_count = 0

        for chunk in llm(
            prompt=formatted_prompt,
            echo=False,
            stream=True,
            max_tokens=256,
        ):
            chunk_dict = cast(Dict[str, Any], chunk)
            token = chunk_dict["choices"][0]["text"]
            token_count += 1
            yield f"data: {json.dumps({'token': token})}\n\n"

        print(
            f"[LOG] Generation complete: {token_count} tokens in {time.time() - start_time:.2f}s"
        )
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":

    def warmup_model():
        print("[LOG] Doing warmup...")
        try:
            messages = [{"role": "user", "content": "Doing warmup..."}]
            formatted_prompt = jinja_template.render(messages=messages)

            token_count = 0
            for chunk in llm(
                prompt=formatted_prompt,
                echo=False,
                stream=True,
                max_tokens=8,
            ):
                token_count += 1
                if token_count >= 1:
                    break

            print("[LOG] Warmup complete.")

        except Exception as e:
            print(f"[LOG] Warmup error: {e}")

    warmup_model()

    app.run(host="0.0.0.0", port=8080, debug=True)
