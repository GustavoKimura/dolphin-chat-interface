import os
import json
import time
import logging
from typing import cast, Dict, Any
from flask import Flask, request, render_template, Response, stream_with_context
from llama_cpp import Llama
from jinja2 import Template

# Constants
MAX_TOKENS = 4096
CONTEXT_SIZE = 131072
THREADS_TO_USE = os.cpu_count() or 4
BATCH_SIZE = 512

BALANCED = "balanced"
RATIONAL = "rational"
CREATIVE = "creative"

GENERATION_PROFILES = {
    BALANCED: {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
    },
    RATIONAL: {
        "temperature": 0.3,
        "top_k": 40,
        "top_p": 0.85,
        "repeat_penalty": 1.15,
    },
    CREATIVE: {
        "temperature": 0.7,
        "top_k": 60,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    },
}

profile = GENERATION_PROFILES[BALANCED]

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app_debug.log")],
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load template
try:
    with open("models/dolphin/custom.jinja", "r", encoding="utf-8") as f:
        jinja_template = Template(f.read())
    logger.info("Successfully loaded Jinja template")
except Exception as e:
    logger.error(f"Failed to load Jinja template: {str(e)}")
    raise

# Initialize Dolphin LLM
try:
    llm = Llama(
        model_path="models/dolphin/dolphin.gguf",
        n_ctx=CONTEXT_SIZE,
        n_threads=THREADS_TO_USE,
        n_threads_batch=THREADS_TO_USE,
        n_batch=BATCH_SIZE,
        n_ubatch=BATCH_SIZE,
        mlock=True,
        repeat_penalty=profile["repeat_penalty"],
        temperature=profile["temperature"],
        top_k=profile["top_k"],
        top_p=profile["top_p"],
        chat_format="chatml",
        stop=[],
        verbose=False,
    )
    logger.info("Successfully initialized LLM model")
except Exception as e:
    logger.error(f"Failed to initialize LLM model: {str(e)}")
    raise


def log_conversation_history(messages: list) -> str:
    """Log the full conversation history and return the formatted prompt."""
    logger.debug("Full conversation history:")
    for i, message in enumerate(messages):
        logger.debug(f"Message {i + 1} - Role: {message.get('role', 'unknown')}")
        logger.debug(f"Content: {message.get('content', '')}")

    formatted_prompt = jinja_template.render(messages=messages).strip()
    logger.debug(f"Formatted prompt sent to model:\n{formatted_prompt}")

    return formatted_prompt


@app.route("/")
def index():
    logger.debug("Serving index page")
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        logger.info(f"Received chat request with {len(messages)} messages")

        profile_name = data.get("profile", BALANCED)
        selected_profile = GENERATION_PROFILES.get(
            profile_name, GENERATION_PROFILES[BALANCED]
        )
        logger.info(f"Using generation profile: {profile_name}")

        formatted_prompt = log_conversation_history(messages)

        def generate():
            start_time = time.time()
            token_count = 0
            full_response = ""
            logger.debug("Starting response generation")

            for chunk in llm(
                prompt=formatted_prompt,
                echo=False,
                stream=True,
                max_tokens=MAX_TOKENS,
                temperature=selected_profile["temperature"],
                top_k=selected_profile["top_k"],
                top_p=selected_profile["top_p"],
                repeat_penalty=selected_profile["repeat_penalty"],
            ):
                chunk_dict = cast(Dict[str, Any], chunk)
                token = chunk_dict["choices"][0]["text"]
                full_response += token
                token_count += 1
                logger.debug(f"Generated token {token_count}: {token}")

                yield f"data: {json.dumps({'token': token})}\n\n"

            generation_time = time.time() - start_time
            tokens_per_second = (
                token_count / generation_time if generation_time > 0 else 0
            )

            logger.info(
                f"Generation completed - Tokens: {token_count}, "
                f"Time: {generation_time:.2f}s, "
                f"Tokens/s: {tokens_per_second:.2f}"
            )
            logger.debug(f"Full response: {full_response}")

            yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return Response(
            json.dumps({"error": "Internal server error"}),
            status=500,
            mimetype="application/json",
        )


if __name__ == "__main__":
    try:
        logger.info("Starting Flask application")
        app.run(host="0.0.0.0", port=8080, debug=False)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise
