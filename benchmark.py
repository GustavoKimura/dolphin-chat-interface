import os, time
from llama_cpp import Llama

MAX_TOKENS = 64
CONTEXT_SIZE = 32768
THREADS_TO_USE = os.cpu_count() or 4
REPEAT_PENALTY = 1.15
TEMPERATURE = 0.5
TOP_K = 50
TOP_P = 0.9

for batch in [
    236,
    240,
    244,
    248,
    252,
    256,
    260,
    264,
    268,
    272,
    276,
    280,
    296,
    300,
    304,
    308,
    312,
    324,
    328,
    332,
    336,
]:
    print(f"\n🔍 Testing batch size: {batch}")

    llm = Llama(
        model_path="models/dolphin/dolphin.gguf",
        n_ctx=CONTEXT_SIZE,
        n_threads=THREADS_TO_USE,
        n_threads_batch=THREADS_TO_USE,
        n_batch=batch,
        n_ubatch=batch,
        mlock=True,
        repeat_penalty=REPEAT_PENALTY,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        chat_format=None,
        stop=["</s>"],
        verbose=False,
    )

    start = time.time()
    output = llm("Olá!", max_tokens=MAX_TOKENS)
    duration = time.time() - start
    print(f"🕒 Took {duration:.2f}s for batch size {batch}")
