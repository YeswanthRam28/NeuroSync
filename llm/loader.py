# llm/loader.py
import os
from llama_cpp import Llama

MODEL_PATH = r"E:\\LLM\\models\\gemma-2-9b\\gemma-2-9b-it-Q4_K_M.gguf"

# ======================================================
# SAFE LIMITS (avoid rope overflow / GGML_ASSERT)
# ======================================================
MAX_CONTEXT = 3072
MAX_OUTPUT = 180
MAX_CHUNK = 1400          # prompt trimming window
N_GPU_LAYERS = 60         # safe for 8GB GPU


# ======================================================
# LOAD MODEL (correct Gemma config)
# ======================================================
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=MAX_CONTEXT,
    n_batch=512,
    n_threads=6,
    n_gpu_layers=N_GPU_LAYERS,
    rope_scaling={"type": "yarn", "factor": 1.0},
    flash_attn=True,
    use_mmap=True,
    use_mlock=False,

    # IMPORTANT: use text mode only.
    # DO NOT use create_chat_completion for Gemma-it GGUF.
    chat_format="gemma",
)


# Warm-up call
try:
    llm("Hello", max_tokens=1)
except:
    pass


# ======================================================
# INTERNAL HELPERS
# ======================================================
SYSTEM_PROMPT = (
    "You are NeuroSync Assistant. "
    "You reply concisely, factually, and clearly. "
    "Avoid hallucinations or assumptions. "
    "Prefer short, helpful answers."
)

def _truncate(prompt: str):
    if len(prompt) <= MAX_CHUNK:
        return prompt
    return prompt[-MAX_CHUNK:]


def build_gemma_prompt(user_prompt: str) -> str:
    """
    Proper Gemma 2 chat format using special turn tokens.
    """
    return (
        f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


# ======================================================
# PUBLIC GENERATE (non-streaming)
# ======================================================
def generate(prompt: str, max_tokens: int = MAX_OUTPUT) -> str:
    safe = _truncate(prompt)
    full_prompt = build_gemma_prompt(safe)

    try:
        out = llm(
            full_prompt,
            max_tokens=max_tokens,
            stop=["<end_of_turn>"],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.05,
        )
    except Exception as e:
        print("[LLM ERROR]", e)
        return "⚠️ Model error."

    # Extract text safely
    try:
        text = out["choices"][0]["text"]
        return text.strip()
    except:
        return "⚠️ No content."


# ======================================================
# STREAMING GENERATOR
# ======================================================
def stream(prompt: str, max_tokens: int = MAX_OUTPUT):
    safe = _truncate(prompt)
    full_prompt = build_gemma_prompt(safe)

    try:
        for out in llm(
            full_prompt,
            max_tokens=max_tokens,
            stop=["<end_of_turn>"],
            stream=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.05,
        ):

            delta = out["choices"][0].get("text", "")
            if delta:
                yield delta

    except Exception as e:
        yield f"[Streaming Error: {e}]"
