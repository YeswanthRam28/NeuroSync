# llm/loader.py
import os
from llama_cpp import Llama

MODEL_PATH = r"E:\\LLM\\models\\gemma-2-9b\\gemma-2-9b-it-Q4_K_M.gguf"

# ======================================================
# PROTECTOR SETTINGS (fix GGML_ASSERT, rope overflow etc.)
# ======================================================
MAX_CONTEXT = 3072        # keep well below 4096 to avoid overflow
MAX_CHUNK = 1500          # max prompt tokens before summarising/truncating
MAX_TOKENS_OUT = 180
N_GPU_LAYERS = 60         # safe for RTX 3050 8GB

# ======================================================
# INIT MODEL
# ======================================================
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=MAX_CONTEXT,
    n_batch=512,           # large batch = faster, safe for 8GB
    n_threads=6,
    n_gpu_layers=N_GPU_LAYERS,
    rope_scaling={"type": "yarn", "factor": 1.0},   # prevents rope crash on long prompts
    flash_attn=True,
    use_mmap=True,
    use_mlock=False,
    chat_format="gemma",
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repeat_penalty=1.05
)

# Warmup (safe)
try:
    llm("Hi", max_tokens=1)
except:
    pass


# ======================================================
# INTERNAL HELPERS
# ======================================================
def _truncate_prompt(prompt: str) -> str:
    """Avoids long-prompt crashes (GGML_ASSERT)."""
    if len(prompt) <= MAX_CHUNK:
        return prompt
    return prompt[-MAX_CHUNK:]   # keep last chunk only


SYSTEM_PROMPT = (
    "You are NeuroSync Assistant. "
    "You reply concisely, clearly, and helpfully. "
    "Avoid hallucinations. "
    "Keep responses under 5 sentences unless asked otherwise."
)


def _format_chat(prompt: str):
    """Correct Gemma chat formatting."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }


# ======================================================
# PUBLIC GENERATE FUNCTIONS
# ======================================================
def generate(prompt: str, max_tokens=MAX_TOKENS_OUT) -> str:
    """
    Non-streaming safe generator.
    Uses Gemma chat format.
    Protects Llama from context overflows.
    """
    safe_prompt = _truncate_prompt(prompt)
    chat = _format_chat(safe_prompt)

    try:
        out = llm.create_chat_completion(
            messages=chat["messages"],
            max_tokens=max_tokens,
        )
    except Exception as e:
        print("[LLM ERROR]", e)
        return "⚠️ Model error occurred."

    try:
        return out["choices"][0]["message"]["content"].strip()
    except:
        return "⚠️ No content."


def stream(prompt: str, max_tokens=MAX_TOKENS_OUT):
    """
    Streaming generator (yields chunks).
    Fixed version that avoids prefix-match crash.
    """
    safe_prompt = _truncate_prompt(prompt)
    chat = _format_chat(safe_prompt)

    try:
        for out in llm.create_chat_completion(
            messages=chat["messages"],
            max_tokens=max_tokens,
            stream=True
        ):
            if "choices" not in out:
                continue

            delta = out["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]

    except Exception as e:
        yield f"\n[Streaming Error: {e}]\n"
