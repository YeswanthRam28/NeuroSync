MODEL_PATH = r"E:\\LLM\\models\\gemma-2-9b\\gemma-2-9b-it-Q4_K_M.gguf"

MEMORY_FILE = "memory.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Runtime settings
CTX_SIZE = 4096
GPU_LAYERS = 60        # RTX 3050 sweet spot
THREADS = 6
BATCH_SIZE = 512
TEMP = 0.6
TOP_K = 40
TOP_P = 0.9
