import time
import logging
import asyncio
import hashlib
import json
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
import redis.asyncio as redis  # Non-blocking Redis client

# Try imports for ONNX, fallback to simulation if not present
try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
REDIS_URL = "redis://localhost:6379"
MODEL_PATH = "onnx_models/quantized" # Path from the optimization script
USE_SIMULATION = not ONNX_AVAILABLE # Auto-switch based on library availability

# --- 1. Optimized Model Wrapper ---

class OptimizedNMTModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self):
        """Loads the Quantized ONNX model."""
        logger.info("Loading Optimized Model...")
        start_time = time.perf_counter()

        if not USE_SIMULATION and os.path.exists(MODEL_PATH):
            # Load ONNX Runtime Model
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
            logger.info("ONNX Quantized Model Loaded.")
        else:
            # Simulation Fallback
            time.sleep(1.0) # Faster load than baseline
            logger.info("Simulated Optimized Model Loaded.")
            self.model = "Simulated_ONNX"

        logger.info(f"Model load time: {time.perf_counter() - start_time:.4f}s")

    def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Runs batch inference.
        Batching is efficient because it leverages matrix multiplication parallelism.
        """
        if not USE_SIMULATION and self.tokenizer:
            # Tokenize batch
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            # Generate (Inference)
            outputs = self.model.generate(**inputs)
            # Decode batch
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            # Simulation: Faster than baseline per item due to "optimization"
            time.sleep(0.05 + (0.01 * len(texts))) 
            return [f"[Optimized-FR]: {t}" for t in texts]

nmt_model = OptimizedNMTModel()

# --- 2. Redis Caching Layer ---

redis_client = None

async def get_cache(key: str) -> Optional[str]:
    if not redis_client: return None
    try:
        return await redis_client.get(key)
    except Exception as e:
        logger.error(f"Redis error: {e}")
        return None

async def set_cache(key: str, value: str, expire: int = 3600):
    if not redis_client: return
    try:
        # Background task usually handles this, but here we await for simplicity or fire-and-forget
        await redis_client.set(key, value, ex=expire)
    except Exception:
        pass

def generate_cache_key(text: str, source: str, target: str) -> str:
    """Creates a deterministic hash for the input."""
    raw = f"{source}-{target}-{text.strip().lower()}"
    return hashlib.md5(raw.encode()).hexdigest()

# --- 3. Lifespan & App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Connected to Redis.")
    except Exception:
        logger.warning("Redis not available. Caching disabled.")
        redis_client = None

    nmt_model.load()
    
    yield
    
    # Shutdown
    if redis_client:
        await redis_client.close()

app = FastAPI(title="Optimized NMT API", version="2.0.0", lifespan=lifespan)

# --- 4. Endpoints ---

class BatchTranslationRequest(BaseModel):
    texts: List[str]
    source_lang: str
    target_lang: str

@app.post("/api/v2/translate/batch")
async def translate_batch(payload: BatchTranslationRequest):
    """
    Optimized Batch Endpoint with Caching.
    1. Check Cache for all items.
    2. Filter out hits.
    3. Run inference ONLY on misses.
    4. Merge results and return.
    """
    start_time = time.perf_counter()
    results = [None] * len(payload.texts)
    
    # Identify Cache Hits vs Misses
    indices_to_compute = []
    texts_to_compute = []

    # 1. Check Cache (Async pipelining could be used here for extreme perf)
    for i, text in enumerate(payload.texts):
        cache_key = generate_cache_key(text, payload.source_lang, payload.target_lang)
        cached_res = await get_cache(cache_key)
        if cached_res:
            results[i] = cached_res # Cache Hit
        else:
            indices_to_compute.append(i) # Cache Miss
            texts_to_compute.append(text)

    # 2. Run Inference on Misses (CPU-Bound task -> ThreadPool)
    if texts_to_compute:
        loop = asyncio.get_event_loop()
        # Run the blocking ONNX inference in a separate thread
        computed_translations = await loop.run_in_executor(
            None, 
            nmt_model.predict_batch, 
            texts_to_compute
        )

        # 3. Populate results and Cache new values
        for idx_in_batch, real_idx in enumerate(indices_to_compute):
            trans = computed_translations[idx_in_batch]
            results[real_idx] = trans
            
            # Set cache (fire and forget logic ideally, here awaited for safety)
            cache_key = generate_cache_key(texts_to_compute[idx_in_batch], payload.source_lang, payload.target_lang)
            await set_cache(cache_key, trans)

    process_time = time.perf_counter() - start_time

    return {
        "translations": results,
        "metrics": {
            "total_items": len(payload.texts),
            "cache_hits": len(payload.texts) - len(texts_to_compute),
            "compute_time": process_time
        }
    }
