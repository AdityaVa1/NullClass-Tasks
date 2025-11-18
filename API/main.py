import time
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, APIRouter
from pydantic import BaseModel

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 1. Model Architecture (Baseline) ---

class NMTModelWrapper:
    """
    Wrapper for the NMT model to handle loading and inference.
    By default, this runs in 'Simulated' mode to allow for immediate testing
    without heavy PyTorch/Transformers dependencies.
    """
    def __init__(self, use_real_model: bool = False):
        self.model = None
        self.tokenizer = None
        self.use_real_model = use_real_model
        self.model_name = "Helsinki-NLP/opus-mt-en-fr"

    def load(self):
        """Loads the model. Simulates a heavy 'Cold Start'."""
        logger.info("Starting model load sequence...")
        start_time = time.perf_counter()

        if self.use_real_model:
            # --- REAL MODEL LOADING (Uncomment to use actual HuggingFace model) ---
            # from transformers import MarianMTModel, MarianTokenizer
            # self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            # self.model = MarianMTModel.from_pretrained(self.model_name)
            pass
        else:
            # --- SIMULATED LOADING ---
            # Sleep to simulate loading heavy weights (e.g., 2.5 seconds)
            time.sleep(2.5) 
            self.model = "Simulated Model Loaded"
        
        end_time = time.perf_counter()
        load_duration = end_time - start_time
        logger.info(f"Model loaded successfully in {load_duration:.4f} seconds.")

    def predict(self, text: str, source_lang: str, target_lang: str) -> str:
        """Performs translation."""
        if self.use_real_model and self.model:
            # --- REAL INFERENCE ---
            # inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            # translated = self.model.generate(**inputs)
            # return self.tokenizer.decode(translated[0], skip_special_tokens=True)
            return "Real model prediction placeholder"
        else:
            # --- SIMULATED INFERENCE ---
            # Simulate processing time (e.g., 100ms - 300ms)
            time.sleep(0.15)
            return f"[Simulated Translation to {target_lang}]: {text}"

# Initialize Global Model Instance
nmt_model = NMTModelWrapper(use_real_model=False)

# --- 2. FastAPI Lifespan (Startup/Shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for the application lifespan.
    Loads the model before the API starts accepting requests.
    """
    # Startup Logic
    logger.info("API Startup initiated.")
    startup_start = time.perf_counter()
    
    try:
        nmt_model.load()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

    startup_end = time.perf_counter()
    startup_time = startup_end - startup_start
    logger.info(f"API Ready. Total Startup Time: {startup_time:.4f} seconds.")
    
    yield
    
    # Shutdown Logic
    logger.info("API Shutting down.")

app = FastAPI(title="Core NMT API", version="1.0.0", lifespan=lifespan)

# --- 3. Performance Middleware ---

@app.middleware("http")
async def measure_latency(request: Request, call_next):
    """
    Middleware to measure the processing time of every request.
    Adds an X-Process-Time header to the response.
    """
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    process_time = time.perf_counter() - start_time
    
    # Log metrics for monitoring
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Latency: {process_time:.4f}s")
    
    # Add header for client visibility
    response.headers["X-Process-Time"] = str(process_time)
    return response

# --- 4. API Schemas & Endpoints ---

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class TranslationResponse(BaseModel):
    translation: str
    metadata: dict = {}

@app.post("/api/v1/translate", response_model=TranslationResponse)
async def translate(payload: TranslationRequest):
    """
    Main translation endpoint.
    """
    # Run the blocking model prediction in a threadpool to avoid blocking the async event loop
    # This is crucial for throughput in Python async apps with CPU-bound tasks
    loop = asyncio.get_event_loop()
    translation_result = await loop.run_in_executor(
        None, 
        nmt_model.predict, 
        payload.text, 
        payload.source_lang, 
        payload.target_lang
    )

    return TranslationResponse(
        translation=translation_result,
        metadata={
            "model_version": "baseline-v1",
            "source": payload.source_lang,
            "target": payload.target_lang
        }
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": nmt_model.model is not None}
