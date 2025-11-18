import os
import time
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

# Define paths
MODEL_ID = "Helsinki-NLP/opus-mt-en-fr"
EXPORT_PATH = Path("onnx_models/base")
QUANTIZED_PATH = Path("onnx_models/quantized")

def export_and_quantize():
    print(f"--- Starting Optimization for {MODEL_ID} ---")
    
    # 1. Export to ONNX (Float32)
    print("1. Exporting model to ONNX Runtime (this may take a minute)...")
    start = time.perf_counter()
    
    # ORTModelForSeq2SeqLM handles the complex tracing of Encoder-Decoder models automatically
    model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_ID, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    model.save_pretrained(EXPORT_PATH)
    tokenizer.save_pretrained(EXPORT_PATH)
    
    print(f"Export complete in {time.perf_counter() - start:.2f}s")

    # 2. Quantization (Float32 -> Int8)
    print("2. Quantizing model to Int8...")
    start = time.perf_counter()

    # Create a quantizer from the exported model
    quantizer = ORTQuantizer.from_pretrained(EXPORT_PATH)
    
    # Define quantization configuration (avx512 is best for modern CPUs, use avx2 otherwise)
    qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=True)
    
    # Apply quantization
    quantizer.quantize(
        save_dir=QUANTIZED_PATH,
        quantization_config=qconfig,
    )
    tokenizer.save_pretrained(QUANTIZED_PATH)
    
    print(f"Quantization complete in {time.perf_counter() - start:.2f}s")
    
    # 3. Size Comparison
    base_size = os.path.getsize(EXPORT_PATH / "encoder_model.onnx") + os.path.getsize(EXPORT_PATH / "decoder_model.onnx")
    quant_size = os.path.getsize(QUANTIZED_PATH / "encoder_model_quantized.onnx") + os.path.getsize(QUANTIZED_PATH / "decoder_model_quantized.onnx")
    
    print(f"\n--- Results ---")
    print(f"Original ONNX Size: {base_size / (1024*1024):.2f} MB")
    print(f"Quantized ONNX Size: {quant_size / (1024*1024):.2f} MB")
    print(f"Size Reduction: {(1 - quant_size/base_size)*100:.2f}%")

if __name__ == "__main__":
    # Requires: pip install optimum[onnxruntime]
    export_and_quantize()
