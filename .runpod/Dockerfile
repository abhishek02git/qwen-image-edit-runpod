# Use a modern Runpod PyTorch base image
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Install dependencies
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors pillow runpod

# (Optional) Pre-download the model to reduce cold start latency
RUN python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509')"

# Copy handler file
WORKDIR /app
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]
