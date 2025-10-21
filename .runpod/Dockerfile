FROM runpod/base:0.4.0-cuda12.1.1

# Install dependencies
RUN pip install --no-cache-dir torch diffusers transformers accelerate safetensors pillow runpod

# Copy code
WORKDIR /app
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]
