import runpod
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import torch
from io import BytesIO
import base64
from PIL import Image

# Load the model once when the container starts
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.float16).to("cuda")

def handler(job):
    try:
        input_data = job["input"]
        prompt = input_data.get("prompt", "Enhance the image")
        image_url = input_data.get("image_url")

        if not image_url:
            return {"error": "Missing 'image_url' parameter."}

        input_image = load_image(image_url)
        output_image = pipe(image=input_image, prompt=prompt).images[0]

        # Convert output image to base64
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"output_image_base64": img_str, "prompt": prompt}

    except Exception as e:
        return {"error": str(e)}

# Start the Runpod serverless handler
runpod.serverless.start({"handler": handler})
