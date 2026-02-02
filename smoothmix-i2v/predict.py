from cog import BasePredictor, Input, Path
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import tempfile
import os

class Predictor(BasePredictor):
    def setup(self):
        # Download SmoothMix weights from Hugging Face
        print("Downloading SmoothMix weights from Hugging Face...")
        weights_path = hf_hub_download(
            repo_id="WZE12345/smoothmix-wan22-i2v",
            filename="model.safetensors",
        )
        print(f"Weights downloaded to: {weights_path}")

        # Load base model
        model_id = "Wan-AI/Wan2.2-I2V-14B-480P-Diffusers"
        self.image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
        self.vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=self.vae, image_encoder=self.image_encoder, torch_dtype=torch.bfloat16)
        
        # Load SmoothMix weights
        smoothmix_weights = load_file(weights_path)
        self.pipe.transformer.load_state_dict(smoothmix_weights, strict=False)
        self.pipe.to("cuda")

    def predict(self, image: Path = Input(description="Input image"), prompt: str = Input(description="Motion prompt"), num_frames: int = Input(default=81, ge=17, le=129), fps: int = Input(default=16, ge=8, le=30), guidance_scale: float = Input(default=5.0, ge=1.0, le=10.0), num_inference_steps: int = Input(default=30, ge=10, le=50)) -> Path:
        input_image = load_image(str(image)).resize((832, 480))
        output = self.pipe(image=input_image, prompt=prompt, negative_prompt="blurry, distorted, low quality", num_frames=num_frames, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
        out_path = Path(tempfile.mktemp(suffix=".mp4"))
        export_to_video(output.frames[0], str(out_path), fps=fps)
        return out_path
