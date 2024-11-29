from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import os
from src.flux.xflux_pipeline import XFluxPipeline
import uuid

app = FastAPI()

xflux_pipeline = XFluxPipeline("flux-dev", os.environ.get("FLUX_DEVICE", "cuda:6"), False)
xflux_pipeline.set_ip(None, "XLabs-AI/flux-ip-adapter-v2", "ip_adapter.safetensors")

class GenerateRequest(BaseModel):
    prompt: str
    neg_prompt: Optional[str] = ""
    img_prompt_path: Optional[str] = None
    num_steps: int = 25
    guidance: float = 4.0
    true_gs: float = 3.5
    timestep_to_start_cfg: int = 1
    width: int = 1024
    height: int = 1024

@app.post("/generate/")
async def generate(request: GenerateRequest):
    image_prompt = Image.open(request.img_prompt_path) if request.img_prompt_path else None

    result = xflux_pipeline(
        prompt=request.prompt,
        width=request.width,
        height=request.height,
        guidance=request.guidance,
        num_steps=request.num_steps,
        true_gs=request.true_gs,
        timestep_to_start_cfg=request.timestep_to_start_cfg,
        image_prompt=image_prompt,
    )

    img_path, ext = request.img_prompt_path.split(".")
    save_path = img_path + "_variantx_" + str(uuid.uuid4()) + '.' + ext
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    result.save(save_path)

    return {"output_path": save_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=os.environ.get("PORT", 8008))