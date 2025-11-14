# api.py

import uvicorn
import shutil
import os
import uuid
import config
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# --- THIS IS THE NEW IMPORT ---
from fastapi.concurrency import run_in_threadpool

# --- Import the generator functions ---
from generator import (
    run_generation,
    predict_style_topk,
    get_style_classifier,
    get_diffusion_pipeline
)



# --- Pass the lifespan to the FastAPI app ---
app = FastAPI(title="AI Art Stylizer API")

# --- Configure CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Temporary Storage ---
TEMP_DIR = "./temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Static File Serving for Gallery ---
output_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config.OUTPUT_DIR))
os.makedirs(output_dir_path, exist_ok=True)
print(f"Serving gallery images from: {output_dir_path}")

app.mount("/outputs", StaticFiles(directory=output_dir_path), name="outputs")


# --- Helper Function ---
def save_temp_file(file: UploadFile) -> str:
    temp_image_name = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(TEMP_DIR, temp_image_name)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        file.file.close()
    return temp_path


# --- Response Models ---
class StylePrediction(BaseModel):
    style: str
    confidence: float


class GalleryImage(BaseModel):
    filename: str
    url: str


# --- API Endpoints (No changes below this line) ---

@app.get("/")
def read_root():
    return {"message": "AI Art Stylizer API is running."}


@app.post("/classify/", response_model=List[StylePrediction])
async def classify_image(
        image: UploadFile = File(...)
):
    temp_path = save_temp_file(image)
    try:
        # We also wrap the inference in a threadpool
        results = await run_in_threadpool(predict_style_topk, temp_path, k=3)
        if results is None:
            raise HTTPException(status_code=500, detail="Failed to classify image.")
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/generate/")
async def generate_image(
        # --- Standard fields ---
        prompt: str = Form(...),
        style_image: Optional[UploadFile] = File(None),
        style_name: Optional[str] = Form(None),

        # --- Advanced fields ---
        negative_prompt: Optional[str] = Form(None),
        negative_prompt_2: Optional[str] = Form(None),
        width: int = Form(1024),
        height: int = Form(1024),
        num_inference_steps: int = Form(50),
        guidance_scale: float = Form(7.5),
        seed: int = Form(-1)
):
    """
    Generates an image.
    Requires 'prompt' and EITHER 'style_image' OR 'style_name'.
    Now accepts advanced SDXL parameters.
    """
    if not style_image and not style_name:
        raise HTTPException(status_code=400, detail="Must provide either 'style_image' or 'style_name'.")

    temp_style_path = None
    try:
        # We must wrap the entire generation process in the threadpool

        # Prepare arguments for run_generation
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "negative_prompt_2": negative_prompt_2,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed
        }

        if style_image:
            temp_style_path = save_temp_file(style_image)
            gen_kwargs["style_image_path"] = temp_style_path
        else:
            gen_kwargs["style_name"] = style_name

        # Run in threadpool
        output_image_path = await run_in_threadpool(run_generation, **gen_kwargs)

        if not os.path.exists(output_image_path):
            raise HTTPException(status_code=500, detail="Generation failed: Output file not found.")

        return FileResponse(output_image_path, media_type="image/png")

    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during generation: {str(e)}")
    finally:
        if temp_style_path and os.path.exists(temp_style_path):
            os.remove(temp_style_path)


@app.get("/gallery/", response_model=List[GalleryImage])
async def get_gallery():
    images = []
    try:
        files = os.listdir(output_dir_path)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        image_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(output_dir_path, f)),
            reverse=True
        )

        for filename in image_files:
            images.append(GalleryImage(
                filename=filename,
                url=f"/outputs/{filename}"
            ))

        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read gallery: {e}")


# --- Run the API ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8080, reload=True)