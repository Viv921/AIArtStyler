import torch
import torch.nn.functional as F  # Import functional
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import config
import model as classifier_model
import dataset
import argparse
import os


# --- Caching has been REMOVED ---
# Models will be loaded and unloaded on demand.


def get_style_classifier():
    """Loads the style classifier."""
    print(f"Loading style classifier from: {config.MODEL_PATH}")
    model = classifier_model.create_model(num_classes=config.NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None, None

    model.to(config.DEVICE)
    model.eval()

    class_names = config.CLASSES

    if config.NUM_CLASSES != len(class_names):
        print(f"Warning: Model has {config.NUM_CLASSES} classes but config has {len(class_names)}.")

    return model, class_names


def get_diffusion_pipeline():
    """Loads the diffusion pipeline."""
    print(f"Loading Stable Diffusion pipeline from: {config.STABLE_DIFFUSION_CHECKPOINT}")

    pipe = StableDiffusionXLPipeline.from_single_file(
        config.STABLE_DIFFUSION_CHECKPOINT,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe = pipe.to(config.DEVICE)

    pipe.enable_attention_slicing()
    print("Stable Diffusion pipeline loaded.")
    return pipe


def predict_style_topk(image_path, k=3):
    """
    Classifies a single image and returns the top k style labels and confidences.
    This function now loads AND unloads the classifier.
    """
    classifier, class_names = get_style_classifier()
    if classifier is None:
        raise Exception("Classifier not loaded.")

    results = []
    try:
        image = Image.open(image_path).convert("RGB")
        _, val_transform = dataset.get_transforms()
        image_tensor = val_transform(image).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            output = classifier(image_tensor)
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)

            # Get top k probabilities and indices
            topk_probs, topk_indices = torch.topk(probabilities, k)

        for i in range(k):
            prob = topk_probs[0, i].item()
            style_index = topk_indices[0, i].item()
            style_label = class_names[style_index]
            results.append({"style": style_label, "confidence": prob})

        print(f"Top predictions: {results}")

    finally:
        # --- UNLOAD THE CLASSIFIER ---
        if classifier:
            del classifier
            torch.cuda.empty_cache()
            print("Classifier unloaded from VRAM.")

    return results


def run_generation(
        prompt: str,
        style_image_path: str = None,
        style_name: str = None,
        # --- New Advanced Options ---
        negative_prompt: str = None,
        negative_prompt_2: str = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = -1  # Use -1 as a sentinel for random seed
):
    """
    Main logic function that loads/unloads models on demand.
    """
    style_label = None

    if style_image_path:
        # This will load, use, and unload the classifier.
        print(f"Predicting style from image: {style_image_path}")
        results = predict_style_topk(style_image_path, k=1)
        if not results:
            raise Exception(f"Failed to predict style for image: {style_image_path}")
        style_label = results[0]["style"]
    elif style_name:
        # Use the provided style name
        print(f"Using provided style name: {style_name}")
        # Load classifier just to get class_names
        classifier, class_names = get_style_classifier()
        if style_name not in class_names:
            print(f"Warning: Style '{style_name}' not in known class list. Using it anyway.")
        # Unload classifier immediately
        del classifier
        torch.cuda.empty_cache()
        print("Classifier unloaded (after name check).")
        style_label = style_name
    else:
        raise ValueError("Must provide either 'style_image_path' or 'style_name'.")

    # --- At this point, the classifier is NOT in VRAM ---

    # --- Load Diffusion Pipeline ---
    diffusion_pipe = get_diffusion_pipeline()

    output_path = None  # Define output_path before try
    try:
        # --- Prompting ---
        prompt1 = prompt
        prompt2 = f"{style_label} style"

        # Use negative prompts if provided
        final_negative_prompt = negative_prompt if negative_prompt else None
        final_negative_prompt_2 = negative_prompt_2 if negative_prompt_2 else final_negative_prompt

        # --- Seed Handling ---
        generator = None
        if seed != -1:
            generator = torch.Generator(device=config.DEVICE).manual_seed(seed)
            print(f"Using manual seed: {seed}")
        else:
            print("Using random seed.")

        print("Generating image... (this may take a moment)")
        with torch.autocast(config.DEVICE):
            image = diffusion_pipe(
                prompt=prompt1,
                prompt_2=prompt2,
                negative_prompt=final_negative_prompt,
                negative_prompt_2=final_negative_prompt_2,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        safe_prompt = "".join(x for x in prompt if x.isalnum())[:20]
        filename = f"{style_label}_{safe_prompt}_{os.urandom(4).hex()}.png"
        output_path = os.path.join(config.OUTPUT_DIR, filename)

        image.save(output_path)
        print(f"Image saved to: {output_path}")

    finally:
        # --- UNLOAD THE DIFFUSION PIPELINE ---
        if diffusion_pipe:
            del diffusion_pipe
            torch.cuda.empty_cache()
            print("Diffusion pipeline unloaded from VRAM.")

    return output_path


# Keep the main block for standalone script running
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Art Stylizer")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Text prompt describing the content (e.g., 'a dog playing chess')")
    parser.add_argument("-s", "--style_image", type=str, help="Path to the style reference image.")
    parser.add_argument("-n", "--style_name", type=str, help="Name of the style (e.g., 'Cubism').")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt.")
    parser.add_argument("--negative_prompt_2", type=str, default=None, help="Second negative prompt.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (CFG).")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for generation (-1 for random).")

    args = parser.parse_args()

    if not args.style_image and not args.style_name:
        print("Error: You must provide either --style_image or --style_name.")
    else:
        try:
            run_generation(
                prompt=args.prompt,
                style_image_path=args.style_image,
                style_name=args.style_name,
                # Pass advanced args
                negative_prompt=args.negative_prompt,
                negative_prompt_2=args.negative_prompt_2,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed
            )
        except Exception as e:
            print(f"An error occurred: {e}")