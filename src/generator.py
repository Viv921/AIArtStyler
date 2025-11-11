# generator.py

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import config
import model as classifier_model
import dataset
import argparse
import os


def load_style_classifier(model_path, num_classes):
    """
    Loads the trained art style classifier.
    """
    print(f"Loading style classifier from: {model_path}")
    model = classifier_model.create_model(num_classes=num_classes)

    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Make sure config.py NUM_CLASSES matches the saved model.")
        return None, None

    model.to(config.DEVICE)
    model.eval()

    print("Loading class names from config...")
    class_names = config.CLASSES

    if num_classes != len(class_names):
        print(f"Warning: Model has {num_classes} classes but config has {len(class_names)}.")
        print("Using class names from config. This may cause a mismatch.")

    return model, class_names


def load_diffusion_pipeline():
    """
    Loads the Stable Diffusion pipeline from the path in config.
    """
    print(f"Loading Stable Diffusion pipeline from: {config.STABLE_DIFFUSION_CHECKPOINT}")

    vae = AutoencoderKL.from_single_file(
        config.STABLE_DIFFUSION_VAE,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    pipe = StableDiffusionXLPipeline.from_single_file(
        config.STABLE_DIFFUSION_CHECKPOINT,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.vae = vae

    pipe = pipe.to(config.DEVICE)

    print("Stable Diffusion pipeline loaded (UNet in fp16, VAE in fp32).")
    return pipe


def predict_style(image_path, classifier, class_names):
    """
    Classifies a single image and returns the style label as a string.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Style reference image not found at {image_path}")
        return None

    _, val_transform = dataset.get_transforms()

    image_tensor = val_transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = classifier(image_tensor)
        _, predicted_idx = torch.max(output, 1)

    style_label = class_names[predicted_idx.item()]
    print(f"Style identified: {style_label}")
    return style_label


def main(args):
    classifier, class_names = load_style_classifier(
        config.MODEL_PATH,
        config.NUM_CLASSES
    )
    if classifier is None:
        return

    diffusion_pipe = load_diffusion_pipeline()

    style_label = predict_style(args.style_image, classifier, class_names)
    if style_label is None:
        return

    final_prompt = f"{style_label} style, {args.prompt}"
    prompt1 = args.prompt
    prompt2 = f"{style_label} style"
    print(f"Final prompt: \"{final_prompt}\"")

    print("Generating image... (this may take a moment)")
    with torch.autocast(config.DEVICE):
        image = diffusion_pipe(
            prompt = prompt1,
            prompt_2 = prompt2,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    safe_prompt = "".join(x for x in args.prompt if x.isalnum())[:20]
    filename = f"{style_label}_{safe_prompt}.png"
    output_path = os.path.join(config.OUTPUT_DIR, filename)

    image.save(output_path)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Art Stylizer")
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the content (e.g., 'a dog playing chess')"
    )
    parser.add_argument(
        "-s", "--style_image",
        type=str,
        required=True,
        help="Path to the style reference image (e.g., 'my_impressionist_art.jpg')"
    )
    args = parser.parse_args()

    main(args)