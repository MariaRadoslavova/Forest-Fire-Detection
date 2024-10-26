import os
import torch
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection, DetrImageProcessor
from torchvision.transforms.functional import to_tensor

def run_inference_on_images(input_folder, output_folder, model, processor, device):
    """
    Runs inference on all images in the input folder and overlays bounding boxes on them.

    Args:
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str): Path to the folder where output images will be saved.
    - model: Pre-trained DETR model.
    - processor: DETR image processor.
    - device: Device to run inference on ('cpu' or 'cuda').
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")

            # Open image
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess image
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            # Run inference
            model.eval()
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
            
            # Post-process results
            results = processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])[0]
            
            # Draw bounding boxes
            draw = ImageDraw.Draw(image)
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                if score > 0.5:  # Set a threshold to filter low-confidence detections
                    draw_box(draw, box, score, label)

            # Save the image with overlays
            output_path = os.path.join(output_folder, filename)
            image.save(output_path)
            print(f"Saved processed image to {output_path}")

def draw_box(draw, box, score, label):
    """
    Draws a bounding box on an image with the given score and label.

    Args:
    - draw (ImageDraw): ImageDraw object for drawing.
    - box (Tensor): Bounding box coordinates as [x_min, y_min, x_max, y_max].
    - score (float): Confidence score of the detection.
    - label (int): Class label of the detection.
    """
    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)
    draw.text((x_min, y_min), f"{label} ({score:.2f})", fill="red")

def main():
    input_folder = 'data/images/fire'
    output_folder = 'data/images/fire_overlay'
    model_weights_path = 'weights/model_epoch_1.pth'  

    # Load the pretrained DETR model and processor
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load the model weights
    if os.path.exists(model_weights_path):
        print(f"Loading model weights from {model_weights_path}...")
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
        print("Model weights loaded successfully.")
    else:
        print(f"Model weights file not found at {model_weights_path}. Using the pretrained weights.")

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run inference and overlay bounding boxes
    run_inference_on_images(input_folder, output_folder, model, processor, device)

if __name__ == "__main__":
    main()
