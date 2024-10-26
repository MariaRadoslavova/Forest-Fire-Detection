import torch
import os

def infer_model(test_dataloader, model, device, results_dir='results'):
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, (pixel_values, _) in enumerate(test_dataloader):
            pixel_values = pixel_values.to(device)

            # Run the model on the input
            outputs = model(pixel_values=pixel_values)

            # Post-process the output
            results = model.post_process(outputs, target_sizes=[pixel_values.shape[-2:]])

            # Save the results
            save_inference_result(batch_idx, results, results_dir)

def save_inference_result(batch_idx, results, results_dir):
    """
    Save the inference results for each batch as a JSON file.
    """
    result_path = os.path.join(results_dir, f"result_{batch_idx}.json")
    with open(result_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved inference result for batch {batch_idx} to {result_path}")
