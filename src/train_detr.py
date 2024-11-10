import torch
import os

def train_model(train_dataloader, test_dataloader, model, optimizer, device, epochs=10, save_dir='weights'):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        total_batches = len(train_dataloader)

        for batch_idx, (pixel_values, target) in enumerate(train_dataloader):
            pixel_values = pixel_values.to(device)
            target = [{k: v.to(device) for k, v in t.items()} for t in target]

            outputs = model(pixel_values=pixel_values, labels=target)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            # Print loss for every mini-batch
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Batch {batch_idx+1}/{total_batches}, Training Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / total_batches
        print(f"  Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_batches = len(test_dataloader)

        with torch.no_grad():
            for batch_idx, (pixel_values, target) in enumerate(test_dataloader):
                pixel_values = pixel_values.to(device)
                target = [{k: v.to(device) for k, v in t.items()} for t in target]

                outputs = model(pixel_values=pixel_values, labels=target)
                val_loss = outputs.loss

                val_running_loss += val_loss.item()

                # Print validation loss for every mini-batch
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == val_batches:
                    print(f"  Batch {batch_idx+1}/{val_batches}, Validation Loss: {val_loss.item():.4f}")

        avg_val_loss = val_running_loss / val_batches
        print(f"  Average Validation Loss: {avg_val_loss:.4f}")

        # Save model weights at the end of each epoch
        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"  Model weights saved at: {model_save_path}")
