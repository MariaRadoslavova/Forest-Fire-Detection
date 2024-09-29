import torch
from torchmetrics import Accuracy  # Make sure to install torchmetrics if not installed

def train_model(train_dataloader, test_dataloader, model, optimizer, device, num_classes, epochs=10):
    model.to(device)

    # Initialize accuracy metric
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train_samples = 0
        total_batches = len(train_dataloader)

        for batch_idx, (pixel_values, target) in enumerate(train_dataloader):
            pixel_values = pixel_values.to(device)
            target = [{k: v.to(device) for k, v in t.items()} for t in target]

            outputs = model(pixel_values=pixel_values, labels=target)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=2)  # Assuming the logits shape is [batch_size, num_queries, num_classes]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            # Flatten the list of targets to match predictions
            true_labels = torch.cat([t['labels'] for t in target], dim=0)
            pred_labels = torch.cat([p for p in preds], dim=0)

            # Calculate accuracy
            running_corrects += accuracy_metric(pred_labels, true_labels).item()
            total_train_samples += true_labels.size(0)

            # Print training loss for every mini-batch
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Batch {batch_idx+1}/{total_batches}, Training Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / total_batches
        avg_train_acc = running_corrects / total_batches
        print(f"  Average Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_acc:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val_samples = 0
        val_batches = len(test_dataloader)

        with torch.no_grad():
            for batch_idx, (pixel_values, target) in enumerate(test_dataloader):
                pixel_values = pixel_values.to(device)
                target = [{k: v.to(device) for k, v in t.items()} for t in target]

                outputs = model(pixel_values=pixel_values, labels=target)
                val_loss = outputs.loss
                preds = torch.argmax(outputs.logits, dim=2)

                val_running_loss += val_loss.item()

                # Flatten the list of targets to match predictions
                true_labels = torch.cat([t['labels'] for t in target], dim=0)
                pred_labels = torch.cat([p for p in preds], dim=0)

                # Calculate accuracy
                val_running_corrects += accuracy_metric(pred_labels, true_labels).item()
                total_val_samples += true_labels.size(0)

                # Print validation loss for every mini-batch
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == val_batches:
                    print(f"  Batch {batch_idx+1}/{val_batches}, Validation Loss: {val_loss.item():.4f}")

        avg_val_loss = val_running_loss / val_batches
        avg_val_acc = val_running_corrects / val_batches
        print(f"  Average Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.4f}")