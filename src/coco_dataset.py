import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

# Define a custom dataset for loading COCO-style annotations
class COCODataset(Dataset):
    def __init__(self, annotations_file, img_dir, processor):
        with open(annotations_file) as f:
            self.coco = json.load(f)

        self.img_dir = img_dir
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image file path
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Get the corresponding annotations
        img_id = img_info['id']
        annotations = [ann for ann in self.annotations if ann['image_id'] == img_id]

        boxes = []
        labels = []
        image_width = img_info['width']
        image_height = img_info['height']

        for ann in annotations:
            bbox = ann['bbox']
            # Convert bbox format from [x, y, width, height] to normalized [x_min, y_min, x_max, y_max]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height

            # Normalize the box coordinates between 0 and 1
            x_min /= image_width
            y_min /= image_height
            x_max /= image_width
            y_max /= image_height

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        encoding = self.processor(images=image, return_tensors="pt")

        target = {'boxes': boxes, 'labels': labels}

        return encoding['pixel_values'].squeeze(0), target


def collate_fn(batch):
    pixel_values = []
    targets = []

    for item in batch:
        # item[0] contains the image tensor, and item[1] contains the annotations (boxes and labels)
        image = item[0]  # Access the image tensor from the tuple

        # Resize images to the same size (e.g., 512x512) if necessary (optional, depending on input size)
        if image.size()[-2:] != (512, 512):
            # Assuming you're working with a tensor, you can use torch.nn.functional.interpolate to resize it
            image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)

        pixel_values.append(image)

        # The second part of the tuple (item[1]) is the target (bounding boxes and labels)
        target = item[1]

        # Rename 'labels' to 'class_labels' as required by the model
        target['class_labels'] = target.pop('labels')
        targets.append(target)

    # Stack the pixel values (images) and return targets
    pixel_values = torch.stack(pixel_values, dim=0)
    return pixel_values, targets