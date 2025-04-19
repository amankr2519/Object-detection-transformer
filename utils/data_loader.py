import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import json

class CustomCocoDataset(Dataset):
    def __init__(self, img_dir, annotation_path, transforms=None):
        self.img_dir = img_dir
        self.annotation_path = annotation_path
        self.transforms = transforms

        with open(annotation_path) as f:
            coco = json.load(f)

        self.images = coco['images']
        self.annotations = coco['annotations']
        self.categories = coco['categories']

        # Create index: image_id -> list of annotations
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        # Map category id to label index
        self.cat2label = {cat['id']: i for i, cat in enumerate(self.categories)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        img_id = image_info['id']
        img_path = os.path.join(self.img_dir, image_info['file_name'])

        image = Image.open(img_path).convert("RGB")

        # Get annotations
        anns = self.image_id_to_annotations.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat2label[ann['category_id']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


# Default image transforms
def get_transform():
    return T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
