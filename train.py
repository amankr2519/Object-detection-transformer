import os
import torch
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import os
from models.detr import DETR
from utils.criterion import SetCriterion, HungarianMatcher

# Get absolute path to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



# Paths
img_dir = os.path.join(BASE_DIR, "data/images")
ann_file = os.path.join(BASE_DIR, "data/annotations/instances_train.json")

print("Loading from path:", ann_file)
print("Exists?", os.path.exists(ann_file))


# Transforms
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

# Dataset
class CocoWrapper(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        if self._transforms is not None:
            img = self._transforms(img)

        # Transform COCO annotation to DETR format
        w, h = img.shape[-1], img.shape[-2]
        boxes = []
        labels = []
        for obj in target:
            xmin, ymin, w_box, h_box = obj["bbox"]
            xmax = xmin + w_box
            ymax = ymin + h_box
            boxes.append([xmin / w, ymin / h, xmax / w, ymax / h])
            labels.append(obj["category_id"] - 1)  # Make labels start at 0

        target_out = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return img, target_out

# Load Data
dataset = CocoWrapper(img_dir, ann_file, transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: list(zip(*x)))


# Model, Matcher, Criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DETR(num_classes=2).to(device)
matcher = HungarianMatcher()
criterion = SetCriterion(num_classes=2, matcher=matcher)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training Loop
for epoch in range(10):  # keep it small for testing
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = torch.stack(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict[k] for k in loss_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
