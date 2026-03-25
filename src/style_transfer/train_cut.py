import os
import torch
import wandb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Settings W&B
wandb.init(
    project="style-transfer",
    name="knn-proj",
    config={"model": "CUT", "batch_size": 1, "input_size": 112, "lr": 0.0002},
)


# Dataset Loader
class NewspaperDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            f for f in os.listdir(root_dir) if f.endswith((".png", ".jpg"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# Transformation preparer
transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Paths to datasets
# TODO: Make sure to have the correct paths to datasets
dataset_clean = NewspaperDataset(root_dir="../../data/trainA", transform=transform)
loader_clean = DataLoader(
    dataset_clean, batch_size=wandb.config.batch_size, shuffle=True
)

# Simulation training loop
print("Starting training loop...")
for epoch in range(1):  # TODO: Number of epochs
    for i, real_A in enumerate(loader_clean):
        mock_loss = 1.0 / (epoch + i + 1)

        wandb.log(
            {
                "epoch": epoch,
                "loss_G": mock_loss,
                "real_A_sample": [wandb.Image(real_A[0], caption="Vstupná čistá tvár")],
            }
        )

        if i == 10:
            break

print("Training loop finished.")
wandb.finish()
