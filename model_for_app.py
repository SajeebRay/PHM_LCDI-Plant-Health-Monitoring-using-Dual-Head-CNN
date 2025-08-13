import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

Names = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn",
    "Grape",
    "Orange",
    "Peach",
    "Pepper,",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]

diseases = [
    "Apple Black Rot",
    "Apple Healthy",
    "Apple Scab",
    "Blueberry Healthy",
    "Cedar Apple Rust",
    "Cherry Healthy",
    "Cherry Powdery Mildew",
    "Corn Cercospora Leaf Spot",
    "Corn Common Rust",
    "Corn Healthy",
    "Corn Northern Leaf Blight",
    "Grape Black Rot",
    "Grape Esca",
    "Grape Healthy",
    "Grape Leaf Blight",
    "Orange Haunglongbing",
    "Peach Bacterial Spot",
    "Peach Healthy",
    "Pepper Bacterial Spot",
    "Pepper Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Healthy",
    "Strawberry Leaf Scorch",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato healthy",
]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(256, 14)  # Plant type
        self.fc2 = nn.Linear(256, 38)  # Disease type

    def forward(self, xb):
        x = self.network(xb)
        return self.fc1(x), self.fc2(x)


def load_model_and_labels():
    model = Model()
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model, Names, diseases


def predict_image(image: Image.Image, model, return_probs=False):
    preprocess = transforms.Compose(
        [transforms.Resize((128, 128), antialias=True), transforms.ToTensor()]
    )
    x = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        out_leaf, out_disease = model(x)
        leaf_probs = F.softmax(out_leaf, dim=1)
        disease_probs = F.softmax(out_disease, dim=1)

        leaf_conf, leaf_pred = torch.max(leaf_probs, dim=1)
        disease_conf, disease_pred = torch.max(disease_probs, dim=1)

    if return_probs:
        return (
            leaf_pred.item(),
            disease_pred.item(),
            leaf_conf.item(),
            disease_conf.item(),
            disease_probs.squeeze().tolist(),  # added
        )
    else:
        return (
            leaf_pred.item(),
            disease_pred.item(),
            leaf_conf.item(),
            disease_conf.item(),
        )
