import torch
import torchvision.transforms as T
from PIL import Image


inference_transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model(scripted_model_path="best_model_scripted.pth", device="cpu"):
    """
    Loads the TorchScript model for efficient inference.
    """
    # 1) Load the scripted model
    model = torch.jit.load(scripted_model_path, map_location=device)
    model.eval()
    return model


def predict_image(model, image_path, device="cpu"):
    """
    Runs inference on a single image file (or PIL Image).
    Returns raw logits or probabilities, depending on your needs.
    """
    # 1) Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = inference_transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)
    img_tensor = img_tensor.to(device)

    # 2) Inference (no grad needed)
    with torch.no_grad():
        outputs = model(img_tensor)  # shape: (1, num_classes)

    # 3) Convert to probabilities, if desired
    probs = torch.softmax(outputs, dim=1)  # shape: (1, num_classes)
    return probs
