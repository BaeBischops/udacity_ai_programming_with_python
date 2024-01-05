import argparse
import torch
from torchvision import models
from torch import optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np

def load_checkpoint(filepath, arch='vgg16', learning_rate=0.001):
    checkpoint = torch.load(filepath)

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])

    epochs = checkpoint['epochs']

    return model, optimizer, epochs

def process_image(image_path):
    """
    Processes an image for a PyTorch model.

    Parameters:
        - image_path (str): Path to the image file.

    Returns:
        - np_image (numpy.ndarray): Processed image as a NumPy array.
    """
    # Load the image with error handling
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening the image at {image_path}: {e}")
        return None

    # Constants for image processing
    target_size = 256
    crop_size = 224

    # Resize the image
    pil_image.thumbnail((target_size, target_size))

    # Crop the center portion of the image
    left_margin = (pil_image.width - crop_size) / 2
    bottom_margin = (pil_image.height - crop_size) / 2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Convert image to numpy array and scale values to range [0, 1]
    np_image = np.array(pil_image) / 255.0

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions for PyTorch
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def imshow(image, ax=None, title=None):
    """
    Displays a PyTorch tensor as an image.

    Parameters:
        - image (torch.Tensor): Input image tensor.
        - ax (matplotlib.axes.Axes): Axes to display the image.
        - title (str): Title for the displayed image.

    Returns:
        - ax (matplotlib.axes.Axes): Axes with the displayed image.
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension,
    # but matplotlib assumes it is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean_values = np.array([0.485, 0.456, 0.406])
    std_values = np.array([0.229, 0.224, 0.225])
    image = std_values * image + mean_values

    # Clip image values between 0 and 1
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    if title:
        ax.set_title(title)

    return ax

def predict(image_path, model, device, topk=5):
    """Predict the class (or classes) of an image using a trained deep learning model.

    Parameters:
        - image_path (str): Path to the image file.
        - model (torch.nn.Module): Pre-trained deep learning model.
        - device (torch.device): Device (cpu or gpu) to perform the inference.
        - topk (int): Number of top predictions to return.

    Returns:
        - top_probs (list): List of top probabilities for each predicted class.
        - top_classes (list): List of top predicted class labels.
    """
    try:
        # Process the image
        img = process_image(image_path)

        # Convert to PyTorch tensor and add batch dimension
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)

        # Ensure the tensor is on the same device as the model
        img_tensor = img_tensor.to(device)

        # Make predictions
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            log_probs = model.forward(img_tensor)

        # Calculate the class probabilities
        ps = torch.exp(log_probs)

        # Get the topk results
        top_probs, top_indices = ps.topk(topk)

        # Convert top_probs to list
        top_probs = top_probs.cpu().numpy().tolist()[0]

        # Convert to class labels
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_classes = [idx_to_class[idx] for idx in top_indices.cpu().numpy()[0]]

        return top_probs, top_classes

    except Exception as e:
        print(f"Error predicting the image: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Predict flower class from an image')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the saved model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default="", help='Path to a JSON file for mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument("--mps", action="store_true", help="Use MPS (Apple Mac) for training")

    args = parser.parse_args()

    # Load model from checkpoint
    model, _, _ = load_checkpoint(args.checkpoint)

    # Determine device
    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model.to(device)

    # Make predictions
    probs, classes = predict(args.image_path, model, device, args.top_k)
    
    # Map classes to real names if --category_names is provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]
    
    print("Probabilities:", probs)
    print("Classes:", classes)

if __name__ == "__main__":
    main()