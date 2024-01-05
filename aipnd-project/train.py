# Python Standard Library
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models

def model_init(arch="vgg16", learning_rate=0.001, hidden_units=4096, dropout_rate=0.5, use_gpu=False, use_mps=False):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, 102)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features
        model.classifier = nn.Linear(input_features, 102)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    if use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    criterion.to(device)

    return model, criterion, optimizer, scheduler, device

import torch

def save_checkpoint(model, optimizer, epochs, save_path, arch='vgg16', class_to_idx=None):
    """
    Save a PyTorch model checkpoint.

    Parameters:
        - model (torch.nn.Module): The PyTorch model to be saved.
        - optimizer (torch.optim.Optimizer): The optimizer associated with the model.
        - epochs (int): Number of training epochs.
        - save_path (str): File path to save the checkpoint.
        - arch (str): Architecture identifier (default: 'vgg16').
        - class_to_idx (dict): Mapping from class labels to indices (default: None).
    """
    checkpoint = {
        'architecture': arch,
        'classifier': model.classifier,
        'class_to_idx': class_to_idx if class_to_idx is not None else model.class_to_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
    }

    torch.save(checkpoint, save_path)



def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    parser.add_argument('data_directory', type=str, help='Data directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', 
                        choices=['vgg16', 'resnet18', 'densenet121'], 
                        help='Architecture [available: vgg16, resnet18, densenet121]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--mps", action="store_true", help="Use MPS (Apple Mac) for training")
    
    args = parser.parse_args()

    epochs = args.epochs
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    print_every = 50
    steps = 0
    best_val_loss = float('inf')
    patience = 5
    no_improve_epoch = 0
    
    # Define transforms
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    }

    batch_size = 64
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True),
    }
    
    if args.gpu and args.mps:
        raise ValueError("Cannot use both GPU and MPS. Choose one.")

    model, criterion, optimizer, scheduler, device = model_init(
        arch=args.arch,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        use_gpu=args.gpu,
        use_mps=args.mps
    )
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    # Optimization: For Gradient Accumulation
    accumulation_steps = 4

    # Optimization: Learning Rate Scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        running_loss = 0
        model.train()
        train_bar = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs} [Train]", position=0, leave=False, dynamic_ncols=True)
        for inputs, labels in train_bar:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()

            if steps % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

            if steps % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}.. Train loss: {running_loss/print_every:.3f}")
                running_loss = 0

        # Evaluation after each epoch
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            valid_bar = tqdm(dataloaders['valid'], desc=f"Epoch {epoch+1}/{epochs} [Validate]", position=0, leave=False, dynamic_ncols=True)
            for inputs, labels in valid_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Check if validation loss improved for early stopping
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss 
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1

            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

            # Early stopping check
            if no_improve_epoch >= patience:
                print("Stopping early due to no improvement!")
                break

            # Step the learning rate scheduler
            scheduler.step(valid_loss)

            print("Training finished!")

    model.class_to_idx = image_datasets['train'].class_to_idx
    save_path = 'checkpoint.pth'
    save_checkpoint(model, optimizer, epochs, save_path)

if __name__ == "__main__":
    main()