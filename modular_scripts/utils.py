"""
Contains various utility functions for PyTorch model training and saving
"""
import torch, os, sys, requests, zipfile, gdown, torchvision
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
import matplotlib.pyplot as plt

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str):
    """Saves a PyTorch model to a target directory
    
    Args:
        model: A target PyTorch model to save
        target_dir: A directory to save the model
        model_name: A filename for the saved model. Should include 
        either .pth ot .pt as the file extension
    
    Example usage:
        save_model(
            model=model_0,
            target_dir="models",
            model_name="rubbish_classifier.pth"ArithmeticError
        )
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def image_convertor(path: str, format: str):
    """Converts Images from a given path into a given format.
    
    Args:
        path: String path of the images to convert
        format: Format to convert to
    
    Example of use:
        # cans class convertion
        image_convertor(path="data/dataset/cans/",
                        format="jpg")
    """
    count=0
    path=Path(path)
    for file in tqdm(path.glob("./*")):
        f, e = os.path.splitext(file)
        renameFile = f + "."+format.lower()
        if e.lower() != "."+format.lower():
            old_file=file
            count+=1
            try:
                with Image.open(file) as img:
                    img.save(renameFile)
            except OSError:
                print("cannot convert", file)
            os.remove(old_file)
    print(f"{count} images converted to '{format}' in '{path}'")

def get_data(zip_file_id: str):
    """Dowload and extrac data from a Zip file from a Google Drive folder
    
    Args:
        zip_file_id: Id of the zip file folder, make sure select sharing
        to anyone with the link
    """
    
    # Setup a path to a data folder
    data_path = Path("data/")
    images_path = data_path / "images_dataset"

    # If the data folder doesn't exist, download it and prepare it.
    if images_path.is_dir():
        print(f"'{images_path}' directory already exists, skipping directory creation...")
    else:
        print(f"'{images_path}' does not exist, creating directory...")
        images_path.mkdir(parents=True, exist_ok=True)

    url = 'https://drive.google.com/uc?id='+ zip_file_id
    output = str(data_path)+'/dataset.zip'
    gdown.download(url, output, quiet=False)

    # Unzip data
    with zipfile.ZipFile(data_path / "dataset.zip", "r") as zip_ref:
        print("Unzipping data...")
        zip_ref.extractall(data_path)
        
    os.remove(str(data_path)+"/dataset.zip")

    return images_path

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = device):
    """Make a prediction and plot an image using a given model.
    
    Args:
        mode: Model to make the prediction
        image_path: Path of the image on which the prediction will occur
        class_names: Names of the classess to predict
        transform: Transforms to apply on the image to predict. Good practices to use same as the used on the pretrained model.
        device: Target device
    
    Example of use:
        pred_and_plot_image(model=model, 
                            image_path=image_path,
                            class_names=class_names,
                            transform=weights.transforms(), optionally pass in a specified transform from our pretrained model weights
                            image_size=(224, 224))
    """
    # Open Image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

        ])
    
    ### Predict on image ### 
    # Make sure mode is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimesion to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)
        # Make prediction on image with extra dimension and send it ot the target device
        target_image_pred = model(transformed_image.to(device))
    
    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)
