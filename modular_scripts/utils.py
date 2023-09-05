"""
Contains various utility functions for PyTorch model training and saving
"""
import torch, os, sys, requests, zipfile, gdown
from pathlib import Path
from tqdm import tqdm
from PIL import Image

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
    image_path = data_path / "images_dataset"

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

    return image_path
