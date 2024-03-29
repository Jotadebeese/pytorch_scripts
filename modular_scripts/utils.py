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
import shutil

try:
    import splitfolders
except ModuleNotFoundError:
    import subprocess
    import sys

    print("[INFO] splitfolders module not found. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'split-folders[full]'])
    import splitfolders


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

def split_data(input_folder: str, output_folder: str, ratio: tuple = (.8, .1, .1)):
    """Split data into train, val and test datasets using split-folders: https://github.com/jfilter/split-folders/tree/main

    Args:
        input_folder: path of the folder with the dataset with the following format:
                    input/
                        class1/
                            img1.jpg
                            img2.jpg
                            ...
                        class2/
                            imgWhatever.jpg
                            ...
                        ...
        output_folder: path to save the datasets as follows:
                    output/
                        train/
                            class1/
                                img1.jpg
                                ...
                            class2/
                                imga.jpg
                                ...
                        val/
                            class1/
                                img2.jpg
                                ...
                            class2/
                                imgb.jpg
                                ...
                        test/
                            class1/
                                img3.jpg
                                ...
                            class2/
                                imgc.jpg
                                ...
        ratio: tuple to specify the ratio to split into train, val and test.
    
    Example of use:
        split_data(input_folder='data/rubbish_dataset', output_folder=data/rubbish_dataset, ratio=(.8, .1, .1))
    """
    # Setup directory path
    train_dir = output_folder / "train"
    test_dir = output_folder / "test"
    val_dir = output_folder / "val"

    splitfolders.ratio(input_folder, output=output_folder,
                        seed=1337, ratio=ratio, group_prefix=None, move=False) # default values
    shutil.rmtree(input_folder)

    return train_dir, test_dir, val_dir
    
def bulk_image_convertor(dataset_path: str, format: str ="jpg"):
    """Converts Images from the labels folders of a given dataset folder into a given format.
    
    Args:
        path: String path of dataset folder
        format: Format to convert to
    
    Example of use:
        # cans class convertion
        bulk_image_convertor(path="data/dataset",
                        format="jpg")
    """
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)

        # Check if the current item in the dataset path is a directory
        if os.path.isdir(label_path):
            # Call image_convertor for the current label folder
            image_convertor(label_path, format=format)

def image_convertor(path: str, format: str = 'jpg'):
    """Converts Images from a given folder into a given format.
    
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

def download_data(
    source: str,
    destination: str,
    from_gdrive: bool = False,
    remove_source: bool = True) -> Path:
    """Download and extract the data of a Zip file from am external source like GitHub or a Gooogle Drive folder.
    
    Args:
        source (str): A link to a zipped file containing data. In case of a Gooogle Drive, provide only the file ID
        destination (str): name of the folder where you want to save the data
        remove_source: boolean to remove source file after downloading
    """
    # Setup a path to a data folder
    data_path = Path("data/")
    images_path = data_path / destination

    # If the data folder doesn't exist, download it and prepare it.
    if images_path.is_dir():
        print(f"'{images_path}' directory already exists, skipping directory creation...")
    else:
        print(f"'{images_path}' does not exist, creating directory...")
        images_path.mkdir(parents=True, exist_ok=True)

    target_file = Path(source).name

    if from_gdrive:
        print(f"[INFO] Upgrading gdown from version: {gdown.__version__}, ")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade','--no-cache-dir', 'gdown'])
        print(f"[INFO] to version: {gdown.__version__}")
        url = 'https://drive.google.com/uc?id='+ source + '&confirm=t'
        output = str(data_path) + '/' + str(target_file)
        print(f"[INFO] Donwloading {target_file} from https://drive.google.com/uc?id={source}")
        gdown.download(url, output, quiet=False)
    else:
        with open(data_path / target_file, 'wb') as f:
            print(f"[INFO] Downloading {target_file} from {source}...")
            res = requests.get(source)
            f.write(res.content)

    # Unzip data
    with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
        print(f"[INFO] Unzipping {target_file} data...")
        zip_ref.extractall(images_path)
    
    if remove_source:
        os.remove(data_path / target_file)

    return images_path

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


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

def save_image(image, label, index, base_folder):
    """
    Saves an image to a given folder.

    Args:
        image: Image to save
        label: Label of the image
        index: Index of the image
        base_folder: Folder to save the image

    Example of use:
        sav_image(image, label, index, base_folder)
    """

    import io
    import os
    from PIL import Image

    folder_path = os.path.join(base_folder, str(label))
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'image_{index}.jpg')

    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        image_data = output.getvalue()
    
    with open(file_path, 'wb') as f:
        f.write(image_data)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
