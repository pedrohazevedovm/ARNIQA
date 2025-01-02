import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from PIL.Image import Image as PILImage
from typing import List
import os
import pandas as pd
from hubconf import ARNIQA


def center_corners_crop(img: PILImage, crop_size: int = 224) -> List[PILImage]:
    """
    Return the center crop and the four corners of the image.

    Args:
        img (PIL.Image): image to crop
        crop_size (int): size of each crop

    Returns:
        crops (List[PIL.Image]): list of the five crops
    """
    width, height = img.size

    # Calculate the coordinates for the center crop and the four corners
    cx = width // 2
    cy = height // 2
    crops = [
        TF.crop(img, cy - crop_size // 2, cx - crop_size // 2, crop_size, crop_size),  # Center
        TF.crop(img, 0, 0, crop_size, crop_size),  # Top-left corner
        TF.crop(img, height - crop_size, 0, crop_size, crop_size),  # Bottom-left corner
        TF.crop(img, 0, width - crop_size, crop_size, crop_size),  # Top-right corner
        TF.crop(img, height - crop_size, width - crop_size, crop_size, crop_size)  # Bottom-right corner
    ]

    return crops

def main(directory: str) -> None:
    iqa_score_dict = {"image_name": [], "regressor_dataset": [], "score": []}

    # Preparing the images list
    dir_path = os.path.join("assets", directory)
    images_list = []
    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            images_list.append(image_path)

    # Set the device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    regressors_datasets_list = ["koniq10k"]

    for regressor_dataset in regressors_datasets_list:
        # Load the model
        # model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
        #                        regressor_dataset=regressor_dataset)
        model = ARNIQA(regressor_dataset).to(device)
        model.eval().to(device)

        # Define the normalization transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Load the full-scale image
        for image_path in images_list:
            img_name = os.path.basename(image_path)  # Nome da imagem
            img = Image.open(image_path).convert("RGB")

            # Get the half-scale image
            img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

            # Get the center and corners crops
            img = center_corners_crop(img_ds, crop_size=224)
            img_ds = center_corners_crop(img_ds, crop_size=224)

            # Preprocess the images
            img = [transforms.ToTensor()(crop) for crop in img]
            img = torch.stack(img, dim=0)
            img = normalize(img).to(device)
            img_ds = [transforms.ToTensor()(crop) for crop in img_ds]
            img_ds = torch.stack(img_ds, dim=0)
            img_ds = normalize(img_ds).to(device)

            # Compute the quality score
            with torch.no_grad(), torch.cuda.amp.autocast():
                score = model(img, img_ds, return_embedding=False, scale_score=True)
                # Compute the average score over the crops
                score = score.mean(0)

            iqa_score_dict["image_name"].append(img_name)
            iqa_score_dict["regressor_dataset"].append(regressor_dataset)
            iqa_score_dict["score"].append(score.item())

            print(f"Image: {img_name}, Regressor Dataset: {regressor_dataset}, Score: {score.item()}")

    # Create a DataFrame to add the result
    df = pd.DataFrame(iqa_score_dict)

    # Save the DataFrame in a csv file
    df.to_csv(f"image_quality_scores_{directory}_with_crop_complete_KONIQ_PRETRAINED.csv", index=False)
    print(f"Resultados salvos em 'image_quality_scores_{directory}_with_crop_KONIQ.csv'.")


if __name__ == "__main__":
    main('HRIQ_HQ')
