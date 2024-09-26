import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd


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
    regressors_datasets_list = ["live", "csiq", "tid2013", "kadid10k", "flive", "spaq", "clive", "koniq10k"]

    for regressor_dataset in regressors_datasets_list:
        # Load the model
        model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                               regressor_dataset=regressor_dataset)
        model.eval().to(device)

        # Define the preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the full-scale image
        for image_path in images_list:
            img_name = os.path.basename(image_path)  # Nome da imagem
            img = Image.open(image_path).convert("RGB")

            # Get the half-scale image
            img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

            # Preprocess the images
            img = preprocess(img).unsqueeze(0).to(device)
            img_ds = preprocess(img_ds).unsqueeze(0).to(device)

            # Compute the quality score
            with torch.no_grad(), torch.cuda.amp.autocast():
                score = model(img, img_ds, return_embedding=False, scale_score=True)


            iqa_score_dict["image_name"].append(img_name)
            iqa_score_dict["regressor_dataset"].append(regressor_dataset)
            iqa_score_dict["score"].append(score.item())

            print(f"Image: {img_name}, Regressor Dataset: {regressor_dataset}, Score: {score.item()}")

    # Create a DataFrame to add the result
    df = pd.DataFrame(iqa_score_dict)

    # Save the DataFrame in a csv file
    df.to_csv(f"image_quality_scores_{directory}_cluster.csv", index=False)
    print(f"Resultados salvos em 'image_quality_scores_{directory}_cluster.csv'.")


if __name__ == "__main__":
    main('HRIQ')
    main('Motorola')
