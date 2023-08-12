import os
import random
import shutil
import cv2

data_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\Train'

positive_pairs_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\positive_pairs'
negative_pairs_dir = 'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\\negative_pairs'

# Create the directory for positive pairs & negative pairs if it doesn't exist
os.makedirs(positive_pairs_dir, exist_ok=True)
os.makedirs(negative_pairs_dir, exist_ok=True)

broken_images = os.listdir(os.path.join(data_dir, "Broken"))
normal_images = os.listdir(os.path.join(data_dir, "Normal"))

# Define the target size for resizing
target_size = (224, 224)


def resize_and_save(source_path, destination_folder, filename):
    image = cv2.imread(source_path)
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    new_image_path = os.path.join(destination_folder, filename)
    cv2.imwrite(new_image_path, resized_image)


for i in range(len(broken_images)):
    img1 = broken_images[i]
    img2 = random.choice(broken_images)

    img1_path = os.path.join(data_dir, "Broken", img1)
    img2_path = os.path.join(data_dir, "Broken", img2)

    new_pair_dir = os.path.join(positive_pairs_dir, f"pair_{i}")
    os.makedirs(new_pair_dir, exist_ok=True)

    # Resize and save images in the pair directory
    resize_and_save(img1_path, new_pair_dir, "image1.jpg")
    resize_and_save(img2_path, new_pair_dir, "image2.jpg")

    with open(os.path.join(new_pair_dir, "label.txt"), "w") as f:
        f.write("1")

for i in range(len(broken_images)):
    img1 = broken_images[i]
    img2 = random.choice(normal_images)

    img1_path = os.path.join(data_dir, "Broken", img1)
    img2_path = os.path.join(data_dir, "Normal", img2)

    new_pair_dir = os.path.join(negative_pairs_dir, f"pair_{i}")
    os.makedirs(new_pair_dir, exist_ok=True)

    # Resize and save images in the pair directory
    resize_and_save(img1_path, new_pair_dir, "image1.jpg")
    resize_and_save(img2_path, new_pair_dir, "image2.jpg")

    with open(os.path.join(new_pair_dir, "label.txt"), "w") as f:
        f.write("0")