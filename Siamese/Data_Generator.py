import os
import random
import shutil

data_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset'
positive_pairs_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\positive_pairs'

# Create the directory for positive pairs if it doesn't exist
os.makedirs(positive_pairs_dir, exist_ok=True)

broken_images = os.listdir(os.path.join(data_dir, "Broken"))

for i in range(len(broken_images)):
    img1 = broken_images[i]
    img2 = random.choice(broken_images)

    img1_path = os.path.join(data_dir, "Broken", img1)
    img2_path = os.path.join(data_dir, "Broken", img2)

    new_pair_dir = os.path.join(positive_pairs_dir, f"pair_{i}")
    os.makedirs(new_pair_dir, exist_ok=True)

    shutil.copy(img1_path, os.path.join(new_pair_dir, "image1.jpg"))
    shutil.copy(img2_path, os.path.join(new_pair_dir, "image2.jpg"))

negative_pairs_dir = 'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\\negative_pairs'

# Create the directory for negative pairs if it doesn't exist
os.makedirs(negative_pairs_dir, exist_ok=True)

normal_images = os.listdir(os.path.join(data_dir, "Normal"))

for i in range(len(broken_images)):
    img1 = broken_images[i]
    img2 = random.choice(normal_images)

    img1_path = os.path.join(data_dir, "Broken", img1)
    img2_path = os.path.join(data_dir, "Normal", img2)

    new_pair_dir = os.path.join(negative_pairs_dir, f"pair_{i}")
    os.makedirs(new_pair_dir, exist_ok=True)

    shutil.copy(img1_path, os.path.join(new_pair_dir, "image1.jpg"))
    shutil.copy(img2_path, os.path.join(new_pair_dir, "image2.jpg"))
