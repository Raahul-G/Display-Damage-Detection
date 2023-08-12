import albumentations as A
import cv2
import os

# data_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\Train'
data_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\Valid'


transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.8),
        A.Blur(p=0.3),
        A.Sharpen(p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.7)
        ])

# Loop through the images in the data directory
for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(class_path, filename)
                image = cv2.imread(image_path)

                try:
                    augmented = transforms(image=image)
                    augmented_image = augmented['image']

                    augmented_image_path = os.path.join(class_path, f'augmented_{filename}')
                    cv2.imwrite(augmented_image_path, augmented_image)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

print("Augmentation completed!")