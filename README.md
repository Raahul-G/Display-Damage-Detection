# Display-Damage-Detection

## Objective:

The objective of this readme is to outline various approaches and considerations for building a Python program to determine if a screen is damaged or not using image classification techniques.

## Approach:

### Approach 1: Edge Detection without Machine Learning

- Utilize edge detection techniques such as Canny.
- Reasons for not choosing:
  1. Only works when the screen is off.
  2. No reflections on the screen should be present.
  3. Limitations of the Canny algorithm.
  4. Sensitivity to slight pixel differences.

### Approach 2: CLIP Model - Zero-Shot Image Classification

The CLIP model by OpenAI is a promising option for image classification.

- Developed to explore robustness and generalization in computer vision.
- Combines ViT-B/32 Transformer for image and masked self-attention Transformer for text.
- Trained to maximize similarity between (image, text) pairs.
- **Shortcomings**:Library compatibility issue with Transformers and Streamlit.
- **Tried**:
  1. Created separate virtual environments for Transformers and Streamlit - Not Optimal Approach.
  2. Downgraded Streamlit to version 1.16.
- **Disadvantages**:
  1. Unsuitable for production.
  2. No fine-tuning for specific use cases.
- **Advantages**:
  1. Lower complexity.
  2. No data building or training costs.
  3. Well-maintained repository.

### Approach 3: CNN - VGG16 Transfer Learning

Transfer learning with VGG16 for binary classification.

- Utilizes pre-trained VGG16 model.
- Reuses lower-level features, and fine-tunes top layers.
- Beneficial for small datasets or related tasks.
- Reduces overfitting, and speeds up convergence.

Custom Dataset Generation

- Researched data sources, and decided to create a custom dataset.
- Used Canva, Google Images, and mobile camera.
- Generated images of normal and broken displays.
- Applied augmentation techniques.
- **Shortcomings**: No testing data for inference.
- **Assumptions**:
- **Summary**: Custom data generation solved the problem, but lacked test data.
- **Future Scope**: Generate more diverse samples for Train, Validation, and Test sets.
- **Assumptions**
    - The document is intended for use with a custom image dataset.
    - The lighting conditions during image capture are assumed to be normal.
    - The display screen should cover more than 85% of the image.
    - Laptop screens are used as a substitute for TV display screens.
    - Images only contain the laptop or TV screen; no other objects or persons are present

### Approach 4: Siamese Architecture

Siamese model for pairwise image comparison tasks.

- Ideal for limited data and nuanced distinctions.
- Suited for one-shot learning.
- Compares image pairs for tasks like face recognition.
- **Shortcomings**:
  1. Memory allocation issues with custom image generators.
  2. Reduce the image size before the custom image generators to avoid the issue

---

_This document outlines various approaches for building a Python program to classify screen damage using computer vision techniques. Each approach has its advantages and limitations, and the choice of approach depends on factors like data availability, complexity, and performance requirements._

