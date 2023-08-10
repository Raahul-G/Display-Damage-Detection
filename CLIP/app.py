from PIL import Image
import requests
import streamlit as st
from transformers import CLIPProcessor, CLIPModel

# Streamlit app title and description
st.title("3Ds - Zero-Shot Image Classification")
st.write("Upload an image using URL and get its predicted label.")

# Load OpenAI CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Streamlit UI components
image_url = st.text_input("Enter Image URL:")
submit_button = st.button("Submit")

# Process image and get label
if submit_button and image_url:
    try:
        # Load and display the image
        image = Image.open(requests.get(image_url, stream=True).raw)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        labels = ["Damaged Screen", "Not Damaged"]

        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        st.write(f"Predicted Label: {labels[probs.argmax().item()]}")
    except Exception as e:
        st.write(f"An error occurred: {e}")
