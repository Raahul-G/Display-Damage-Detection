from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
url = "https://guide-images.cdn.ifixit.com/igi/XCWIlmWmWrODikjU.standard"

image = Image.open(requests.get(url, stream=True).raw)

labels = ["Damaged Screen", "Not Damaged"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

print(labels[probs.argmax().item()])


