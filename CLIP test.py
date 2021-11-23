from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)


def loadImages(path):

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = Image.open(path + image)
        loadedImages.append(img)

    return loadedImages

path = "C:/Users/echapman/Desktop/Snafler/pixelImages/"


imgs = loadImages(path)

score =[]
for img in imgs:
    
    inputs = processor(text=["a red octopus. #pixelart"], images=img, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    score.append(logits_per_image.detach().numpy()[0])


i = np.arange(0,5,1)
print(i)
plt.plot(i,score)
plt.show()