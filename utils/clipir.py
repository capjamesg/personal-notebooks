import json
import os

import clip
import faiss
import numpy as np
import torch
from PIL import Image

# Instantiate model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)


def create_index(img_dir: str) -> tuple:
    """
    Calculate image embeddings using CLIP then save them to a faiss vector store.

    :return: None
    """

    images = [
        i
        for i in os.listdir(img_dir)
        if i.endswith(".jpg") and os.path.isfile(os.path.join(img_dir, i))
    ]

    # append dir to each image
    images = [os.path.join(img_dir, i) for i in images]

    embeddings = []

    with torch.no_grad():
        for i in images:
            image = preprocess(Image.open(i)).unsqueeze(0).to(device)
            image_features = model.encode_image(image).float()
            embeddings.append(image_features)

    index = faiss.IndexFlatL2(len(embeddings[0][0]))

    for embedding in embeddings:
        index.add(np.array(embedding.cpu().numpy(), dtype="float32").reshape(1, -1))

    # build reference dict
    reference = {i: image for i, image in enumerate(images)}

    faiss.write_index(index, "index.bin")

    with open("reference.json", "w") as f:
        json.dump(reference, f)

    return index, reference


def load_index() -> tuple:
    """
    Load the faiss index and reference dict that maps faiss index numbers to image file names.

    :return: A tuple containing the faiss index and reference dict
    """
    index = faiss.read_index("index.bin")

    with open("reference.json", "r") as f:
        reference = json.load(f)

    return index, reference


def search(query, index, reference, k=10) -> list:
    """
    Search the faiss index for the top k most similar images to the query text.

    :param query: The query image
    :param index: The faiss index

    :return: A list of the top k most similar images
    """
    text_query = model.encode_text(clip.tokenize(query).to(device))

    _, I = index.search(
        np.array(text_query.cpu().numpy(), dtype="float32").reshape(1, -1), k
    )

    results = []

    for i in I[0]:
        if str(i) in reference:
            results.append(reference[str(i)])

    return results


if __name__ == "__main__":
    # create_index("/Users/james/src/machine-learning/Mug-Detector-6/train")
    index, reference = load_index("")

    while True:
        query = input("Query: ")
        print(search(query, index, reference, k=10))

