import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
import faiss
import os
import json
import numpy as np

IMAGE_DIR = "/Users/James/Documents"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

def create_index() -> None:
    """
    Calculate image embeddings using ImageBind then save them to a faiss vector store.

    :return: None
    """

    images = [i for i in os.listdir(IMAGE_DIR) if i.endswith(".jpeg") and os.path.isfile(os.path.join(IMAGE_DIR, i))]

    # append dir to each image
    images = [os.path.join(IMAGE_DIR, i) for i in images]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    # Load data
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(images, device),
    }

    with torch.no_grad():
        print("Calculating embeddings...")
        embeddings = model(inputs)

    index = faiss.IndexFlatL2(len(embeddings[ModalityType.VISION][0]))

    # Add each embedding to index
    for embedding in embeddings[ModalityType.VISION]:
        print(embedding)
        # convert to p.array(e[0], dtype="float32").reshape(1, -1)

        index.add(np.array(embedding.cpu().numpy(), dtype="float32").reshape(1, -1))

    # build reference dict
    reference = {i: image for i, image in enumerate(images)}

    faiss.write_index(index, "index.bin")

    with open("reference.json", "w") as f:
        json.dump(reference, f)

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text([query], device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    D, I = index.search(np.array(embeddings[ModalityType.TEXT][0].cpu().numpy(), dtype="float32").reshape(1, -1), k)

    results = []

    for i in I[0]:
        if str(i) in reference:
            results.append(reference[str(i)])

    return results

if __name__ == "__main__":
    # create_index()
    index, reference = load_index()

    while True:
        query = input("Query: ")
        print(search(query, index, reference, k=10))