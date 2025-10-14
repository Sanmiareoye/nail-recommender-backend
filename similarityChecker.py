from sentence_transformers import SentenceTransformer
import os
from PIL import Image
from glob import glob
import numpy as np

model = SentenceTransformer("clip-ViT-B-32")


def generate_clip_embeddings(images_path, model):

    image_extensions = ["jpg", "jpeg", "png", "webp"]

    image_paths = []
    for ext in image_extensions:
        pattern = os.path.join(images_path, f"**/*.{ext}")
        image_paths.extend(glob(pattern, recursive=True))

    embeddings = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        embedding = model.encode(image)
        embeddings.append(embedding)

    return embeddings, image_paths


def search_similar_by_image(query, embeddings, image_paths, top_k):
    if not embeddings or not image_paths:
        return []

    embedding_matrix = np.array(embeddings).astype(np.float32)

    embedding_matrix = embedding_matrix / np.linalg.norm(
        embedding_matrix, axis=1, keepdims=True
    )

    query_image = Image.open(query).convert("RGB")

    query_embedding = model.encode(query_image)

    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    similarities = np.dot(embedding_matrix, query_embedding)

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append(
            {"image_path": image_paths[idx], "similarity": float(similarities[idx])}
        )
    return results


IMAGES_PATH = "nail_dataset"

embeddings, image_paths = generate_clip_embeddings(IMAGES_PATH, model)
