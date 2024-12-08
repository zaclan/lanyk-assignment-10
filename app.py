from flask import Flask, request, render_template
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
from sklearn.metrics.pairwise import euclidean_distances

# Load model and tokenizer
model_name = "ViT-B-32"
pretrained = "openai"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()
tokenizer = get_tokenizer(model_name)

# Load CLIP embeddings DataFrame
df = pd.read_pickle('image_embeddings.pickle')
embeddings_tensor = torch.tensor(np.stack(df['embedding'].values)).to(device)  # shape: [num_images, emb_dim]
image_folder = "coco_images_resized"

# Load PCA components
pca_components = np.load('pca_components.npy')  # shape: [emb_dim, k]
pca_mean = np.load('pca_mean.npy')              # shape: [emb_dim]
pca_embeddings = np.load('pca_embeddings.npy')  # shape: [num_images, k]

def get_clip_image_embedding(img: Image.Image):
    img_t = preprocess_val(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_t)
    emb = F.normalize(emb, p=2, dim=1)
    return emb

def get_clip_text_embedding(text: str):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
    emb = F.normalize(emb, p=2, dim=1)
    return emb

def hybrid_embedding(text_emb, image_emb, lam):
    combined = lam * text_emb + (1 - lam) * image_emb
    combined = F.normalize(combined, p=2, dim=1)
    return combined

def search_with_clip(query_emb, k=5):
    # Use cosine similarity for CLIP embeddings
    sims = F.cosine_similarity(query_emb, embeddings_tensor, dim=1)
    topk = torch.topk(sims, k)
    indices = topk.indices.cpu().tolist()
    scores = topk.values.cpu().tolist()
    result_files = df.iloc[indices]['file_name'].values
    return list(zip(result_files, scores))

def search_with_pca(query_emb, k=5):
    # query_emb: [1, emb_dim], use Euclidean distance in PCA space
    q = query_emb.cpu().numpy() - pca_mean
    q_pca = q.dot(pca_components)  # shape: [1, k]

    distances = euclidean_distances(q_pca, pca_embeddings).flatten()  # shape: [num_images]
    nearest_indices = np.argsort(distances)[:k]
    top_files = df.iloc[nearest_indices]['file_name'].values
    top_distances = distances[nearest_indices].tolist()
    return list(zip(top_files, top_distances))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        query_type = request.form.get('query_type', 'text')  # text, image, or hybrid
        text_query = request.form.get('text_query', '').strip()
        lam = request.form.get('lambda', '0.5')
        lam = float(lam) if lam else 0.5
        use_pca = request.form.get('use_pca', 'off') == 'on'
        image_file = request.files.get('image_query', None)

        text_query_emb = None
        image_query_emb = None

        # Validate query inputs based on query_type
        if query_type == 'text':
            # Expect text only
            if text_query:
                text_query_emb = get_clip_text_embedding(text_query)
            # If no text given, no results
        elif query_type == 'image':
            # Expect image only
            if image_file and image_file.filename != '':
                img = Image.open(image_file).convert("RGB")
                image_query_emb = get_clip_image_embedding(img)
            # If no image given, no results
        elif query_type == 'hybrid':
            # Expect both text and image
            if text_query and image_file and image_file.filename != '':
                text_query_emb = get_clip_text_embedding(text_query)
                img = Image.open(image_file).convert("RGB")
                image_query_emb = get_clip_image_embedding(img)
            # If missing either text or image, no results

        # Now form the final query embedding based on query_type
        query_emb = None
        if query_type == 'text' and text_query_emb is not None:
            query_emb = text_query_emb
        elif query_type == 'image' and image_query_emb is not None:
            query_emb = image_query_emb
        elif query_type == 'hybrid' and text_query_emb is not None and image_query_emb is not None:
            query_emb = hybrid_embedding(text_query_emb, image_query_emb, lam)

        # Only if we have a valid query embedding do we search
        if query_emb is not None:
            # If query_type is image-only and use_pca is selected -> PCA embeddings
            if query_type == 'image' and use_pca:
                # Use PCA embeddings with Euclidean distance
                results = search_with_pca(query_emb, k=5)
            else:
                # Use CLIP embeddings with cosine similarity
                results = search_with_clip(query_emb, k=5)

    return render_template('index.html', results=results, image_folder=image_folder)

if __name__ == "__main__":
    app.run(debug=True)