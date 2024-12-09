import pandas as pd
from IPython.display import Image
import numpy as np
# import torch
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import open_clip
from sklearn.decomposition import PCA

df = pd.read_pickle('image_embeddings.pickle')
df 

model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()



def text_to_image(text):
    # tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text = tokenizer([text])
    query_embedding = F.normalize(model.encode_text(text))

    # Retrieve the image path that corresponds to the embedding in `df`
    # with the highest cosine similarity to query_embedding
    df_embeddings = np.stack(df['embedding'].values)

    # Convert the query embedding to a numpy array for cosine similarity calculation
    query_embedding_np = query_embedding.detach().numpy()

    # Calculate cosine similarities
    cosine_similarities = np.dot(df_embeddings, query_embedding_np.T).flatten()

    # Find the indices of the top 5 most similar images
    top_5_indices = cosine_similarities.argsort()[-5:][::-1]

    # Retrieve the corresponding image paths and similarity scores
    top_5_paths = [
        "static/coco_images_resized/" + df.iloc[idx]['file_name'] 
        for idx in top_5_indices
    ]
    top_5_scores = cosine_similarities[top_5_indices]

    return list(zip(top_5_paths, top_5_scores))

def image_to_image(image_path):
    # This converts the image to a tensor
    image = preprocess(Image.open(image_path)).unsqueeze(0)

    # This calculates the query embedding
    query_embedding = F.normalize(model.encode_image(image))

    # Convert the DataFrame embeddings to a numpy array
    df_embeddings = np.stack(df['embedding'].values)

    # Convert the query embedding to a numpy array for cosine similarity calculation
    query_embedding_np = query_embedding.detach().numpy()

    # Calculate cosine similarities
    cosine_similarities = np.dot(df_embeddings, query_embedding_np.T).flatten()

    # Find the indices of the top 5 most similar images
    top_5_indices = cosine_similarities.argsort()[-5:][::-1]

    # Retrieve the corresponding image paths and similarity scores
    top_5_paths = [
        "static/coco_images_resized/" + df.iloc[idx]['file_name'] 
        for idx in top_5_indices
    ]
    top_5_scores = cosine_similarities[top_5_indices]

    return list(zip(top_5_paths, top_5_scores))

def hybrid_query(image_path, text_input):
    # tokenizer = open_clip.get_tokenizer('ViT-B-32')
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    image_query = F.normalize(model.encode_image(image))
    text = tokenizer([text_input])
    text_query = F.normalize(model.encode_text(text))

    lam  = 0.8 # tune this

    query = F.normalize(lam * text_query + (1.0 - lam) * image_query)

    # Convert the DataFrame embeddings to a numpy array
    df_embeddings = np.stack(df['embedding'].values)

    # Convert the combined query embedding to a numpy array
    query_np = query.detach().numpy()

    # Calculate cosine similarities
    cosine_similarities = np.dot(df_embeddings, query_np.T).flatten()

    # Find the indices of the top 5 most similar images
    top_5_indices = cosine_similarities.argsort()[-5:][::-1]

    # Retrieve the corresponding image paths and similarity scores
    top_5_paths = [
        "static/coco_images_resized/" + df.iloc[idx]['file_name'] 
        for idx in top_5_indices
    ]
    top_5_scores = cosine_similarities[top_5_indices]

    return list(zip(top_5_paths, top_5_scores))

def apply_pca(k):
    # Assuming embeddings are in the DataFrame 'df'
    embeddings = np.stack(df['embedding'].values)  
    pca = PCA(n_components=k)
    reduced_embeddings = pca.fit_transform(embeddings)  # Reduce embeddings using PCA
    return pca, reduced_embeddings

def pca_image_to_image(image_path, k):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    image_query = F.normalize(model.encode_image(image))

    # Apply PCA to the embeddings dataset and get the reduced embeddings
    pca, reduced_embeddings = apply_pca(k)  # Apply PCA and get reduced embeddings
    
    query_np = image_query.detach().numpy()

    # Project the image query embedding onto the PCA space
    pca_query = pca.transform(query_np.reshape(1, -1))  # Reshape to 2D for PCA

    # Calculate cosine similarities between the PCA query and the embeddings
    cosine_similarities = np.dot(reduced_embeddings, pca_query.T).flatten()

    # Find the indices of the top 5 most similar images
    top_5_indices = cosine_similarities.argsort()[-5:][::-1]

    # Retrieve the corresponding image paths and similarity scores
    top_5_paths = [
        "static/coco_images_resized/" + df.iloc[idx]['file_name']
        for idx in top_5_indices
    ]
    top_5_scores = cosine_similarities[top_5_indices]

    return list(zip(top_5_paths, top_5_scores))