import torch
import torch.nn.functional as F
import numpy as np
from transformers  import CLIPProcessor, CLIPModel


class EmbeddingsAnalyzer:

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    @staticmethod
    def calculate_similarity(embedding1, embedding2):
        return torch.matmul(embedding1, embedding2)

    @staticmethod
    def normalize(embedding, dim = 1):
        return F.normalize(embedding, p=2, dim = dim)
    
    @classmethod
    def project_embeddings(cls, embeddings):
        with torch.no_grad():
            return cls.model.text_projection(embeddings)