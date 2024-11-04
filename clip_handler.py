import torch
import torch.nn as nn
from PIL import Image
from transformers  import CLIPProcessor, CLIPModel

from image_handler import ImageHandler


class ClipHandler:
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    @classmethod
    def generate_visual_embeddings(cls, image):
        i = Image.open(image).convert("RGB")
        inputs = cls.processor(images=i, return_tensors="pt")
        #i.resize((224,224)).save("clip.png")
        #inputs = cls.processor(images=i.resize((224,224)), return_tensors="pt", do_resize=False, do_center_crop=False)
        with torch.no_grad():
            image_embedding = cls.model.get_image_features(**inputs)
        
        del inputs
        torch.cuda.empty_cache()
        return image_embedding