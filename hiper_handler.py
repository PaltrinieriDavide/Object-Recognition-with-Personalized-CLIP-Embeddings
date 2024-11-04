import torch
import torch.nn as nn
from PIL import Image
from transformers  import CLIPProcessor, CLIPModel
import subprocess

from image_handler import ImageHandler


class HiPerHandler:
    
    path_train = "HiPer/train.py"
    pretrained_model_name = "CompVis/stable-diffusion-v1-4"
    emb_learning_rate = "5e-3"
    seed = 200000
    
    @classmethod
    def generate_embeddings(cls, image, text, output_dir, emb_train_steps = 5000, n_hiper = 5):
        command = [
            "python", cls.path_train,
            "--pretrained_model_name", cls.pretrained_model_name,
            "--input_image", image,
            "--target_text", text,
            "--source_text", text,
            "--output_dir", output_dir,
            "--n_hiper", str(n_hiper),
            "--emb_learning_rate", cls.emb_learning_rate,
            "--emb_train_steps", str(emb_train_steps),
            "--seed", str(cls.seed),
            "--no_pipe"
            ]
        process = subprocess.Popen(command, text=True)
        process.wait()
        if process.returncode == 0:
            print('\n' + '#' * 80 + '\n')
            print("\t\tTraining on " + image + " completed")
            print('\n' + '#' * 80 + '\n')