import torch
import pandas as pd
import os
import random
from PIL import Image, ImageOps
import torch.nn.functional as F
import argparse

from PIL import Image
from transformers import AutoImageProcessor
from transformers import AutoTokenizer

from hiper_handler import HiPerHandler
from clip_handler import ClipHandler
from embeddings_analyzer import EmbeddingsAnalyzer

class DatasetEmbeddingsGenerator:
    
    @staticmethod
    def generate_clip_embeddings(path_folder_dataset, output_dir):
        processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip = EmbeddingsAnalyzer.model
        clip.eval()

        cls_folder_list = [item.path for item in os.scandir(path_folder_dataset) if item.is_dir() and ('1' in item.name or '0' in item.name)]

        ret = []
        with torch.no_grad():
            for path in cls_folder_list:
                print("\n\nActual directory: " + path)
                cls_ret = []
                for item in os.scandir(path):
                    try:
                        img = Image.open(item.path).convert('RGB').resize((224, 224), Image.BILINEAR)
                        inputs = processor(img, return_tensors='pt', do_center_crop=False, do_rescale=True, do_resize=True)
                        inputs = {key: value for key, value in inputs.items()}
                        clip_feat = clip.get_image_features(**inputs).float()
                    except Exception as e:
                        print(f"Error processing {item.path}: {e}")
                        continue
                    cls_ret.append(clip_feat)
                    print("Image's embeddings --  > " + item.name)
                ret.append(torch.cat(cls_ret))

            print("CLIP GENERATED EMBEDDINGS:\n")
            print(ret)
            return ret
                    
    @staticmethod
    def generate_hiper_embeddings(path_folder_dataset, output_dir, text, num_images, train_steps, n_hiper):
        for item in os.scandir(path_folder_dataset):
            if item.is_dir():
                if '1' in item.name:
                    dataset_path = item.path
        images = []
        for item in os.scandir(dataset_path):
            images.append(item.path)
        if len(images) < num_images:
            num_images = len(images)
        selected_images = random.sample(images, num_images)
        
        for i in selected_images:
            i = "/homes/dpaltrinieri/sdod/dataset_target_blue/splitted/train/backpack_1/blue_15.png"
            HiPerHandler.generate_embeddings(i, text, output_dir, train_steps, n_hiper=n_hiper)
            
    @staticmethod
    def calculate_similarities(path_hiper, clip_feat, PROMPT):
        hiper_embs = []
        for hiper_file in os.scandir(path_hiper):
            emb = torch.load(hiper_file.path)
            emb = emb.squeeze(dim = 0)
            hiper_embs.append(emb)
        
        hiper_tensors = torch.stack(hiper_embs, dim=0)
        hiper_tensors = torch.mean(hiper_tensors, dim=0)
        hiper_tensors = hiper_tensors.squeeze(0)
        
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        with torch.no_grad():
            text_features = EmbeddingsAnalyzer.model.text_model(**tokenizer([PROMPT], padding='max_length', return_tensors='pt')).last_hidden_state
            text_features[:, -hiper_tensors.size(0):] = hiper_tensors
            text_features = EmbeddingsAnalyzer.model.text_projection(text_features).squeeze(0)
            text_features = F.normalize(text_features, p=2, dim=-1)

        print(hiper_tensors.shape)
        
        #hiper_tensors = hiper_tensors.transpose(0, 1)
                
        data = []
        for cls, feat in zip([0, 1], clip_feat):
            feat = F.normalize(feat, p=2, dim=-1)
            sim = feat @ text_features.T
            sim = torch.cat([sim, torch.tensor([[cls]]).expand(sim.size(0), 1)], dim=1)
            data.append(sim)
        
        return torch.cat(data, dim=0)
        
        
    @staticmethod
    def generate_similarities_dataset(output_file, column_names, tensors_list):
        data = [tensor.flatten().tolist() for tensor in tensors_list]
        df = pd.DataFrame(data, columns=column_names)
        print(f"Class 0:\n {df[df.y == 0].mean()}")
        print(f"Class 1:\n {df[df.y == 1].mean()}")
        df.to_csv(output_file, index=False)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_train', help="Training dataset")
    parser.add_argument('path_test', nargs='?', help="Testing dataset", default=None)
    return parser.parse_args()

def main():
    
    text = "A photo of a blue backpack"
    num_images = 1
    train_steps = 5000
    PROMPT = "A photo of a blue backpack" #define the target prompt
    
    args = parse_command_line_args()
    
    
    if args.path_test is None:
        images_dataset_path = args.path_train
        
        clip_embs_output_path = "data/joined/backpack_clip_embeddings"
        hiper_embs_output_path = "data/joined/backpack_hiper_embeddings_train"
        
        clip_feat = DatasetEmbeddingsGenerator.generate_clip_embeddings(images_dataset_path, clip_embs_output_path)
        
        DatasetEmbeddingsGenerator.generate_hiper_embeddings(images_dataset_path, hiper_embs_output_path, text, num_images, train_steps)
        
        data = DatasetEmbeddingsGenerator.calculate_similarities(hiper_embs_output_path, clip_feat, PROMPT)
        
        output_file = "data/joined/similarities_dataset.csv"
        DatasetEmbeddingsGenerator.generate_similarities_dataset(output_file, column_names, data)
    else:
        images_dataset_path_train = args.path_train
        images_dataset_path_test = args.path_test
        
        hiper_embs_output_path = "data/splitted/backpack_hiper_embeddings_train"
        # DatasetEmbeddingsGenerator.generate_hiper_embeddings(images_dataset_path_train, hiper_embs_output_path, text, num_images, train_steps, 5)
        
        clip_embs_output_train = "data/splitted/backpack_clip_embeddings_train"
        clip_embs_output_test = "data/splitted/backpack_clip_embeddings_test"
        
        clip_feat_train = DatasetEmbeddingsGenerator.generate_clip_embeddings(images_dataset_path_train, clip_embs_output_train)
        clip_feat_test = DatasetEmbeddingsGenerator.generate_clip_embeddings(images_dataset_path_test, clip_embs_output_test)
        
        data = DatasetEmbeddingsGenerator.calculate_similarities(hiper_embs_output_path, clip_feat_train, PROMPT)
        
        column_names = ['sim' + str(i) for i in range(data.size(1))]
        column_names[-1] = 'y'
        
        output_file = "data/splitted/similarities_datset_train.csv"
        DatasetEmbeddingsGenerator.generate_similarities_dataset(output_file, column_names, data)
        
        data = DatasetEmbeddingsGenerator.calculate_similarities(hiper_embs_output_path, clip_feat_test, PROMPT)

        output_file = "data/splitted/similarities_datset_test.csv"
        DatasetEmbeddingsGenerator.generate_similarities_dataset(output_file, column_names, data)

if __name__ == "__main__":
    main()