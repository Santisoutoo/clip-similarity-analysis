import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


class PreCLIPAnalyzer(object):

    def __init__(
        self,
        images_base_path: Path,
        captions_json_path: Path
    ):
        self.images_base_path = Path(images_base_path)
        self.captions_json_path = Path(captions_json_path)
        self.image_model = self._load_resnet()
        self.text_model = SentenceTransformer(
            'sentence-transformers/distiluse-base-multilingual-cased'
        )
        # Image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_model.to(self.device)
        print(f"Using: {self.device}")
        
    def _load_resnet(self):
        """
        Load ResNet-50 and remove classification head.
        """
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer to get feature embeddings
        modules = list(resnet.children())[:-1]
        resnet = nn.Sequential(*modules)
        resnet.eval()
        return resnet
    
    def load_data(self):
        """Load image paths and captions"""
        
        with open(self.captions_json_path, 'r') as file:
            data = json.load(file)
        
        self.dataset: list = []
        for category, items in data.items():
            for item in items:
                
                image_path = self.images_base_path / category / item['filename']
                
                if image_path.exists():
                    self.dataset.append({
                        'category': category,
                        'filename': item['filename'],
                        'image_path': str(image_path),
                        'caption': item['description']
                    })
                else:
                    print(f"Image not found: {image_path}")
                    
        return self.dataset
    
    def extract_image_embedding(self, image_path):
        """
        Extract embedding from image.
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.image_model(image_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    
    def extract_text_embedding(self, text):
        """
        Extract embedding from text using sentence-transformers.
        """
        embedding = self.text_model.encode(text)
        return embedding

    def compute_all_embeddings(self):
        """
        Compute embeddings for all images and captions in the dataset.
        """
        image_embeddings = []
        text_embeddings = []

        for item in self.dataset:

            # Extract image embedding
            img_emb = self.extract_image_embedding(item['image_path'])
            image_embeddings.append(img_emb)

            # Extract text embedding
            txt_emb = self.extract_text_embedding(item['caption'])
            text_embeddings.append(txt_emb)

        self.image_embeddings = np.array(image_embeddings)
        self.text_embeddings = np.array(text_embeddings)

        return self.image_embeddings, self.text_embeddings

    def analyze_results(self):

        print("\nPRE-CLIP ANALYSIS RESULTS")

        print("\nEmbedding Dimensions:")
        print(f"\tImage: {self.image_embeddings.shape[1]}D")
        print(f"\tText: {self.text_embeddings.shape[1]}D\n")
        
        print("CRITICAL LIMITATION: Different embedding dimensions!")
        print(f"\tImage space: {self.image_embeddings.shape[1]}-dimensional")
        print(f"\tText space:  {self.text_embeddings.shape[1]}-dimensional")
        

        print("\nWITHIN-MODALITY SIMILARITY ANALYSIS")
        img_similarities = cosine_similarity(self.image_embeddings)
        
        print(f"\nImage-to-Image Similarities:")
        print(f"\tMean: {np.mean(img_similarities):.3f}")
        print(f"\tStd:  {np.std(img_similarities):.3f}")
        print(f"\tMin:  {np.min(img_similarities):.3f}")
        print(f"\tMax:  {np.max(img_similarities):.3f}")
        
        # Text-to-text similarities
        txt_similarities = cosine_similarity(self.text_embeddings)
        print(f"\nText-to-Text Similarities:")
        print(f"\tMean: {np.mean(txt_similarities):.3f}")
        print(f"\tStd:  {np.std(txt_similarities):.3f}")
        print(f"\tMin:  {np.min(txt_similarities):.3f}")
        print(f"\tMax:  {np.max(txt_similarities):.3f}")
        

        print("\nCATEGORY-BASED ANALYSIS")

        # Agrupar índices por categoría
        categories = {}
        for i, item in enumerate(self.dataset):
            cat = item['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(i)

        print(f"\nCategories: {list(categories.keys())}")

        # Analizar cada categoría
        for cat_name, i in sorted(categories.items()):
            
            cat_img_embs = self.image_embeddings[i]
            cat_txt_embs = self.text_embeddings[i]
            
            img_sim = cosine_similarity(cat_img_embs)
            txt_sim = cosine_similarity(cat_txt_embs)
            
            # get matrix values without the diagonal
            n = len(i)
            img_values = img_sim[~np.eye(n, dtype=bool)]
            txt_values = txt_sim[~np.eye(n, dtype=bool)]
            
            print(f"\n{cat_name.upper()} ({len(i)} items):")
            print(f"\tImage similarity: {img_values.mean():.3f} (±{img_values.std():.3f})")
            print(f"\tText similarity:  {txt_values.mean():.3f} (±{txt_values.std():.3f})")
        
        return {
            'img_similarities': img_similarities,
            'txt_similarities': txt_similarities,
            'categories': categories
        }
    
    def visualize_results(self, results, output_dir='outputs'):
        """
        Create visualizations of similarity matrices and category analysis.
        
        Args:
            results: Dictionary containing analysis results
            output_dir: Directory to save visualization files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\nGENERATING VISUALIZATIONS")
        
        # Create heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Image similarity heatmap
        sns.heatmap(
            results['img_similarities'], 
            ax=axes[0],
            cmap='RdYlBu_r',
            square=True,
            cbar_kws={'label': 'Cosine Similarity'},
            vmin=0, vmax=1
        )
        axes[0].set_title(
            'Image-to-Image Similarity (ResNet-50)',
            fontsize=14,
            fontweight='bold'
        )
        axes[0].set_xlabel('Image Index')
        axes[0].set_ylabel('Image Index')
        
        # Text similarity heatmap
        sns.heatmap(
            results['txt_similarities'],
            ax=axes[1], 
            cmap='RdYlBu_r',
            square=True,
            cbar_kws={'label': 'Cosine Similarity'},
            vmin=0, vmax=1
        )
        axes[1].set_title(
            'Text-to-Text Similarity (Sentence Transformer)',
            fontsize=14,
            fontweight='bold'
        )
        axes[1].set_xlabel('Caption Index')
        axes[1].set_ylabel('Caption Index')
        
        plt.tight_layout()
        heatmap_path = output_path / 'preclip_similarity_heatmaps.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved on: {heatmap_path}")
        
        # Create category-based visualization
        self._visualize_by_category(results, output_path)
        
    def _visualize_by_category(self, results, output_path):
        """
        Create visualization of within-category similarities.
        
        Args:
            results: Dictionary containing analysis results
            output_path: Path object for output directory
        """
        categories = results['categories']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        cat_names = sorted(categories.keys())
        img_within_cat = []
        txt_within_cat = []
        
        for cat_name in cat_names:
            i = categories[cat_name]
            if len(i) > 1:
                cat_img_embs = self.image_embeddings[i]
                cat_txt_embs = self.text_embeddings[i]
                
                img_sim = cosine_similarity(cat_img_embs)
                txt_sim = cosine_similarity(cat_txt_embs)
                
                mask = ~np.eye(img_sim.shape[0], dtype=bool)
                img_sim_no_diag = img_sim[mask]
                txt_sim_no_diag = txt_sim[mask]
                
                img_within_cat.append(np.mean(img_sim_no_diag))
                txt_within_cat.append(np.mean(txt_sim_no_diag))
        
        x = np.arange(len(cat_names))
        width = 0.35
        
        axes[0].bar(x, img_within_cat, width, label='Image Similarity',
                    color='steelblue', edgecolor='black', linewidth=1.2)
        axes[0].set_xlabel('Category', fontsize=12)
        axes[0].set_ylabel('Average Cosine Similarity', fontsize=12)
        axes[0].set_title('Within-Category Image Similarity',
                        fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cat_names, rotation=45, ha='right')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        
        axes[1].bar(x, txt_within_cat, width, label='Text Similarity',
                    color='coral', edgecolor='black', linewidth=1.2)
        axes[1].set_xlabel('Category', fontsize=12)
        axes[1].set_ylabel('Average Cosine Similarity', fontsize=12)
        axes[1].set_title('Within-Category Text Similarity',
                        fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cat_names, rotation=45, ha='right')
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        category_path = output_path / 'preclip_category_analysis.png'
        plt.savefig(category_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved on: {category_path}")
