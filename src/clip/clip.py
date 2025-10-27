import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


class CLIPAnalyzer(object):

    def __init__(
        self,
        images_base_path: Path,
        captions_json_path: Path,
        model_name="openai/clip-vit-base-patch32"
    ):
        self.images_base_path = Path(images_base_path)
        self.captions_json_path = Path(captions_json_path)

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Using device: {self.device}")

    def load_data(self):
        """
        Load image paths and captions from JSON file.
        """
        with open(self.captions_json_path, 'r') as f:
            data = json.load(f)

        self.dataset = []
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
        Extract embedding from image using CLIP.
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.squeeze().cpu().numpy()

        return embedding

    def extract_text_embedding(self, text):
        """
        Extract embedding from text using CLIP.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize the features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embedding = text_features.squeeze().cpu().numpy()

        return embedding

    def compute_all_embeddings(self):
        """
        Compute embeddings for all images and captions using CLIP.
        """
        image_embeddings = []
        text_embeddings = []

        for item in self.dataset:

            img_emb = self.extract_image_embedding(item['image_path'])
            image_embeddings.append(img_emb)

            # Extract text embedding
            txt_emb = self.extract_text_embedding(item['caption'])
            text_embeddings.append(txt_emb)

        self.image_embeddings = np.array(image_embeddings)
        self.text_embeddings = np.array(text_embeddings)

        return self.image_embeddings, self.text_embeddings

    def compute_cross_modal_similarity(self):
        """
        Compute cross-modal similarity matrix between images and texts.
        """
        similarity_matrix = cosine_similarity(self.image_embeddings, self.text_embeddings)

        return similarity_matrix

    def analyze_results(self):
        """
        Analyze CLIP embeddings and matching performance.
        """

        print("CLIP ANALYSIS RESULTS")

        print("\nEmbedding Dimensions:")
        print(f"\tImage (CLIP): {self.image_embeddings.shape[1]}D")
        print(f"\tText (CLIP):  {self.text_embeddings.shape[1]}D")

        # Compute cross-modal similarities
        similarity_matrix = self.compute_cross_modal_similarity()
        
        correct_matches = np.diag(similarity_matrix)
        print(f"\nCorrect Match Similarities (image[i] ↔ caption[i]):")
        print(f"\tMean: {np.mean(correct_matches):.4f}")
        print(f"\tStd:  {np.std(correct_matches):.4f}")
        print(f"\tMin:  {np.min(correct_matches):.4f}")
        print(f"\tMax:  {np.max(correct_matches):.4f}")
        
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        incorrect_matches = similarity_matrix[mask]
        print(f"\nIncorrect Match Similarities (image[i] ↔ caption[j], i≠j):")
        print(f"\t Mean: {np.mean(incorrect_matches):.4f}")
        print(f"\t Std:  {np.std(incorrect_matches):.4f}")
        
        # Calculate matching accuracy
        predictions = np.argmax(similarity_matrix, axis=1)
        correct = np.arange(len(self.dataset))
        accuracy = np.mean(predictions == correct)
        print(f"\nImage→Text Matching Accuracy: {accuracy*100:.2f}%")
        print(f"  ({np.sum(predictions == correct)}/{len(self.dataset)} correct matches)")
        
        # Category-based analysis
        self._analyze_by_category(similarity_matrix)
        
        return {
            'similarity_matrix': similarity_matrix,
            'correct_matches': correct_matches,
            'accuracy': accuracy
        }
    
    def _analyze_by_category(self, similarity_matrix):
        """
        Analyze CLIP performance by category.
        """
        print("CATEGORY-BASED ANALYSIS")

        categories = {}
        for i, item in enumerate(self.dataset):
            cat = item['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(i)
        
        print(f"\nCategories: {list(categories.keys())}")
        
        for cat_name, i in sorted(categories.items()):
            cat_sim = similarity_matrix[i][:, i]
            correct_sim = np.diag(cat_sim)
            
            print(f"\n  📁 {cat_name.upper()} ({len(i)} items):")
            print(f"     Correct match similarity: {np.mean(correct_sim):.4f} " +
                  f"(±{np.std(correct_sim):.4f})")
            
            # Check if all matches are correct
            predictions = np.argmax(cat_sim, axis=1)
            accuracy = np.mean(predictions == np.arange(len(i)))
            print(f"     Category accuracy: {accuracy*100:.1f}%")
        
        self.categories = categories
    
    def visualize_results(self, results, output_dir='outputs'):
        """
        Create visualizations of CLIP analysis results.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        
        # Create cross-modal similarity heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            results['similarity_matrix'],
            cmap='RdYlBu_r',
            square=True,
            cbar_kws={'label': 'Cosine Similarity'},
            vmin=0, vmax=1,
            xticklabels=False,
            yticklabels=False
        )
        plt.title('CLIP Cross-Modal Similarity Matrix\n(Image → Text)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Text Index')
        plt.ylabel('Image Index')
        
        # Highlight the diagonal (correct matches)
        n = len(results['similarity_matrix'])
        for i in range(n):
            plt.plot([i, i+1], [i, i], 'g-', linewidth=2, alpha=0.3)
            plt.plot([i, i+1], [i+1, i+1], 'g-', linewidth=2, alpha=0.3)
            plt.plot([i, i], [i, i+1], 'g-', linewidth=2, alpha=0.3)
            plt.plot([i+1, i+1], [i, i+1], 'g-', linewidth=2, alpha=0.3)
        
        plt.tight_layout()
        similarity_path = output_path / 'clip_similarity_matrix.png'
        plt.savefig(similarity_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {similarity_path}")
        
        # Create category analysis visualization
        self._visualize_by_category(results, output_path)
    
    def _visualize_by_category(self, results, output_path):
        """
        Create visualization of category-wise performance.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cat_names = sorted(self.categories.keys())
        cat_scores = []
        
        similarity_matrix = results['similarity_matrix']
        
        for cat_name in cat_names:
            i = self.categories[cat_name]
            cat_sim = similarity_matrix[i][:, i]
            correct_sim = np.diag(cat_sim)
            cat_scores.append(np.mean(correct_sim))
        
        bars = ax.bar(cat_names, cat_scores, color='mediumseagreen', 
                      edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, score in zip(bars, cat_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Average Correct Match Similarity', fontsize=12)
        ax.set_title('CLIP Performance by Category', fontsize=14, fontweight='bold')
        ax.set_xticklabels(cat_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=np.mean(cat_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(cat_scores):.3f}', linewidth=2)
        ax.legend()
        
        plt.tight_layout()
        category_path = output_path / 'clip_category_analysis.png'
        plt.savefig(category_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {category_path}")
