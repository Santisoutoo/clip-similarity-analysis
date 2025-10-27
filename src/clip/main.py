from pathlib import Path
from clip import CLIPAnalyzer


def main():
    # src/clip/
    script_dir = Path(__file__).parent
    # proyect/
    project_root = script_dir.parent.parent
    
    IMAGES_PATH = project_root / "img"
    CAPTIONS_PATH = project_root / "img" / "img.json"
    OUTPUT_DIR = project_root / "outputs"
    
    analyzer = CLIPAnalyzer(IMAGES_PATH, CAPTIONS_PATH)
    
    analyzer.load_data()
    
    analyzer.compute_all_embeddings()
    
    results = analyzer.analyze_results()
    
    analyzer.visualize_results(results, OUTPUT_DIR)


if __name__ == "__main__":
    print("CLIP ANALYSIS")
    main()