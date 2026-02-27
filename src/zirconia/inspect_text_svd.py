
import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to sys.path to make imports work
sys.path.append(os.getcwd())

try:
    from config import path_config
    from etl.material_data_processor import MaterialDataProcessor
    from features.preprocessor import build_feature_pipeline
except ImportError:
    # If running from src/zirconia directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import path_config
    from etl.material_data_processor import MaterialDataProcessor
    from features.preprocessor import build_feature_pipeline

def inspect_text_components():
    print("Loading data...")
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()
    
    print(f"Data loaded: {len(df)} rows")

    # Set seed to match the main experiment (02_interpret_mechanisms.py)
    # This ensures the SVD components are identical to those in the paper/report.
    np.random.seed(42)

    # Build and fit pipeline
    print("Fitting feature pipeline...")
    pipeline = build_feature_pipeline()
    pipeline.fit(df)

    # Extract text pipeline
    # The pipeline is a ColumnTransformer. The text part is named 'text'.
    text_pipe = pipeline.named_transformers_['text']
    try:
        tfidf = text_pipe.named_steps['tfidf']
        svd = text_pipe.named_steps['svd']
    except KeyError:
        print("Error accessing pipeline steps. Dumping steps:", text_pipe.named_steps.keys())
        return

    feature_names = tfidf.get_feature_names_out()

    # Output path
    output_path = os.path.join(path_config.RESULTS_DIR, "text_svd_interpretation.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("=== Interpretation of Text SVD Components (Topic Analysis) ===")
    lines.append("Each component represents a latent theme in the material descriptions.")
    
    for i, comp in enumerate(svd.components_):
        # Get indices sorted by value
        sorted_indices = np.argsort(comp)
        
        # Top positive words (largest positive values) - end of the array
        top_pos_indices = sorted_indices[-8:][::-1]
        pos_words = [f"{feature_names[j]}({comp[j]:.2f})" for j in top_pos_indices if comp[j] > 0.05]
        
        # Top negative words (largest negative values, if any) - start of the array
        top_neg_indices = sorted_indices[:8]
        neg_words = [f"{feature_names[j]}({comp[j]:.2f})" for j in top_neg_indices if comp[j] < -0.05]
        
        if not pos_words and not neg_words:
            continue
            
        lines.append(f"\n[text_svd_{i}]")
        if pos_words:
            lines.append(f"  (+) Associated with: {', '.join(pos_words)}")
        if neg_words:
            lines.append(f"  (-) Associated with: {', '.join(neg_words)}")

    # Output to both console and file
    output_text = "\n".join(lines)
    print(output_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")
    print(f"\n-> Results saved to '{output_path}'")

if __name__ == "__main__":
    inspect_text_components()
