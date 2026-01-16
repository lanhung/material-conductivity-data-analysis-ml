import os
from config import path_config
from config import path_config
if __name__ == '__main__':
    os.makedirs(path_config.RESULTS_DIR,exist_ok=True)
    os.makedirs(path_config.IMAGE_DIR,exist_ok=True)
    os.makedirs(path_config.MODEL_DIR,exist_ok=True)
    os.makedirs(path_config.IMAGE_DIR, exist_ok=True)
    piml_model_dir=os.path.join(path_config.MODEL_DIR,"piml")
    os.makedirs(piml_model_dir, exist_ok=True)
    baseline_model_dir=os.path.join(path_config.MODEL_DIR,"baseline")
    os.makedirs(baseline_model_dir, exist_ok=True)
