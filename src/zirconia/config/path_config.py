import os.path
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

RESULTS_DIR=PROJECT_ROOT / "results"
IMAGE_DIR = RESULTS_DIR / "images"
MODEL_DIR = RESULTS_DIR / "checkpoint"

## 01_train_physics_model.py
BEST_PIML_MODEL_PATH=os.path.join(MODEL_DIR,"piml","best_piml_model.pth")
PIML_PREDICTION_EA_DISTANCE_IMAGE_PATH=os.path.join(IMAGE_DIR,"piml_prediction_and_ea_dist.png")
#02_interpret_mechanisms.py,三张图，还有一张paper_lodo_{target_element}需要用到代码中的变量
LATENT_SPACE_IMAGE_PATH=os.path.join(IMAGE_DIR,"paper_latent_space.png")
PAPER_FEATURE_IMPORTANCE_EA_IMAGE_PATH=os.path.join(IMAGE_DIR,"paper_feature_importance_Ea.png")
#03a_train_baseline_model.py
BASELINE_MODEL_PATH=os.path.join(MODEL_DIR,"baseline","best_baseline_dnn.pth")
BASELINE_DNN_RESULT_IMAGE_PATH = os.path.join(IMAGE_DIR,"baseline_dnn_prediction_vs_actual.png")
COMPARISON_BASELINE_DNN_CSV=os.path.join(RESULTS_DIR,"comparison_baseline_dnn.csv")
#03b_evaluate_benchmarks.py
EA_VS_STRUCTURE_AND_DOPING_IMAGE_PATH=os.path.join(IMAGE_DIR,"Ea_vs_structure_and_doping.png")
VIRTUAL_SCREENING_RESULTS_CSV=os.path.join(RESULTS_DIR,"virtual_screening_results.csv")
FINAL_METRICS_COMPARISON_CSV=os.path.join(RESULTS_DIR,"final_metrics_comparison.csv")
#04_discover_materials.py
PAPER_STRAIN_THEORY_VERIFICATION_IMAGE_PATH=os.path.join(IMAGE_DIR,"paper_strain_theory_verification.png")
PAPER_INVERSE_DESIGN_GA_IMAGE_PATH=os.path.join(IMAGE_DIR,"paper_inverse_design_ga.png")
#05_simulate_lab_experiments.py
UQ_CALIBRATION_IMAGE_PATH = os.path.join(IMAGE_DIR,"paper_uq_calibration.png")
ACTIVE_LEARNING_IMAGE_PATH = os.path.join(IMAGE_DIR,"paper_active_learning.png")