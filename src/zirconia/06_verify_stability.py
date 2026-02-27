import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import path_config

# Extended ionic radii table (Shannon radii, pm, 8-coordination)
IONIC_RADII = {
    'Zr': 84.0, 'O': 138.0,
    'Sc': 87.0, 'Yb': 98.5, 'Y': 101.9, 'Gd': 105.3,
    'Sm': 107.9, 'Nd': 110.9, 'Ca': 112.0, 'Mg': 89.0,
    'Ce': 97.0, 'Ti': 74.0, 'Al': 54.0
}

# [Baseline] Theoretical parameters for 8 at% Y (on the cation sublattice)
# Note: Zr(0.92)+Y(0.08) means Y is 8% on the cation sublattice,
#       not the commonly cited "8 mol% Y2O3-ZrO2" (where Y cations are ~14.8%).
# Zr(0.92) + Y(0.08)
YSZ_8_RADIUS = 0.92 * 84.0 + 0.08 * 101.9 # ~ 85.43 pm
YSZ_8_RATIO  = YSZ_8_RADIUS / 138.0       # ~ 0.619

def load_best_recipe_from_csv():
    csv_path = path_config.AI_DISCOVERY_RESULTS_CSV
    if not os.path.exists(csv_path):
        print(f"Error: Discovery result not found at {csv_path}")
        sys.exit(1)

    print(f">>> [Setup] Loading best recipe from: {csv_path}")
    df = pd.read_csv(csv_path)
    best_row = df.iloc[0]

    dopants = []
    if pd.notna(best_row.get('dopant_1_element')):
        dopants.append((best_row['dopant_1_element'], float(best_row['dopant_1_fraction'])))
    if pd.notna(best_row.get('dopant_2_element')):
        dopants.append((best_row['dopant_2_element'], float(best_row['dopant_2_fraction'])))

    candidate = {
        "dopants": dopants,
        "host_element": "Zr",
        "host_radius": IONIC_RADII['Zr'],
        "oxygen_radius": IONIC_RADII['O']
    }
    print(f"    Loaded System: {candidate['dopants']}")
    return candidate

def calculate_stability_metrics(candidate):
    print(f"\n>>> [Validation] Checking Thermodynamic Stability...")
    print(f"    (Baseline Reference: 8YSZ Radius={YSZ_8_RADIUS:.2f} pm, Ratio={YSZ_8_RATIO:.4f})")

    # 1. Compute average cation radius
    total_conc = sum([x[1] for x in candidate['dopants']])
    host_conc = 1.0 - total_conc
    r_avg = host_conc * IONIC_RADII[candidate['host_element']]
    for el, conc in candidate['dopants']:
        r_avg += conc * IONIC_RADII.get(el, 85.0)

    print(f"    Average Cation Radius: {r_avg:.2f} pm")

    # 2. Cation radius mismatch (radius mismatch)
    variance = host_conc * (IONIC_RADII[candidate['host_element']] - r_avg)**2
    for el, conc in candidate['dopants']:
        variance += conc * (IONIC_RADII.get(el, 85.0) - r_avg)**2
    radius_mismatch = np.sqrt(variance)

    print(f"    Radius Mismatch (DR):  {radius_mismatch:.4f} pm")

    # 3. Tolerance-factor criterion (calibrated for zirconia)
    # Zirconia is not an ideal Pauling crystal; compare against YSZ.
    ratio = r_avg / IONIC_RADII['O']
    print(f"    Cation/Anion Ratio:    {ratio:.4f}")

    # --- [Key change] Refined decision logic ---
    # Region 1: Stable cubic (close to or above YSZ)
    if ratio >= 0.615 and radius_mismatch < 6.5:
        status = "STABLE (Cubic Phase)"
        color_code = "\033[1;32m" # Green
        is_stable = True

    # Region 2: Metastable / tetragonal (slightly below YSZ, but very high conductivity, e.g., ScSZ)
    elif ratio >= 0.605:
        status = "METASTABLE (Tetragonal/Cubic Mixed - High Conductivity)"
        color_code = "\033[1;33m" # Yellow
        is_stable = True # Engineering-viable, potentially tougher.

    # Region 3: Unstable (too small -> monoclinic) or mismatch too large
    else:
        if radius_mismatch >= 6.5:
            status = "UNSTABLE (Phase Separation Risk)"
        else:
            status = "UNSTABLE (Monoclinic Distortion Risk)"
        color_code = "\033[1;31m" # Red
        is_stable = False

    print(f"    Phase Prediction:      {color_code}{status}\033[0m")
    return r_avg, radius_mismatch, ratio, is_stable

def generate_dft_input(candidate, r_avg):
    print("\n>>> [Handoff] Generating DFT Input Structure (POSCAR stub)...")
    est_lattice_constant = 5.125 + (r_avg - 84.0) * 0.015
    filename = os.path.join(path_config.RESULTS_DIR, "POSCAR_AI_Discovery.vasp")

    with open(filename, 'w') as f:
        f.write(f"System {candidate['dopants']}\n")
        f.write("! WARNING: This is a POSCAR stub. Missing element counts and full atomic coordinates.\n")
        f.write("! Use pymatgen/VESTA/SQS to build a complete supercell before running DFT.\n")
        f.write(f"1.0\n")
        f.write(f"  {est_lattice_constant:.5f} 0.00 0.00\n  0.00 {est_lattice_constant:.5f} 0.00\n  0.00 0.00 {est_lattice_constant:.5f}\n")
        f.write(f"  Zr O {' '.join([d[0] for d in candidate['dopants']])}\n")
        f.write("  Direct\n  0.00 0.00 0.00\n")
    print(f"    -> Generated structural file: {filename}")

def plot_stability_map(r_avg, mismatch, is_stable):
    plt.figure(figsize=(8, 6))

    # Draw region background
    plt.fill_between([82, 88], 0, 10, color='red', alpha=0.1, label='Unstable (Monoclinic)')
    plt.fill_between([84.5, 88], 0, 6.5, color='yellow', alpha=0.2, label='Metastable (Tetragonal)')
    plt.fill_between([85.2, 88], 0, 6.0, color='green', alpha=0.2, label='Stable (Cubic)')

    # YSZ reference point
    plt.scatter([YSZ_8_RADIUS], [1.5], c='blue', s=150, marker='s', label='8-YSZ (Reference)')

    # AI discovery point
    color = 'green' if is_stable else 'red'
    # [Fix] Unfilled marker 'x' should not use edgecolors
    if is_stable:
        plt.scatter([r_avg], [mismatch], c='gold', s=300, marker='*', edgecolors='black', label='AI Discovery')
    else:
        plt.scatter([r_avg], [mismatch], c='red', s=200, marker='x', linewidth=3, label='AI Discovery')

    plt.title("Physical Validation: Phase Stability Map")
    plt.xlabel("Average Cation Radius (pm)")
    plt.ylabel("Radius Mismatch (pm)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(path_config.VALIDATION_STABILITY_MAP_IMAGE_PATH)
    plt.savefig(save_path)
    print(f"\n>>> [Visual] Stability map saved to {save_path}")

if __name__ == "__main__":
    candidate_data = load_best_recipe_from_csv()
    r, dr, ratio, stable = calculate_stability_metrics(candidate_data)
    generate_dft_input(candidate_data, r)
    plot_stability_map(r, dr, stable)
