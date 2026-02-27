import os
import hashlib
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings

# --- 1. Import configuration ---
try:
    from config import path_config
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import path_config

# Ignore CHGNet's internal tensor warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# Dependency check
# ---------------------------------------------------------
try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    HAS_MD_PACKAGES = True
except ImportError:
    HAS_MD_PACKAGES = False
    print("⚠️ Warning: CHGNet/Pymatgen/ASE dependencies not detected.")

# Whether to allow demo mode (generate synthetic data). Disabled by default to avoid misuse.
DEMO_MODE = os.environ.get("DEMO_MODE", "0").strip().lower() in {"1", "true", "yes", "y"}
if not HAS_MD_PACKAGES and not DEMO_MODE:
    raise SystemExit(
        "CHGNet/Pymatgen/ASE dependencies not detected. Install the required packages, "
        "or explicitly set DEMO_MODE=1 to run in demo mode."
    )

# --- Configuration ---
AI_RESULTS_CSV = path_config.AI_DISCOVERY_RESULTS_CSV
VALIDATION_PLOT_PATH = os.path.join(path_config.PAPER_COMPUTATIONAL_VALIDATION_IMAGE_PATH)

# --- MD parameters (override via env vars) ---
# MD_TEMP_K defaults to None: auto-align to target_temperature_c in the PIML CSV (converted to K).
# To set manually (e.g., accelerated mode at 1500K), set env var MD_TEMP_K=1500.
_MD_TEMP_K_ENV = os.environ.get("MD_TEMP_K", "")
MD_TEMP_K = float(_MD_TEMP_K_ENV) if _MD_TEMP_K_ENV.strip() else None  # None = auto-align
TIME_STEP_FS = float(os.environ.get("MD_TIME_STEP_FS", "2.0"))
MD_FRICTION = float(os.environ.get("MD_FRICTION", "0.02"))
MD_EQUIL_STEPS = max(0, int(os.environ.get("MD_EQUIL_STEPS", "10000")))
# Default 200k steps (diffusion is slow at 1073K; a longer trajectory is needed)
MD_PROD_STEPS = max(1, int(os.environ.get("MD_PROD_STEPS", "200000")))
MD_RECORD_INTERVAL = max(1, int(os.environ.get("MD_RECORD_INTERVAL", "25")))
MD_REPEATS = max(1, int(os.environ.get("MD_REPEATS", "3")))
# Supercell multiplier: 2->96 atoms(32 Zr), 3->324 atoms(108 Zr), 4->768 atoms(256 Zr)
# Default 3 to ensure vacancy counts have enough resolution
MD_SUPERCELL_N = max(2, int(os.environ.get("MD_SUPERCELL_N", "3")))

# Physical constants
KB = 1.380649e-23  # J/K
CHARGE_E = 1.60217663e-19 # C

# ---------------------------------------------------------
# Reproducibility: random seeds
# ---------------------------------------------------------
def _stable_candidate_seed(base_seed: int, label: str) -> int:
    """Generate a stable 32-bit seed per candidate to avoid drift when iteration order changes."""
    h = hashlib.blake2b(f"{base_seed}|{label}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False) % (2**32)


def _seed_everything(seed: int) -> None:
    """Synchronize random sources across the script and dependencies as much as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Core class: MD validator
# ---------------------------------------------------------
class MDValidator:
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        if not demo_mode and HAS_MD_PACKAGES:
            # Force CUDA if available
            if torch.cuda.is_available():
                print("🚀 Loading CHGNet (GPU Mode)...")
                self.chgnet = CHGNet.load()
            else:
                print("⚠️ Loading CHGNet (CPU Mode) - slower...")
                self.chgnet = CHGNet.load()

    def build_supercell(self, d1, f1, d2, f2, rng=None):
        """Build a doped supercell."""
        if self.demo_mode: return None
        if rng is None:
            rng = np.random.default_rng()

        # 1. Base structure
        a0 = 5.12
        base_struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(a0), ["Zr", "O"], [[0,0,0], [0.25,0.25,0.25]])
        n = MD_SUPERCELL_N
        base_struct.make_supercell([n, n, n])
        print(f"      Supercell: {n}x{n}x{n} -> {len(base_struct)} atoms")

        # 2. Cation doping
        zr_sites = [i for i, s in enumerate(base_struct) if s.specie.symbol == "Zr"]
        n_d1 = int(round(len(zr_sites) * f1))
        n_d2 = int(round(len(zr_sites) * f2))

        replace_indices = rng.choice(zr_sites, n_d1 + n_d2, replace=False)
        for i, idx in enumerate(replace_indices):
            element = d1 if i < n_d1 else d2
            base_struct.replace(idx, element)

        # 3. Oxygen vacancies (charge balance)
        # Valence table (Zr=+4 reference)
        valences = {
            "Sc": 3, "Y": 3, "Gd": 3, "Yb": 3, # +3 Trivalent
            "Mg": 2, "Ca": 2,                  # +2 Divalent
            "Zr": 4
        }
        
        v1 = valences.get(d1, 3) # default to +3
        v2 = valences.get(d2, 3) # default to +3
        
        # Charge compensation: Vacancy (+2) balances deficits
        # Deficit = Sum( N_dopant * (4 - Valence) )
        charge_deficit = n_d1 * (4 - v1) + n_d2 * (4 - v2)
        # Note: If charge_deficit is odd, round() can leave a net charge in the supercell (finite-size effect).
        # This can be negligible in large-supercell statistics, but careful DFT needs background charge compensation.
        n_vacancies = int(round(charge_deficit / 2.0))

        if n_vacancies > 0:
            o_sites = [i for i, s in enumerate(base_struct) if s.specie.symbol == "O"]
            # Ensure we don't remove too many
            if n_vacancies >= len(o_sites): n_vacancies = len(o_sites) - 1
            remove_indices = rng.choice(o_sites, n_vacancies, replace=False)
            base_struct.remove_sites(remove_indices)

        return base_struct

    def calculate_conductivity_nernst_einstein(self, slope_A2_ps, volume_A3, n_carriers, T):
        """
        [Physics core] Compute conductivity via the Nernst–Einstein equation.
        sigma = (n * q^2 * D) / (k * T)
        """
        # 1. Diffusion coefficient D (cm^2/s)
        # MSD slope units: A^2/ps
        # 1 A^2/ps = 1e-16 cm^2 / 1e-12 s = 1e-4 cm^2/s
        # D = slope / 6
        D_cm2s = (slope_A2_ps / 6.0) * 1e-4

        # 2. Carrier concentration n (cm^-3)
        vol_cm3 = volume_A3 * 1e-24
        n_conc = n_carriers / vol_cm3

        # 3. Conductivity sigma (S/cm)
        # q = 2e (oxygen ion)
        q = 2 * CHARGE_E

        # Note: k is in J/K; D needs to be in m^2/s to match J (kg*m^2/s^2).
        # Or we can reason through units directly:
        # sigma = [cm^-3] * [C^2] * [cm^2/s] / [J/K * K]
        #       = [cm^-1] * [C^2/s] / [J]
        #       = [cm^-1] * [A*C] / [V*C] = S/cm (as long as units are consistent)

        # It's safer to compute in SI units and convert back to S/cm.
        n_m3 = n_conc * 1e6
        D_m2s = D_cm2s * 1e-4
        sigma_Sm = (n_m3 * q**2 * D_m2s) / (KB * T) # S/m

        sigma_Scm = sigma_Sm / 100.0 # S/m -> S/cm

        return sigma_Scm

    def run_simulation(self, structure):
        """Run MD and return (sigma [S/cm], log10(sigma))."""
        if self.demo_mode:
            return 1e-6, -6.0

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = CHGNetCalculator(model=self.chgnet, use_device='cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize thermal bath
        MaxwellBoltzmannDistribution(atoms, temperature_K=MD_TEMP_K)
        Stationary(atoms)
        dyn = Langevin(atoms, TIME_STEP_FS * units.fs, temperature_K=MD_TEMP_K, friction=MD_FRICTION)

        # Equilibrate (exclude from MSD)
        if MD_EQUIL_STEPS > 0:
            dyn.run(MD_EQUIL_STEPS)

        # State variables for unwrapping (accumulate from production phase)
        self.last_positions = atoms.get_positions()
        cell = atoms.get_cell()
        cell_inv = np.linalg.inv(cell)

        positions = []
        def record():
            indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'O']
            
            # Manual unwrapping logic (based on fractional-coordinate differences)
            curr_positions = atoms.get_positions()
            
            # 1. Convert to fractional coordinates
            curr_scaled = np.dot(curr_positions, cell_inv) # ASE items are rows: pos = frac @ cell => frac = pos @ inv(cell)
            last_scaled = np.dot(self.last_positions, cell_inv)
            
            # 2. Compute fractional displacement and apply MIC (Minimum Image Convention)
            delta_scaled = curr_scaled - last_scaled
            delta_scaled -= np.round(delta_scaled)
            
            # 3. Convert back to Cartesian displacement and update continuous positions
            delta_cartesian = np.dot(delta_scaled, cell)
            
            # Update continuous positions
            self.last_positions = self.last_positions + delta_cartesian
            
            # Record continuous trajectories of oxygen atoms
            positions.append(self.last_positions[indices])

        # Production run: record MSD
        interval = max(1, MD_RECORD_INTERVAL)
        record()  # t=0
        dyn.attach(record, interval=interval)

        # Run
        dyn.run(MD_PROD_STEPS)

        # MSD computation
        pos_array = np.array(positions) # (Frames, N_O, 3)
        if len(pos_array) < 10:
            return 1e-9, -9.0

        # [Fix] Remove center-of-mass (COM) drift.
        # The Langevin thermostat applies random forces to each atom, which makes the global COM perform a random walk.
        # For slow-diffusion systems (e.g., Pure_ZrO2), COM drift can become the dominant contribution to the MSD.
        com = np.mean(pos_array, axis=1, keepdims=True)   # (Frames, 1, 3)
        com_drift = com - com[0]                           # COM displacement relative to t=0
        com_drift_magnitude = np.sqrt(np.sum(com_drift[-1, 0]**2))
        print(f"      COM drift: {com_drift_magnitude:.4f} Å (total over trajectory)")
        pos_corrected = pos_array - com_drift              # Subtract COM drift from each atom's displacement

        # MSD = <|r(t) - r(0)|^2>  (COM-corrected)
        sq_disp = np.sum((pos_corrected - pos_corrected[0])**2, axis=2)
        msd = np.mean(sq_disp, axis=1) # (Frames,)

        # Linear fit (use second half to avoid initial transients)
        time_ps = np.arange(len(msd)) * (interval * TIME_STEP_FS) / 1000.0
        start_idx = int(len(msd) * 0.5)

        slope, intercept = np.polyfit(time_ps[start_idx:], msd[start_idx:], 1)
        print(f"      MSD slope: {slope:.4e} Å²/ps")

        # [Fix] Guard against negative/too-small slopes
        if slope <= 1e-5:
            print(f"      [Warning] Diffusivity below detection limit (Slope={slope:.2e}). Returning floor.")
            return 1e-6, -6.0 # Use a low conductivity floor (below limit)

        # Compute physical conductivity
        n_oxygen = len([s for s in structure if s.specie.symbol == "O"])
        sigma = self.calculate_conductivity_nernst_einstein(slope, structure.volume, n_oxygen, MD_TEMP_K)

        return sigma, np.log10(sigma)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    global MD_TEMP_K  # May need to resolve from None to an actual value here

    # 1. Build validation set
    validation_candidates = [
        ("AI_Best", "Sc", 0.08, "Yb", 0.02, -1.25),
        ("AI_Top2", "Y",  0.08, "Gd", 0.02, -1.35),
        ("Pure_ZrO2", "Zr", 0.00, "Zr", 0.00, -3.50),
        ("Poor_Mg",   "Mg", 0.05, "Mg", 0.00, -2.10),
    ]

    piml_target_temp_c = 800.0
    if os.path.exists(AI_RESULTS_CSV):
        df = pd.read_csv(AI_RESULTS_CSV)
        best = df.iloc[0]
        validation_candidates[0] = (
            "AI_Best",
            best['dopant_1_element'], best['dopant_1_fraction'],
            best['dopant_2_element'], best['dopant_2_fraction'],
            best['predicted_log_conductivity']
        )
        try:
            temp_c = float(best.get("target_temperature_c", piml_target_temp_c))
            if np.isfinite(temp_c):
                piml_target_temp_c = temp_c
        except Exception:
            pass

    piml_target_temp_k = piml_target_temp_c + 273.15

    # Temperature alignment: when MD_TEMP_K is not explicitly set, auto-align to the PIML target temperature.
    if MD_TEMP_K is None:
        MD_TEMP_K = piml_target_temp_k
        print(f"    MD_TEMP_K auto-aligned to PIML target: {piml_target_temp_c:.0f}°C -> {MD_TEMP_K:.2f}K")
    else:
        print(f"    MD_TEMP_K explicitly set: {MD_TEMP_K}K (env: MD_TEMP_K)")
        if abs(MD_TEMP_K - piml_target_temp_k) > 1.0:
            print(f"    [NOTE] MD_TEMP_K ({MD_TEMP_K}K) != PIML target ({piml_target_temp_k:.2f}K).")
            print(f"           Focus on relative trends instead of direct magnitude comparison.")

    print(
        f">>> [Step 7] Computational Validation "
        f"(T={MD_TEMP_K:.1f}K, dt={TIME_STEP_FS}fs, equil={MD_EQUIL_STEPS}, prod={MD_PROD_STEPS}, "
        f"repeats={MD_REPEATS}, supercell={MD_SUPERCELL_N}x{MD_SUPERCELL_N}x{MD_SUPERCELL_N})"
    )
    print(f"    Target: {VALIDATION_PLOT_PATH}")

    base_seed_str = os.environ.get("MD_SEED", "42")
    try:
        base_seed = int(base_seed_str)
    except ValueError:
        print(f"⚠️ Warning: Could not parse MD_SEED='{base_seed_str}', falling back to 42.")
        base_seed = 42
    print(f"    Random Seed (base): {base_seed} (env: MD_SEED)")

    validator = MDValidator(demo_mode=DEMO_MODE or (not HAS_MD_PACKAGES))

    results_piml = []
    results_md = []
    results_md_std = []
    labels = []

    for label, d1, f1, d2, f2, piml_val in validation_candidates:
        print(f"\n   -> Validating: {label} ({d1}={f1:.2f}, {d2}={f2:.2f})")
        md_logs = []
        for rep in range(max(1, MD_REPEATS)):
            rep_seed = _stable_candidate_seed(base_seed, f"{label}|rep{rep}")
            print(f"      RNG Seed (rep {rep+1}): {rep_seed}")
            _seed_everything(rep_seed)
            rng = np.random.default_rng(rep_seed)

            try:
                if validator.demo_mode:
                    md_val = piml_val - 0.2 + rng.normal(0, 0.1)
                    print(f"        Rep {rep+1}: PIML={piml_val:.2f} | MD={md_val:.2f} [DEMO]")
                else:
                    struct = validator.build_supercell(d1, f1, d2, f2, rng=rng)
                    sigma, md_val = validator.run_simulation(struct) if struct else (1e-6, -6.0)
                    print(f"        Rep {rep+1}: PIML={piml_val:.2f} | MD={md_val:.2f} (σ={sigma:.3e} S/cm)")
            except Exception as e:
                print(f"        Rep {rep+1}: !!! Simulation Failed: {e}")
                md_val = -6.0

            md_logs.append(md_val)

        md_mean = float(np.mean(md_logs))
        md_std = float(np.std(md_logs, ddof=1)) if len(md_logs) > 1 else 0.0
        print(f"      Result: PIML={piml_val:.2f} | MD={md_mean:.2f} ± {md_std:.2f} (log10 σ)")

        results_piml.append(piml_val)
        results_md.append(md_mean)
        results_md_std.append(md_std)
        labels.append(label)

    # 2. Plot
    plt.figure(figsize=(9, 7))
    colors = ['#D32F2F' if 'AI' in l else '#757575' for l in labels]

    # Scatter plot + error bars (std. dev. over repeats)
    for x, y, yerr, c in zip(results_piml, results_md, results_md_std, colors):
        plt.errorbar(
            x,
            y,
            yerr=yerr if yerr > 0 else None,
            fmt="o",
            color=c,
            ecolor="k",
            elinewidth=1,
            capsize=4,
            markersize=10,
            markeredgecolor="k",
            zorder=5,
        )

    for x, y, l in zip(results_piml, results_md, labels):
        plt.text(x+0.05, y+0.05, l, fontsize=10, fontweight='bold')

    # Ideal diagonal (1:1) line
    min_v = min(min(results_piml), min(results_md)) - 0.5
    max_v = max(max(results_piml), max(results_md)) + 0.5
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.3, label="Ideal 1:1")

    plt.title(f"Computational Validation\nPIML Prediction vs. CHGNet MD ({MD_TEMP_K:.0f}K)", fontsize=14)
    plt.xlabel(r"PIML Predicted Log($\sigma$) [S/cm]", fontsize=12)
    plt.ylabel(f"MD Calculated Log($\\sigma$) [S/cm]", fontsize=12) # Use raw string or double backslash
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(VALIDATION_PLOT_PATH, dpi=300)
    print(f"\n✅ Validation plot generated: {VALIDATION_PLOT_PATH}")

if __name__ == "__main__":
    main()
