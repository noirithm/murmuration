"""Global settings, seeding and environment logging."""
import json
import platform
import random
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank_campaign_participant_release.xlsx"
RESULTS = ROOT / "results"

# Stage 1
N_QUBITS_S1 = 5
FEATURE_MAP_REPS = 2
QUANTUM_SCALES = [0.5, 1.0, 2.0]     # feature-map bandwidth variants, compared by CV on train only
QUANTUM_ENCODERS = ["pca", "supervised"]
CV_FOLDS, CV_REPEATS = 5, 3

# Stage 2
QAOA_REPS = 3
QAOA_RESTARTS = 8
QAOA_MAXITER = 250
CVAR_ALPHA = 0.1
STAGE2_SHOTS = 8192
STAGE2_NOISE_SCALES = [0.0, 1.0, 3.0]
STAGE2_NOISE_SHOTS = 1000            # noisy 15-qubit trajectories are slow

# Stage 3
STAGE3_SHOTS = 8192
MC_SAMPLES = 1_000_000
NOISE_SCALES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
FAKE_BACKEND = "FakeBrisbane"


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def env_info(seed: int = SEED) -> dict:
    import qiskit, qiskit_aer, qiskit_ibm_runtime, scipy, sklearn
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
        "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__,
        "scikit_learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "seeds": {"global": seed, "cv_split": seed, "models": seed, "svc": seed,
                  "qaoa_optimizer": seed, "aer_simulator": seed,
                  "transpiler": seed, "monte_carlo": seed},
    }


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj, name: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, default=_np))
    return path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)   # sklearn 1.9 deprecation noise
