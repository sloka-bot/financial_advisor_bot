"""Configure native runtimes before importing application numerical libraries."""

import os
import sys

# Apple Silicon installations can load separate OpenMP runtimes through Torch,
# sklearn and XGBoost. Serial execution avoids their worker-barrier crash path.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
