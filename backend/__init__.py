"""Configure native runtimes before importing application numerical libraries."""

import os
import sys

# Limit native threads to avoid conflicts between numerical runtimes on Apple Silicon.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
