"""Exercise mixed native libraries in isolated processes so crashes are observable."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS OpenMP compatibility check")
@pytest.mark.parametrize("order", ["xgboost, torch, sklearn", "torch, sklearn, xgboost"])
def test_mixed_numerical_libraries_in_fresh_process(order):
    script = f"""
import backend
import {order}
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_info

rng = np.random.default_rng(42)
X = rng.normal(size=(1200, 8))
y = (X[:, 0] > 0).astype(int)
def exercise():
    for _ in range(3):
        matrix = xgboost.DMatrix(X, label=y, nthread=1)
        model = xgboost.train({{"nthread": 1, "objective": "binary:logistic"}}, matrix, num_boost_round=5)
        assert np.isfinite(model.predict(matrix)).all()
        tensor = torch.tensor([[2., .5], [.5, 1.]], dtype=torch.double)
        assert torch.isfinite(torch.linalg.cholesky(tensor)).all()
        KMeans(n_clusters=2, n_init=1, random_state=42).fit(X)
with ThreadPoolExecutor(max_workers=1) as executor:
    executor.submit(exercise).result()
assert all(pool["num_threads"] == 1 for pool in threadpool_info() if pool["user_api"] == "openmp")
print("Native runtime check passed")
"""
    env = os.environ.copy()
    # Confirm the application, rather than an inherited shell setting, applies the fix.
    env["OMP_NUM_THREADS"] = "8"
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Native runtime check passed" in result.stdout
