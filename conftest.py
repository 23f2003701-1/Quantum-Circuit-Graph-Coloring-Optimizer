# conftest.py — Root pytest configuration
#
# On Windows, Qiskit Aer (specifically qiskit_aer.backends.backend_utils) triggers
# a fatal C-level exception (0xc0000008 — invalid handle) when first loaded inside
# pytest's assertion-rewriting exec_module path.
#
# The fix: pre-import qiskit_aer (and therefore its problematic C extensions) during
# conftest loading, BEFORE pytest's assertion rewriter imports test_all.py.
# Once the DLL is loaded into the process, subsequent imports are no-ops and are safe.

import sys
import os

# Ensure the project root is on the path so `modules.*` imports work from tests/
sys.path.insert(0, os.path.dirname(__file__))

# Pre-load Qiskit Aer to avoid the Windows DLL fatal exception during assertion
# rewriting. This runs before pytest collects any test modules.
try:
    import qiskit_aer  # noqa: F401 — side-effect import only
except Exception:
    pass  # If unavailable, noise tests will be skipped via _NOISE_AVAILABLE flag
