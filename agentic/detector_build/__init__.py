"""Deterministic support code for the agentic detector-build workflow.

The package deliberately contains the parts that do not require code-agent
judgment: artifact discovery, provenance joins, stable contracts, and executable
verification.  Feature semantics remain the responsibility of the builder agent.
"""

from .contracts import BUILD_ARTIFACT_NAMES, DetectorTarget, RunContext

__all__ = ["BUILD_ARTIFACT_NAMES", "DetectorTarget", "RunContext"]
