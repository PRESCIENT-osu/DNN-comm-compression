"""Artifact store for optimization experiments.

Manages file paths and cache validity for artifacts produced by the various
optimization sub-experiment phases (profiling, accuracy model training,
estimator state persistence).

Artifacts are stored under a per-experiment root directory:

    {artifacts_dir}/
        profiling/
            tau_{pipeline}_{node}.json    -- per-node compute latency
            a_i_{pipeline}_{link}.json    -- activation tensor size at link boundary
        accuracy_models/
            {pipeline}_{model_id}.pkl     -- serialised sklearn surrogate model
            {pipeline}_{model_id}.json    -- Stein oracle gradient state
        estimators/
            {from_node}_{to_node}.json    -- channel estimator observation history
        slots/
            slot_{n:06d}.json             -- per-slot summary written by opt_runner
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Manages artifact paths and cache validation for one optimization experiment.

    Two roots are maintained:
    - ``root``: per-experiment directory for profiling, estimator state, and slot
      summaries.  These depend on live network measurements and are not portable
      across experiments.
    - ``shared_root``: experiment-agnostic directory for accuracy model artifacts.
      Accuracy models depend only on simulation (model, partitions, compression
      scheme, dataset seed) and are safely shared across experiments that use
      identical pipeline configs.

    Cache validation uses a short SHA-256 hash of the config dict that produced
    the artifact.  On read, the stored hash is compared to the expected hash; a
    mismatch triggers re-computation.  Pass ``expected_hash=None`` to skip the
    check and treat any existing file as valid.

    Args:
        artifacts_dir: Root directory for per-experiment artifacts.  Created on
            init if it does not exist.
        shared_artifacts_dir: Root directory for shared accuracy model artifacts.
            Defaults to ``artifacts/shared`` relative to the current working
            directory if not provided.  Created on init if it does not exist.
    """

    def __init__(
        self,
        artifacts_dir: Path | str,
        shared_artifacts_dir: Path | str | None = None,
    ) -> None:
        self.root = Path(artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shared_root = Path(
            shared_artifacts_dir
            if shared_artifacts_dir is not None
            else "artifacts/shared"
        )
        self.shared_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def profiling_tau_path(self, pipeline_id: str, node_id: str) -> Path:
        """Path for the compute latency artifact for one (pipeline, node) pair.

        Args:
            pipeline_id: Pipeline identifier.
            node_id: Node identifier.

        Returns:
            Path to JSON file containing ``{"tau_s": float, ...}``.
        """
        return self.root / "profiling" / f"tau_{pipeline_id}_{node_id}.json"

    def profiling_a_i_path(self, pipeline_id: str, link_id: str) -> Path:
        """Path for the activation size artifact for one (pipeline, link) pair.

        Args:
            pipeline_id: Pipeline identifier.
            link_id: Link identifier, e.g. ``"A-B"``.

        Returns:
            Path to JSON file containing ``{"a_i_bytes": int, ...}``.
        """
        return self.root / "profiling" / f"a_i_{pipeline_id}_{link_id}.json"

    def accuracy_model_path(
        self,
        model: str,
        partitions: dict[str, list[str]],
        flow: list[str],
        simulation_path: str | None,
        surrogate_type: str,
        compression_method_per_link: dict[str, str],
        sweep_design: str,
        n_sweep_samples: int,
        dataset_seed: int | None,
        dataset_max_samples: int | None,
        dataset_name: str | None = None,
    ) -> Path:
        """Content-addressed path for a shared accuracy model artifact.

        The filename encodes the key dimensions of the accuracy model (model
        family, partition layout, surrogate type, compression scheme) and a
        short hash of the full uniqueness set so that two experiments using
        identical pipeline configs and dataset seeds reuse the same artifact.

        Artifacts are stored under ``shared_root/accuracy_models/`` rather than
        the per-experiment root so they are portable across experiments.

        ``.pkl`` suffix is used for surrogate sklearn models; the caller is
        responsible for appending the correct suffix.

        Args:
            model: Model family string, e.g. ``"resnet"`` or ``"llama"``.
            partitions: Mapping of node name to list of partition IDs, e.g.
                ``{"A": ["p1"], "B": ["p2", "p3"], "C": ["p4", "p5"]}``.
            flow: Ordered node names for this pipeline.
            simulation_path: Path or name of the full-model checkpoint used by
                the simulation (affects hook positions for Llama).
            surrogate_type: sklearn surrogate model type, e.g. ``"gbm"``.
            compression_method_per_link: ``{link_id: method}`` for each link in
                flow order, e.g. ``{"A-B": "topk", "B-C": "topk"}``.
            sweep_design: ``"diagonal"`` or ``"random"``.
            n_sweep_samples: Number of sweep samples used for training.
            dataset_seed: RNG seed used for the sweep dataset.
            dataset_max_samples: Cap on dataset size (affects evaluation cost
                and training data distribution).
            dataset_name: Dataset identifier, e.g. ``"wikitext2"`` or ``"mmlu"``.
                Included in the uniqueness hash to prevent collisions between
                accuracy models trained on different datasets with the same seed
                and sample count.

        Returns:
            Base path (without suffix) under ``shared_root/accuracy_models/``.
        """
        # Human-readable components
        model_slug = (
            model.lower().replace("-", "").replace(".", "")
        )  # e.g. "resnet", "llama318b"

        # Partition layout in flow order: "A.p1_B.p2p3_C.p4p5"
        partition_parts = []
        for node in flow:
            node_parts = "".join(partitions.get(node, []))
            partition_parts.append(f"{node}.{node_parts}")
        partition_layout = "_".join(partition_parts)

        # Compression scheme in link flow order: "topk+topk"
        link_methods = []
        for i in range(len(flow) - 1):
            link_id = f"{flow[i]}-{flow[i + 1]}"
            link_methods.append(compression_method_per_link.get(link_id, "none"))
        compression_scheme = "+".join(link_methods) if link_methods else "none"

        # Short hash over the full uniqueness set
        content = {
            "model": model.lower(),
            "partitions": {k: sorted(v) for k, v in partitions.items()},
            "flow": flow,
            "simulation_path": simulation_path,
            "surrogate_type": surrogate_type,
            "compression_method_per_link": compression_method_per_link,
            "sweep_design": sweep_design,
            "n_sweep_samples": n_sweep_samples,
            "dataset_name": dataset_name,
            "dataset_seed": dataset_seed,
            "dataset_max_samples": dataset_max_samples,
        }
        blob = json.dumps(content, sort_keys=True, default=str).encode()
        content_hash8 = hashlib.sha256(blob).hexdigest()[:8]

        name = f"{model_slug}__{partition_layout}__{surrogate_type}__{compression_scheme}__{content_hash8}"
        return self.shared_root / "accuracy_models" / name

    def estimator_state_path(self, from_node: str, to_node: str) -> Path:
        """Path for the persisted channel estimator state for one link.

        Args:
            from_node: Source node name.
            to_node: Destination node name.

        Returns:
            Path to JSON file containing ``{"observations": [...], ...}``.
        """
        return self.root / "estimators" / f"{from_node}_{to_node}.json"

    def slot_summary_path(self, slot_id: int) -> Path:
        """Path for the per-slot summary written by the optimization runner.

        Args:
            slot_id: Zero-based slot index.

        Returns:
            Path to JSON file for this slot.
        """
        return self.root / "slots" / f"slot_{slot_id:06d}.json"

    # ------------------------------------------------------------------
    # Hash and validity
    # ------------------------------------------------------------------

    def config_hash(self, config: dict[str, Any]) -> str:
        """Compute a short SHA-256 hash of a config dict for cache invalidation.

        The dict is JSON-serialized with sorted keys before hashing so that
        equivalent configs with different insertion orders produce the same hash.

        Args:
            config: Config dict to hash.

        Returns:
            16-character hex string (first 64 bits of SHA-256).
        """
        blob = json.dumps(config, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def is_valid(self, path: Path, expected_hash: str | None = None) -> bool:
        """Return True if the artifact exists and its stored hash matches.

        Args:
            path: Path to the artifact file (JSON only; .pkl files always
                return True if they exist since they cannot store a hash).
            expected_hash: Hash to compare against the ``_config_hash`` field
                embedded in the JSON file.  Pass ``None`` to skip hash check.

        Returns:
            True if the artifact is present and (when expected_hash is given)
            the stored hash matches.
        """
        if not path.exists():
            return False
        if expected_hash is None:
            return True
        if path.suffix != ".json":
            # Non-JSON artifacts (e.g. .pkl) cannot embed a hash; treat as valid.
            return True
        try:
            with open(path) as f:
                data = json.load(f)
            stored = data.get("_config_hash")
            if stored != expected_hash:
                logger.debug(
                    "Artifact %s hash mismatch: stored=%s expected=%s — will recompute",
                    path,
                    stored,
                    expected_hash,
                )
                return False
            return True
        except Exception as exc:
            logger.warning("Could not read artifact %s for validation: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # JSON read / write
    # ------------------------------------------------------------------

    def write_json(
        self,
        path: Path,
        data: dict[str, Any],
        config_hash: str | None = None,
    ) -> None:
        """Write a dict to a JSON artifact file, optionally embedding a hash.

        Creates parent directories as needed.

        Args:
            path: Destination file path.
            data: Dict to serialise.
            config_hash: If provided, embedded as ``_config_hash`` in the file
                for future validity checks.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = dict(data)
        if config_hash is not None:
            payload["_config_hash"] = config_hash
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.debug("Wrote artifact: %s", path)

    def read_json(self, path: Path) -> dict[str, Any]:
        """Read a JSON artifact file.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed dict (may contain ``_config_hash`` key).

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        with open(path) as f:
            return json.load(f)
