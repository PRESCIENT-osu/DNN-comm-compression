"""Factory for building simulation pipeline objects used by the Stein gradient oracle.

Constructs ``SimulatedResNetPipeline`` and ``SimulatedLlamaPipeline`` instances
from the experiment config.  Simulations are built only for pipelines that have
``simulation_path`` set.

The factory is called by ``opt_runner._run_opt_with_adapter`` whenever the active
sub-experiment has a non-None ``stein_config``.  The returned dict is passed
directly to ``build_inference_tasks`` as the ``simulations`` argument.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from framework.datamodels.experiment import DatasetConfig
    from framework.datamodels.opt_experiment import (
        GeneratedOptExperimentConfig,
        OptLinkConfig,
        SteinOracleConfig,
    )

logger = logging.getLogger(__name__)

_MODEL_RESNET = {"resnet", "resnet56"}
_MODEL_LLAMA = {"llama", "llama-3.1-8b"}
_DATASET_WIKITEXT = {"wikitext2", "wikitext-2"}


# ---------------------------------------------------------------------------
# CIFAR-10 loader
# ---------------------------------------------------------------------------


def _build_cifar10_loader(
    cfg: DatasetConfig,
) -> Any:
    """Build a CIFAR-10 test DataLoader from a dataset config.

    Args:
        cfg: Dataset config with ``path``, ``batch_size``, ``max_samples``,
            and ``seed`` fields.

    Returns:
        ``torch.utils.data.DataLoader`` over the CIFAR-10 test split.
    """
    import torchvision.datasets as tv_datasets  # noqa: PLC0415
    import torchvision.transforms as transforms  # noqa: PLC0415
    from torch.utils.data import DataLoader, Subset  # noqa: PLC0415

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset = tv_datasets.CIFAR10(
        root=cfg.path, train=False, download=False, transform=transform
    )
    if cfg.max_samples is not None:
        if cfg.seed is not None:
            generator = torch.Generator().manual_seed(cfg.seed)
            idx = torch.randperm(len(dataset), generator=generator)[
                : cfg.max_samples
            ].tolist()
        else:
            idx = list(range(min(cfg.max_samples, len(dataset))))
        dataset = Subset(dataset, idx)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# MMLU local evaluator
# ---------------------------------------------------------------------------


class _MMLULocalEvaluator:
    """MMLU evaluator that runs the full model locally.

    Implements the ``LlamaEvaluator`` protocol required by
    ``SimulatedLlamaPipeline``.  Tokenizes MMLU questions once at construction
    and runs them through the hooked model at evaluation time.

    Args:
        cfg: Dataset config with MMLU subjects, samples_per_subject, etc.
        tokenizer_path: Path to the saved Llama tokenizer directory.
        max_items: Cap on the total number of MMLU items to evaluate.
            ``None`` uses all items produced by ``cfg``.
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        tokenizer_path: str,
        max_items: int | None = None,
    ) -> None:
        from datasets import load_dataset  # type: ignore[import]  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self._tokenizer = tokenizer
        self._answer_token_ids: list[int] = [
            tokenizer.encode(" " + c, add_special_tokens=False)[-1]
            for c in ["A", "B", "C", "D"]
        ]

        subjects = cfg.subjects or [
            "college_computer_science",
            "high_school_mathematics",
            "professional_law",
            "global_facts",
            "miscellaneous",
            "business_ethics",
        ]
        samples_per = cfg.samples_per_subject or 50

        rng = torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None

        items: list[dict[str, Any]] = []
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test")
                n = min(len(ds), samples_per)
                if rng is not None:
                    indices = torch.randperm(len(ds), generator=rng).tolist()[:n]
                else:
                    indices = list(range(n))
                for i in indices:
                    row = ds[i]
                    items.append(
                        {
                            "question": row["question"],
                            "choices": row["choices"],
                            "answer": row["answer"],
                        }
                    )
            except Exception as exc:
                logger.warning("Skipping MMLU subject %s: %s", subj, exc)

        if max_items is not None:
            items = items[:max_items]
        elif cfg.max_samples is not None:
            items = items[: cfg.max_samples]

        self._items = items
        self._batch_size = cfg.batch_size
        logger.debug(
            "_MMLULocalEvaluator: %d items, batch_size=%d", len(items), cfg.batch_size
        )

    def _format(self, item: dict[str, Any]) -> str:
        choices = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(item["choices"])
        )
        return f"Question: {item['question']}\n{choices}\nAnswer:"

    def evaluate(self, model: torch.nn.Module, device: torch.device) -> float:
        """Run MMLU accuracy evaluation with the given model.

        Args:
            model: The full Llama model (with compression hooks installed).
            device: Device the model lives on.

        Returns:
            Top-1 accuracy in [0, 1].
        """
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for i in range(0, len(self._items), self._batch_size):
                batch_items = self._items[i : i + self._batch_size]
                prompts = [self._format(it) for it in batch_items]
                labels: list[int] = [it["answer"] for it in batch_items]
                encoded = self._tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                input_ids = encoded.input_ids.to(device)
                attention_mask = (
                    encoded.attention_mask.to(device) if len(batch_items) > 1 else None
                )
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_logits = outputs.logits[:, -1, :]  # [B, V]
                cand = last_logits[:, self._answer_token_ids]  # [B, 4]
                predicted: list[int] = cand.argmax(dim=-1).tolist()
                for pred, label in zip(predicted, labels, strict=False):
                    correct += int(pred == label)
                    total += 1
        return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def _build_compress_fns(
    pipeline_name: str,
    flow: list[str],
    links: list[OptLinkConfig],
    mapper: Any,
) -> list[Callable[[torch.Tensor, float], torch.Tensor]]:
    """Build one compress_fn per inter-node link for a pipeline.

    Args:
        pipeline_name: Pipeline identifier (used as pipeline_id in the mapper).
        flow: Ordered node IDs for this pipeline.
        links: All link configs from the experiment.
        mapper: Initialised ``CompressionMapper`` for the experiment.

    Returns:
        List of callables, one per link, in flow order.
    """
    from framework.optimizer.compression_simulator import (  # noqa: PLC0415
        build_compress_fn,
        identity_compress_fn,
    )

    link_map = {(lk.from_node, lk.to_node): lk for lk in links}
    fns: list[Callable[[torch.Tensor, float], torch.Tensor]] = []
    for i in range(len(flow) - 1):
        key = (flow[i], flow[i + 1])
        lk = link_map.get(key)
        if lk is None:
            logger.warning(
                "No link config found for %s→%s in pipeline '%s'; "
                "using identity (no compression)",
                flow[i],
                flow[i + 1],
                pipeline_name,
            )
            fns.append(identity_compress_fn)
        else:
            fns.append(build_compress_fn(mapper, lk.link_id, pipeline_name))
    return fns


def build_simulations(
    exp: GeneratedOptExperimentConfig,
    stein_cfg: SteinOracleConfig | None = None,
    mapper: Any | None = None,
    dataset_override: dict[str, DatasetConfig] | None = None,
) -> dict[str, Any]:
    """Build simulation pipeline objects for all pipelines with ``simulation_path`` set.

    Only pipelines with a non-None ``simulation_path`` are built.  Pipelines
    without one are silently skipped — ``build_inference_tasks`` will install
    dummy callables for them.

    Args:
        exp: Resolved experiment config.
        stein_cfg: Stein oracle config, used to size the fast evaluator for Llama
            (``n_fast_samples``).  If ``None``, the full dataset config is used
            for both fast and full evaluators.
        mapper: Initialised ``CompressionMapper`` for the experiment.  Used to
            build per-link ``compress_fns`` that match the deployed compressor
            scheme.  If ``None``, simulation pipelines default to per-sample
            top-k for all links.
        dataset_override: When provided, replaces ``exp.datasets`` for dataset
            lookups.  Use this to pass accuracy-model sweep datasets or Stein
            oracle datasets so that each evaluation role uses its own data
            slice.  For Llama, the full_evaluator is capped at
            ``cfg.max_samples`` (not None) when an override is supplied.

    Returns:
        Dict mapping ``pipeline.name → simulation object``
        (``SimulatedResNetPipeline`` or ``SimulatedLlamaPipeline``).
    """
    from models.llama.simulation.pipeline import (  # noqa: PLC0415
        SimulatedLlamaPipeline,
    )
    from models.resnet.simulation.pipeline import (  # noqa: PLC0415
        SimulatedResNetPipeline,
    )

    datasets = dataset_override if dataset_override is not None else exp.datasets
    result: dict[str, Any] = {}

    for pipeline in exp.pipelines:
        if pipeline.simulation_path is None:
            continue

        model_key = pipeline.model.lower()

        compress_fns = (
            _build_compress_fns(pipeline.name, pipeline.flow, exp.links, mapper)
            if mapper is not None
            else None
        )

        if model_key in _MODEL_RESNET:
            cfg = datasets.get("resnet")
            if cfg is None:
                logger.warning(
                    "Pipeline '%s' has simulation_path but no 'resnet' dataset config",
                    pipeline.name,
                )
                continue
            loader = _build_cifar10_loader(cfg)
            sim = SimulatedResNetPipeline(
                checkpoint_path=Path(pipeline.simulation_path),
                partitions=pipeline.partitions,
                flow=pipeline.flow,
                test_loader=loader,
                compress_fns=compress_fns,
            )
            result[pipeline.name] = sim
            logger.info(
                "Built SimulatedResNetPipeline for pipeline '%s' from '%s'",
                pipeline.name,
                pipeline.simulation_path,
            )

        elif model_key in _MODEL_LLAMA:
            cfg = datasets.get("llama")
            if cfg is None:
                logger.warning(
                    "Pipeline '%s' has simulation_path but no 'llama' dataset config",
                    pipeline.name,
                )
                continue
            tokenizer_path = cfg.tokenizer_path
            if tokenizer_path is None:
                logger.warning(
                    "Llama dataset config missing tokenizer_path; "
                    "skipping Stein simulation for pipeline '%s'",
                    pipeline.name,
                )
                continue

            if stein_cfg is not None:
                fast_max = stein_cfg.n_fast_samples
            elif dataset_override is not None:
                # No Stein oracle (e.g. accuracy model sweep) but a dataset
                # override is active — cap the fast evaluator at cfg.max_samples
                # so sim.accuracy() doesn't run the entire dataset on every call.
                fast_max = cfg.max_samples
            else:
                fast_max = None
            # When a dataset override is in effect, cap the full evaluator at
            # cfg.max_samples so we don't accidentally run the entire dataset on
            # every oracle call.
            full_max = cfg.max_samples if dataset_override is not None else None

            is_wikitext = cfg.name.lower() in _DATASET_WIKITEXT

            if is_wikitext:
                from transformers import AutoTokenizer  # noqa: PLC0415

                from framework.optimizer.evaluators import (  # noqa: PLC0415
                    WikiTextPerplexityEvaluator,
                )

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                fast_evaluator: Any = WikiTextPerplexityEvaluator(
                    tokenizer,
                    dataset_path=cfg.path,
                    n_samples=fast_max,
                )
                full_evaluator: Any = WikiTextPerplexityEvaluator(
                    tokenizer,
                    dataset_path=cfg.path,
                    n_samples=full_max,
                )
            else:
                fast_evaluator = _MMLULocalEvaluator(
                    cfg, tokenizer_path, max_items=fast_max
                )
                full_evaluator = _MMLULocalEvaluator(
                    cfg, tokenizer_path, max_items=full_max
                )

            sim = SimulatedLlamaPipeline(
                model_name=pipeline.simulation_path,
                partitions=pipeline.partitions,
                flow=pipeline.flow,
                fast_evaluator=fast_evaluator,
                full_evaluator=full_evaluator,
                compress_fns=compress_fns,
            )

            if is_wikitext:
                # Measure uncompressed baseline NLL so the evaluator can
                # normalise future evaluations to [0, 1] via exp(-nll/baseline).
                # set_eta([1.0, ...]) installs identity hooks (no compression).
                sim.set_eta(torch.ones(sim.n_links))
                fast_evaluator.set_baseline(sim.model, sim.device)
                if full_evaluator is not fast_evaluator:
                    full_evaluator.set_baseline(sim.model, sim.device)
                logger.info("WikiText baseline set for pipeline '%s'", pipeline.name)

            result[pipeline.name] = sim
            logger.info(
                "Built SimulatedLlamaPipeline for pipeline '%s' from '%s'",
                pipeline.name,
                pipeline.simulation_path,
            )

        else:
            logger.warning(
                "Unknown model type '%s' for pipeline '%s'; skipping Stein simulation",
                pipeline.model,
                pipeline.name,
            )

    return result
