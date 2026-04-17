"""DataCollector — produce (prompt, response, reward) triples for custom training loops.

For users who own their own training stack and want ISOPro to supply the
data rather than run the training loop itself. Outputs can be serialised
to JSONL or converted to a HuggingFace Dataset.

Typical usage:
    collector = DataCollector(env, backend)
    dataset = collector.collect(n_episodes=500)
    dataset.to_jsonl("data/training.jsonl")
    # or: hf_ds = dataset.to_huggingface_dataset()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..backends.base import GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """A single (prompt, response, reward) training sample.

    Attributes:
        task_id: Unique task identifier for deduplication / analysis.
        difficulty: DifficultyLevel value string.
        category: Task category string.
        prompt: The full prompt sent to the model.
        response: The model's raw text response.
        reward: Scalar reward from the environment's deterministic scorer.
        latency_ms: Wall-clock time for the backend.generate() call.
        metadata: Arbitrary per-task extras from Task.metadata.
    """

    task_id: str
    difficulty: str
    category: str
    prompt: str
    response: str
    reward: float
    latency_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class CollectedDataset:
    """Container for a collection of Samples with serialisation helpers.

    Attributes:
        samples: Ordered list of Sample objects.
        environment_name: Class name of the environment that produced samples.
        model_id: Model identifier.
        n_episodes: Total episodes collected.
        mean_reward: Mean reward across all samples.
        collection_duration_s: Wall-clock time for the full collect() call.
    """

    samples: list[Sample]
    environment_name: str
    model_id: str
    n_episodes: int
    mean_reward: float
    collection_duration_s: float

    def to_jsonl(self, path: str) -> None:
        """Write all samples to a JSONL file (one JSON object per line).

        Args:
            path: Destination file path. Created or overwritten.
        """
        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(asdict(sample)) + "\n")
        logger.info("Wrote %d samples to %s", len(self.samples), path)

    def to_huggingface_dataset(self):
        """Convert to a HuggingFace Dataset object.

        Requires the ``datasets`` package to be installed.

        Returns:
            A ``datasets.Dataset`` instance with one row per sample.

        Raises:
            ImportError: If ``datasets`` is not installed.
        """
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "Install datasets: pip install datasets"
            ) from exc

        rows = [asdict(s) for s in self.samples]
        return Dataset.from_list(rows)

    def filter_by_reward(self, min_reward: float = 0.5) -> "CollectedDataset":
        """Return a new dataset containing only samples above a reward threshold.

        Useful for filtering out low-quality training signal before fine-tuning.

        Args:
            min_reward: Minimum reward to include. Defaults to 0.5.

        Returns:
            A new CollectedDataset with filtered samples.
        """
        filtered = [s for s in self.samples if s.reward >= min_reward]
        mean_r = sum(s.reward for s in filtered) / len(filtered) if filtered else 0.0
        return CollectedDataset(
            samples=filtered,
            environment_name=self.environment_name,
            model_id=self.model_id,
            n_episodes=len(filtered),
            mean_reward=mean_r,
            collection_duration_s=self.collection_duration_s,
        )


class DataCollector:
    """Collect (prompt, response, reward) samples from an environment + backend.

    Args:
        env: Any task-based BaseEnvironment with generate_task() and score().
        backend: Any ModelBackend instance.
        config: Optional GenerationConfig. Defaults to 512 max tokens.
    """

    def __init__(self, env, backend, config: Optional[GenerationConfig] = None) -> None:
        self._env = env
        self._backend = backend
        self._config = config or GenerationConfig(max_new_tokens=512)

    def collect(
        self,
        n_episodes: int = 500,
        curriculum=None,
    ) -> CollectedDataset:
        """Run n_episodes and collect samples.

        Args:
            n_episodes: Number of episodes to run.
            curriculum: Optional CurriculumScheduler. If provided, task
                difficulty is managed automatically per episode.

        Returns:
            CollectedDataset with all samples and summary statistics.
        """
        model_info = self._backend.get_model_info()
        start = time.perf_counter()
        samples: list[Sample] = []

        for _ in range(n_episodes):
            if curriculum is not None:
                task = curriculum.next_task()
            else:
                task = self._env.generate_task()

            t0 = time.perf_counter()
            result = self._backend.generate(task.prompt, self._config)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            reward = self._env.score(task, result.text)

            if curriculum is not None:
                curriculum.update(reward)

            samples.append(Sample(
                task_id=task.task_id,
                difficulty=task.difficulty.value,
                category=task.category,
                prompt=task.prompt,
                response=result.text,
                reward=reward,
                latency_ms=round(latency_ms, 2),
                metadata=task.metadata,
            ))

        duration = time.perf_counter() - start
        mean_r = sum(s.reward for s in samples) / len(samples) if samples else 0.0

        logger.info(
            "Collected %d samples from %s | mean_reward=%.3f | duration=%.1fs",
            len(samples),
            self._env.__class__.__name__,
            mean_r,
            duration,
        )

        return CollectedDataset(
            samples=samples,
            environment_name=self._env.__class__.__name__,
            model_id=model_info.model_id,
            n_episodes=len(samples),
            mean_reward=round(mean_r, 4),
            collection_duration_s=round(duration, 2),
        )
