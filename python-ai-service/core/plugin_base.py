from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PluginOutput:
    plugin_name: str
    score: float = 0.0
    objects: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)


class BasePlugin:
    name: str = "base"
    requires_gpu: bool = False
    priority: int = 100
    version: str = "1.0.0"

    _loaded: bool = False
    _load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        """Synchronously load models. Called lazily or during background preload."""
        self._loaded = True

    def unload(self) -> None:
        """Release resources during shutdown or eviction."""
        self._loaded = False
        self._load_error = None

    def ensure_loaded(self) -> None:
        """Called at start of analyze(). Loads if not yet loaded."""
        if not self._loaded:
            try:
                self.load()
            except Exception as e:
                self._load_error = str(e)
                raise
        if self._load_error:
            raise RuntimeError(f"Plugin {self.name} failed to load: {self._load_error}")

    def analyze(self, image_path: str, image: np.ndarray | None = None) -> PluginOutput:
        raise NotImplementedError
