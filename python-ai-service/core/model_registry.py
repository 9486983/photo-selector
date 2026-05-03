from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class ModelInfo:
    name: str
    model_type: str
    device: DeviceType
    memory_mb: float = 0.0
    loaded: bool = False
    ref_count: int = 0
    model_path: str | None = None


class ModelRegistry:
    _models: dict[str, Any] = {}
    _model_info: dict[str, ModelInfo] = {}
    _access_order: OrderedDict[str, int] = OrderedDict()
    _gpu_memory_limit_mb: float = 8 * 1024

    @classmethod
    def set_gpu_memory_limit(cls, limit_mb: float) -> None:
        cls._gpu_memory_limit_mb = limit_mb

    @classmethod
    def register(
        cls,
        name: str,
        model_type: str,
        device: DeviceType,
        memory_mb: float = 0,
        model_path: str | None = None,
    ) -> None:
        cls._model_info[name] = ModelInfo(
            name=name,
            model_type=model_type,
            device=device,
            memory_mb=memory_mb,
            model_path=model_path,
        )
        cls._access_order[name] = 0
        logger.debug("Registered model: %s (type=%s, device=%s, memory=%dMB)", name, model_type, device.value, memory_mb)

    @classmethod
    def get(cls, name: str) -> Any | None:
        cls._access_order[name] = cls._access_order.get(name, 0) + 1
        return cls._models.get(name)

    @classmethod
    async def ensure_loaded(cls, name: str, loader_fn: Callable[[], Any]) -> Any:
        existing = cls._models.get(name)
        if existing is not None:
            cls._model_info[name].ref_count += 1
            cls._access_order[name] = cls._access_order.get(name, 0) + 1
            return existing

        info = cls._model_info.get(name)
        if info is None:
            raise ValueError(f"Model '{name}' is not registered")

        if info.device == DeviceType.CUDA:
            await cls._ensure_gpu_headroom(info.memory_mb)

        logger.info("Loading model: %s", name)
        model = await loader_fn() if hasattr(loader_fn, "__await__") else loader_fn()
        cls._models[name] = model
        info.loaded = True
        info.ref_count = 1
        cls._access_order[name] = cls._access_order.get(name, 0) + 1
        logger.info("Model %s loaded", name)
        return model

    @classmethod
    async def unload(cls, name: str) -> None:
        info = cls._model_info.get(name)
        if info is None:
            return
        if info.ref_count > 0:
            info.ref_count -= 1
        if info.ref_count <= 0:
            cls._models.pop(name, None)
            info.loaded = False
            logger.info("Model %s unloaded", name)
            cls._try_gc()

    @classmethod
    def unload_all(cls) -> None:
        cls._models.clear()
        for info in cls._model_info.values():
            info.loaded = False
            info.ref_count = 0
        cls._access_order.clear()
        cls._try_gc()

    @classmethod
    def get_status(cls) -> list[dict]:
        return [
            {
                "name": info.name,
                "model_type": info.model_type,
                "device": info.device.value,
                "memory_mb": info.memory_mb,
                "loaded": info.loaded,
                "ref_count": info.ref_count,
            }
            for info in cls._model_info.values()
        ]

    @classmethod
    def total_gpu_memory_mb(cls) -> float:
        return sum(
            info.memory_mb
            for info in cls._model_info.values()
            if info.loaded and info.device == DeviceType.CUDA
        )

    @classmethod
    async def _ensure_gpu_headroom(cls, needed_mb: float) -> None:
        available = await cls._get_available_gpu_memory_mb()
        if available >= needed_mb * 1.2:
            return
        for _ in range(10):
            if not await cls._evict_lru_gpu_model():
                break
            available = await cls._get_available_gpu_memory_mb()
            if available >= needed_mb * 1.2:
                return

    @classmethod
    async def _get_available_gpu_memory_mb(cls) -> float:
        try:
            import torch

            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                return free / (1024 * 1024)
        except Exception:
            pass
        return cls._gpu_memory_limit_mb - cls.total_gpu_memory_mb()

    @classmethod
    async def _evict_lru_gpu_model(cls) -> bool:
        evictable = [
            (name, info)
            for name, info in cls._model_info.items()
            if info.loaded and info.device == DeviceType.CUDA and info.ref_count == 0
        ]
        if not evictable:
            return False
        evictable.sort(key=lambda x: cls._access_order.get(x[0], 0))
        name = evictable[0][0]
        logger.info("Evicting GPU model: %s", name)
        cls._models.pop(name, None)
        cls._model_info[name].loaded = False
        cls._try_gc()
        return True

    @classmethod
    def _try_gc(cls) -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
