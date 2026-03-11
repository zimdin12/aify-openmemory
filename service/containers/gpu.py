"""
GPU allocation tracker.

Tracks which containers are assigned to which GPU devices and enforces
exclusive locks and memory fraction limits.
"""

import logging
from dataclasses import dataclass, field

from .models import GPUConfig

logger = logging.getLogger(__name__)


@dataclass
class DeviceState:
    active_containers: dict[str, float] = field(default_factory=dict)  # name -> fraction
    exclusive_lock: str | None = None

    @property
    def total_memory_fraction(self) -> float:
        return sum(self.active_containers.values())


class GPUAllocator:
    def __init__(self):
        self.devices: dict[str, DeviceState] = {}

    def _ensure_device(self, device_id: str) -> DeviceState:
        if device_id not in self.devices:
            self.devices[device_id] = DeviceState()
        return self.devices[device_id]

    def can_allocate(self, name: str, gpu: GPUConfig) -> tuple[bool, str]:
        """Check if GPU resources are available. Returns (ok, reason)."""
        if not gpu.device_ids:
            return True, ""

        for device_id in gpu.device_ids:
            dev = self._ensure_device(device_id)

            if dev.exclusive_lock is not None and dev.exclusive_lock != name:
                return False, f"GPU {device_id} exclusively locked by '{dev.exclusive_lock}'"

            others = {k for k in dev.active_containers if k != name}
            if gpu.exclusive and others:
                return False, f"GPU {device_id} in use by [{', '.join(others)}], cannot get exclusive access"

            current = dev.total_memory_fraction
            if name in dev.active_containers:
                current -= dev.active_containers[name]  # Don't double-count re-allocation
            if current + gpu.memory_fraction > 1.05:  # 5% tolerance for rounding
                return False, (
                    f"GPU {device_id}: {current:.0%} used + {gpu.memory_fraction:.0%} requested > 100%"
                )

        return True, ""

    def allocate(self, name: str, gpu: GPUConfig):
        for device_id in gpu.device_ids:
            dev = self._ensure_device(device_id)
            dev.active_containers[name] = gpu.memory_fraction
            if gpu.exclusive:
                dev.exclusive_lock = name
            logger.info(f"GPU {device_id}: +{name} ({dev.total_memory_fraction:.0%} used)")

    def release_with_fraction(self, name: str, gpu: GPUConfig):
        """Release GPU resources for a container."""
        for device_id in gpu.device_ids:
            dev = self._ensure_device(device_id)
            dev.active_containers.pop(name, None)
            if dev.exclusive_lock == name:
                dev.exclusive_lock = None
            logger.info(f"GPU {device_id}: -{name} ({dev.total_memory_fraction:.0%} used)")

    def get_status(self) -> dict:
        return {
            device_id: {
                "active_containers": dict(dev.active_containers),
                "total_memory_fraction": round(dev.total_memory_fraction, 2),
                "exclusive_lock": dev.exclusive_lock,
            }
            for device_id, dev in self.devices.items()
        }
