"""
Pydantic models for container definitions, state, and GPU config.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ContainerStatus(str, Enum):
    DEFINED = "defined"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class GPUConfig(BaseModel):
    device_ids: list[str] = Field(default_factory=list, description="NVIDIA device IDs, e.g. ['0']")
    memory_fraction: float = Field(1.0, description="Fraction of GPU memory this container needs (for scheduling)")
    exclusive: bool = Field(False, description="If true, no other container can share this GPU device")


class HealthCheckConfig(BaseModel):
    endpoint: str = "/health"
    interval_seconds: int = 10
    timeout_seconds: int = 5
    retries: int = 3


class ResourceConfig(BaseModel):
    cpu_limit: str = "4"
    memory_limit: str = "8g"


class ContainerDefinition(BaseModel):
    """Definition of a managed sub-container."""
    model_config = ConfigDict(extra="ignore")  # Allow _comment and unknown fields

    image: str
    internal_port: int = 8080
    command: list[str] = Field(default_factory=list)
    volumes: dict[str, str] = Field(default_factory=dict, description="Named volume -> mount path")
    environment: dict[str, str] = Field(default_factory=dict)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    idle_timeout_seconds: int = Field(300, description="Auto-stop after N seconds idle. 0 = never")
    startup_timeout_seconds: int = 120
    auto_start: bool = False
    labels: dict[str, str] = Field(default_factory=dict)
    group: str = ""
    shared_with: str = Field("", description="Use another container's URL instead of starting own")
    provides_url_as: str = Field("", description="Env var name to inject this container's URL into dependent services")
    ports: dict[str, int] = Field(default_factory=dict, description="Host port -> container port mappings")


class ContainerState(BaseModel):
    """Runtime state of a managed container."""
    name: str
    status: ContainerStatus = ContainerStatus.DEFINED
    container_id: Optional[str] = None
    container_hostname: Optional[str] = None
    internal_port: int = 8080
    started_at: Optional[datetime] = None
    last_request_at: Optional[datetime] = None
    consecutive_health_failures: int = 0
    error_message: Optional[str] = None

    @property
    def idle_seconds(self) -> float:
        if self.last_request_at is None:
            return 0.0
        return (datetime.now(timezone.utc) - self.last_request_at).total_seconds()

    @property
    def internal_url(self) -> Optional[str]:
        if self.container_hostname and self.internal_port:
            return f"http://{self.container_hostname}:{self.internal_port}"
        return None
