"""Edit images through a running exo node (POST /v1/images/edits)."""

import base64
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final, final

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from exo.likeness.recipes import EditStep

DEFAULT_EDIT_MODEL: Final = "exolabs/Qwen-Image-Edit-2509-8bit"
READY_RUNNER_STATES: Final = frozenset({"RunnerReady", "RunnerRunning"})


class ExoRequestError(RuntimeError):
    pass


class _ExoStateModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, extra="ignore", frozen=True)


class _ShardAssignments(_ExoStateModel):
    model_id: str
    runner_to_shard: dict[str, object] = Field(default_factory=dict)


class _InstanceBody(_ExoStateModel):
    shard_assignments: _ShardAssignments


class _ClusterState(_ExoStateModel):
    instances: dict[str, dict[str, _InstanceBody]] = Field(default_factory=dict)
    runners: dict[str, dict[str, object]] = Field(default_factory=dict)


class _GeneratedImage(BaseModel):
    b64_json: str | None = None


class _ImageResponse(BaseModel):
    data: list[_GeneratedImage]


def runner_states_for_model(state: _ClusterState, model_id: str) -> list[str] | None:
    """Status names of the runners serving `model_id`, or None if no instance exists."""
    for tagged_instance in state.instances.values():
        for body in tagged_instance.values():
            if body.shard_assignments.model_id != model_id:
                continue
            return [
                next(iter(state.runners.get(runner_id, {"RunnerIdle": {}})))
                for runner_id in body.shard_assignments.runner_to_shard
            ]
    return None


@final
class ExoImageEditor:
    def __init__(
        self,
        *,
        base_url: str,
        model_id: str = DEFAULT_EDIT_MODEL,
        timeout_seconds: float = 1800.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_id = model_id
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout_seconds, transport=transport
        )

    def _state(self) -> _ClusterState:
        response = self._client.get("/state")
        self._raise_for_status(response)
        return _ClusterState.model_validate_json(response.content)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise ExoRequestError(
                f"exo returned HTTP {response.status_code} for {response.request.url.path}: "
                f"{response.text[:300]}"
            )

    def ensure_model_running(
        self,
        *,
        launch_if_missing: bool,
        ready_timeout_seconds: float,
        on_progress: Callable[[str], None],
    ) -> None:
        states = runner_states_for_model(self._state(), self.model_id)
        if states is None:
            if not launch_if_missing:
                raise ExoRequestError(
                    f"{self.model_id} is not running in exo. Launch it from the dashboard "
                    "(with EXO_ENABLE_IMAGE_MODELS=true), or pass --launch-model."
                )
            on_progress(
                f"Launching {self.model_id} in exo (downloads it first if needed)..."
            )
            response = self._client.post(
                "/place_instance", json={"model_id": self.model_id}
            )
            self._raise_for_status(response)

        deadline = time.monotonic() + ready_timeout_seconds
        last_reported: list[str] | None = None
        while time.monotonic() < deadline:
            states = runner_states_for_model(self._state(), self.model_id)
            if states and "RunnerFailed" in states:
                raise ExoRequestError(
                    f"exo failed to load {self.model_id}; check the exo logs."
                )
            if states and all(state in READY_RUNNER_STATES for state in states):
                return
            if states != last_reported:
                on_progress(f"Waiting for {self.model_id}: {states or 'placing'}")
                last_reported = states
            time.sleep(5.0)
        raise ExoRequestError(
            f"{self.model_id} was not ready after {ready_timeout_seconds:.0f}s."
        )

    def edit(
        self, *, source: Path, step: EditStep, seed: int, destination: Path
    ) -> Path:
        advanced_params: dict[str, int | float] = {"seed": seed}
        if step.guidance is not None:
            advanced_params["guidance"] = step.guidance
        response = self._client.post(
            "/v1/images/edits",
            data={
                "prompt": step.instruction,
                "model": self.model_id,
                "n": "1",
                "input_fidelity": step.input_fidelity,
                "quality": step.quality,
                "output_format": "png",
                "response_format": "b64_json",
                "advanced_params": json.dumps(advanced_params),
            },
            files={"image": (source.name, source.read_bytes(), "image/png")},
        )
        self._raise_for_status(response)
        images = _ImageResponse.model_validate_json(response.content).data
        if not images or images[0].b64_json is None:
            raise ExoRequestError("exo returned no image for the edit request.")
        destination.write_bytes(base64.b64decode(images[0].b64_json))
        return destination
