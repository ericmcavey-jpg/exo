import base64
import json
from pathlib import Path

import httpx
import pytest

from exo.likeness.exo_client import (
    ExoImageEditor,
    ExoRequestError,
    _ClusterState,  # pyright: ignore[reportPrivateUsage]
    runner_states_for_model,
)
from exo.likeness.recipes import EditStep

MODEL = "exolabs/Qwen-Image-Edit-2509-8bit"

STATE: dict[str, object] = {
    "instances": {
        "instance-1": {
            "MlxRingInstance": {
                "instanceId": "instance-1",
                "shardAssignments": {
                    "modelId": MODEL,
                    "runnerToShard": {"runner-1": {}, "runner-2": {}},
                    "nodeToRunner": {},
                },
            }
        }
    },
    "runners": {
        "runner-1": {"RunnerReady": {}},
        "runner-2": {"RunnerLoading": {"layersLoaded": 3, "totalLayers": 60}},
    },
    "topology": {},
}


def test_runner_states_for_model():
    state = _ClusterState.model_validate_json(json.dumps(STATE))
    assert runner_states_for_model(state, MODEL) == ["RunnerReady", "RunnerLoading"]
    assert runner_states_for_model(state, "other/model") is None


def test_edit_sends_multipart_request_and_saves_the_image(tmp_path: Path):
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("latin-1")
        seen["path"] = request.url.path
        seen["body"] = body
        image = base64.b64encode(b"PNGDATA").decode()
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": image}]})

    editor = ExoImageEditor(
        base_url="http://exo", model_id=MODEL, transport=httpx.MockTransport(handler)
    )
    source = tmp_path / "in.png"
    source.write_bytes(b"input")
    destination = editor.edit(
        source=source,
        step=EditStep(instruction="retouch", guidance=4.0),
        seed=11,
        destination=tmp_path / "out.png",
    )
    assert destination.read_bytes() == b"PNGDATA"
    assert seen["path"] == "/v1/images/edits"
    assert '"seed": 11' in seen["body"] and '"guidance": 4.0' in seen["body"]
    assert 'name="input_fidelity"' in seen["body"]


def test_missing_model_is_reported_without_launching():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"instances": {}, "runners": {}})

    editor = ExoImageEditor(
        base_url="http://exo", model_id=MODEL, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ExoRequestError, match="not running in exo"):
        editor.ensure_model_running(
            launch_if_missing=False, ready_timeout_seconds=1, on_progress=print
        )
