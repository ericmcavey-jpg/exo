# pyright: reportUnusedFunction=false, reportAny=false
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.common import CommandId


def _make_api() -> Any:
    """Create a minimal API instance with cancel route and error handler."""

    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._cancelled_commands = set()  # pyright: ignore[reportPrivateUsage]
    api.state = SimpleNamespace(tasks={})  # pyright: ignore[reportAttributeAccessIssue]
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/v1/cancel/{command_id}")(api.cancel_command)
    return api


def test_cancel_nonexistent_command_returns_404() -> None:
    """Cancel for an unknown command_id returns 404 in OpenAI error format."""
    api = _make_api()
    client = TestClient(api.app)

    response = client.post("/v1/cancel/nonexistent-id")
    assert response.status_code == 404
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Command not found or already completed"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404


def test_cancel_active_text_generation() -> None:
    """Cancel an active text generation command: returns 200, sender.close() called."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("text-cmd-123")
    sender = MagicMock()
    api._text_generation_queues[cid] = sender
    api.state.tasks = {"task": SimpleNamespace(command_id=cid)}

    async def acknowledge(_command: Any) -> None:
        api.state.tasks = {}

    api._send = AsyncMock(side_effect=acknowledge)

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancellation acknowledged by all runners."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()
    task_cancelled = api._send.call_args[0][0]
    assert task_cancelled.cancelled_command_id == cid


def test_cancel_active_image_generation() -> None:
    """Cancel an active image generation command: returns 200, sender.close() called."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("img-cmd-456")
    sender = MagicMock()
    api._image_generation_queues[cid] = sender
    api.state.tasks = {"task": SimpleNamespace(command_id=cid)}

    async def acknowledge(_command: Any) -> None:
        api.state.tasks = {}

    api._send = AsyncMock(side_effect=acknowledge)

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancellation acknowledged by all runners."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()
    task_cancelled = api._send.call_args[0][0]
    assert task_cancelled.cancelled_command_id == cid


@pytest.mark.anyio
async def test_cancelled_stream_does_not_race_task_finished_ahead_of_ack() -> None:
    api = _make_api()
    cid = CommandId("cancelled-stream")
    api._cancelled_commands.add(cid)
    done = anyio.Event()

    async def consume() -> None:
        async for _chunk in api._token_chunk_stream(cid):
            pass
        done.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        for _ in range(100):
            if cid in api._text_generation_queues:
                break
            await anyio.sleep(0.001)
        else:
            raise AssertionError("stream queue was not created")
        api._text_generation_queues[cid].close()
        await done.wait()
        tg.cancel_scope.cancel()

    api._send.assert_not_awaited()
