import importlib.util
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from exo.likeness import restoration
from exo.likeness.restoration import (
    InsufficientMemoryError,
    SeedVR2Restorer,
    memory_shortfall_message,
)

GIGABYTE = 1024**3


def test_memory_shortfall_message():
    assert memory_shortfall_message(64 * GIGABYTE, 32) is None
    assert memory_shortfall_message(8 * GIGABYTE, 0) is None
    message = memory_shortfall_message(8 * GIGABYTE, 32)
    assert message is not None and "Only 8 GB" in message


def mflux_is_installed(name: str) -> object:
    return object()


def fake_environment(monkeypatch: pytest.MonkeyPatch, available_gigabytes: int) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", mflux_is_installed)
    monkeypatch.setattr(
        restoration.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=available_gigabytes * GIGABYTE),
    )


def test_stale_output_is_removed_before_seedvr2_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake_environment(monkeypatch, available_gigabytes=256)
    destination = tmp_path / "01-restore.png"
    destination.write_text("stale image from an interrupted run")

    def fake_run(command: Sequence[str], check: bool) -> None:
        output = Path(command[command.index("--output") + 1])
        assert not output.exists()  # otherwise mflux would write 01-restore_1.png
        output.write_text("fresh")

    monkeypatch.setattr(subprocess, "run", fake_run)
    SeedVR2Restorer().restore(
        source=tmp_path / "in.png",
        destination=destination,
        short_edge=1024,
        softness=0.0,
        seed=1,
    )
    assert destination.read_text() == "fresh"


def test_refuses_to_start_when_memory_is_low(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake_environment(monkeypatch, available_gigabytes=10)
    with pytest.raises(InsufficientMemoryError):
        SeedVR2Restorer(minimum_free_gigabytes=32).restore(
            source=tmp_path / "in.png",
            destination=tmp_path / "out.png",
            short_edge=1024,
            softness=0.0,
            seed=1,
        )
