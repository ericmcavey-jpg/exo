"""SeedVR2 restoration and upscaling through mflux (installed with exo's mlx extra)."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Literal, final

import psutil

SeedVR2Model = Literal["seedvr2-3b", "seedvr2-7b"]


class InsufficientMemoryError(RuntimeError):
    pass


def memory_shortfall_message(
    available_bytes: int, minimum_free_gigabytes: float
) -> str | None:
    """Why SeedVR2 should not start now, or None when there is enough free memory."""
    available_gigabytes = available_bytes / 1024**3
    if minimum_free_gigabytes <= 0 or available_gigabytes >= minimum_free_gigabytes:
        return None
    return (
        f"Only {available_gigabytes:.0f} GB of memory is free, below the "
        f"{minimum_free_gigabytes:.0f} GB this run requires (--min-free-memory-gb). "
        "Running SeedVR2 under memory pressure can stall the GPU and force the Mac to "
        "restart. Unload large models in exo first, or retry with --low-ram and a "
        "smaller --megapixels. Pass --min-free-memory-gb 0 to skip this check."
    )


@final
class SeedVR2Restorer:
    """Runs mflux's SeedVR2 upscaler in a subprocess so its memory is freed after each image."""

    def __init__(
        self,
        *,
        model: SeedVR2Model = "seedvr2-3b",
        quantize: int | None = None,
        low_ram: bool = False,
        minimum_free_gigabytes: float = 32.0,
    ) -> None:
        self._model: SeedVR2Model = model
        self._quantize = quantize
        self._low_ram = low_ram
        self._minimum_free_gigabytes = minimum_free_gigabytes

    def command(
        self,
        *,
        source: Path,
        destination: Path,
        short_edge: int,
        softness: float,
        seed: int,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "mflux.models.seedvr2.cli.seedvr2_upscale",
            "--model",
            self._model,
            "--image-path",
            str(source),
            "--resolution",
            str(short_edge),
            "--softness",
            str(softness),
            "--seed",
            str(seed),
            "--output",
            str(destination),
        ]
        if self._quantize is not None:
            command.extend(["--quantize", str(self._quantize)])
        if self._low_ram:
            command.append("--low-ram")
        return command

    def restore(
        self,
        *,
        source: Path,
        destination: Path,
        short_edge: int,
        softness: float,
        seed: int,
    ) -> Path:
        if importlib.util.find_spec("mflux") is None:
            raise RuntimeError(
                "mflux is not installed in this environment. Run the toolkit through "
                "scripts/likeness, which adds it."
            )
        shortfall = memory_shortfall_message(
            psutil.virtual_memory().available, self._minimum_free_gigabytes
        )
        if shortfall is not None:
            raise InsufficientMemoryError(shortfall)
        # mflux never overwrites: an existing file makes it save as "<name>_1.png",
        # which would leave this run reading a stale image from an earlier attempt.
        destination.unlink(missing_ok=True)
        subprocess.run(
            self.command(
                source=source,
                destination=destination,
                short_edge=short_edge,
                softness=softness,
                seed=seed,
            ),
            check=True,
        )
        if not destination.exists():
            raise RuntimeError(f"SeedVR2 finished without writing {destination}")
        return destination
