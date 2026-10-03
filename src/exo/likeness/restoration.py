"""SeedVR2 restoration and upscaling through mflux (installed with exo's mlx extra)."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Literal, final

SeedVR2Model = Literal["seedvr2-3b", "seedvr2-7b"]


@final
class SeedVR2Restorer:
    """Runs mflux's SeedVR2 upscaler in a subprocess so its memory is freed after each image."""

    def __init__(
        self,
        *,
        model: SeedVR2Model = "seedvr2-3b",
        quantize: int | None = None,
        low_ram: bool = False,
    ) -> None:
        self._model: SeedVR2Model = model
        self._quantize = quantize
        self._low_ram = low_ram

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
