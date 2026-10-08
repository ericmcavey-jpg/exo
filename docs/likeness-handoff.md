# Handoff: likeness toolkit (as of 2026-10-08)

For a Claude Code session running locally on m3a. Usage is in `docs/likeness.md`.

## Goal

Build a realistic, flattering AI likeness of the owner (Eric) from his own Photos
library, for staging in images and videos: curate photos, enhance them realistically
(plus optional stylized looks), train a likeness LoRA, later narrate with a cloned
voice. The owner has confirmed it is his own library and likeness.

## Where things are

- **Code:** `src/exo/likeness/`, launcher `scripts/likeness`, docs
  `docs/likeness.md`, on branch `claude/busy-ramanujan-iomjoc` of
  `ericmcavey-jpg/exo`. See `git log` for the latest commit.
- **On m3a:** the toolkit is cloned at `~/exo-likeness` (update with
  `git -C ~/exo-likeness pull`); task data lives in `~/.exo/likeness/me/`.
- **Photos library:** `/Volumes/PHOTOS/Photos Library.photoslibrary`, last
  updated 2026-05-16. m3a has no system Photos library. The person is tagged "Eric".
- **Checks:** 50 tests pass (`uv run pytest src/exo/likeness`), and basedpyright (with Pillow in the venv) and
  ruff are clean for the package.

## Verified on m3a

1. `select` (with `--library`) found 85 usable photos and picked 41. `pull` copied them (96 MB).
2. `identity` built the face reference from 40 faces, 0 discarded.
3. `clarity` ran with `--low-ram --megapixels 3`, peaking at 13 GB of MLX memory.
   The face-match score dropped from 0.54 to 0.46. `839cf73` then changed SeedVR2
   to a single pass with a stricter identity check, but that has not been re-run yet.

## Open items, in order

1. **`clarity` looked cartoony/plastic: fix implemented 2026-10-08, GPU run pending.**
   Findings: the photo used (`F8A43F4B`) is itself an AI-generated caricature, not
   a camera photo; SeedVR2 then turned the brown eyes green, added a dotted grid
   artifact on the forehead, a waxy sheen, and drawn-looking beard hair. Now:
   restoration is skipped when the face already measures sharp (`--sharp-threshold`,
   `texture.acutance`); SeedVR2's finest detail is blended with the input's at
   `--texture-strength` (default 0.5) and colors always come from the input;
   `clarity` restores at native size up to 6 MP and never enlarges (`--megapixels 0`
   for full native size); `--seedvr2-model seedvr2-7b` is available. Still to do:
   a real SeedVR2 run on a soft photo (e.g. `43954508`, acutance 0.11) to check
   the blend strength and to measure peak memory at 6 MP with 3B and 7B. On
   2026-10-08 it could not run, because a Qwen3.8-27B llama-server on :8082
   held about 40 GB and left 24.5 GB free, below the 32 GB gate.

   Keep LoRA training on unrestored originals, which `train-prepare` already uses.
1a. **Selection let non-photos through.** Of the 41 pulled: `F8A43F4B` and
   `5593B398` are AI caricatures, `04312D95` and `83615D7B` are face-filter app
   edits, `0AABF3AB` is a photo of an ID card, and `52671C89` and `4FA974FA` are
   re-photographed prints or documents. They feed the identity reference and would
   feed LoRA training. Exclude them and rebuild `identity`; consider filtering on
   missing camera EXIF and document-like images in `select`.
2. **`polish` and the other edit presets have not run yet.** They need exo running
   with `EXO_ENABLE_IMAGE_MODELS=true` and `exolabs/Qwen-Image-Edit-2509-8bit`
   (37 GB; `--launch-model` places it).
3. **LoRA training has not run yet:** `train-prepare` → `train-run` → `train-export`
   (Z-Image Turbo via mflux 0.17.5).
4. **Not started:** `narrate` (Qwen3-TTS voice clone via mlx-audio) and `animate`
   (LTX-2.5 via a community MLX conversion, ~62 GB peak). Both were proposed but
   neither has been approved yet.
5. **Original request, not started:** update the image/video landing page (said to be
   on m3a; not found in this repo) with per-model harnesses, including the two new
   Qwen models and MiniMax H3, plus prompt drafting by an abliterated 27B model.

## m3a facts and cautions

- **Hardware and limits:** Mac Studio M3 Ultra with 96 GB. `iogpu.wired_limit_mb`
  read 80896 after a reboot; the owner intends 94 GB for m3a and 254 GB for m5a,
  and says the ceiling counts all wired memory, macOS's included. Don't re-argue
  this; it is his call.
- **Panic history:** on 2026-10-06 m3a had a "SoC watchdog reset" panic during the
  first, unconstrained `clarity` run. The cause is unproven. `enhance` now refuses
  to start SeedVR2 below 32 GB free (`--min-free-memory-gb`) and supports `--low-ram`.
- **Unmounted drive:** after that reboot `/Volumes/Hermes405BLocal` was not mounted,
  and nobody has said it was fixed.
- **Access:** the owner reaches m3a over SSH via Tailscale. Over SSH, Full Disk
  Access needs System Settings > General > Sharing > Remote Login (i) >
  "Allow full disk access for remote users".
