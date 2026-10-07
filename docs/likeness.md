# likeness

Curate photos of yourself from Apple Photos, enhance them realistically (or
stylize them), and train a likeness LoRA, all locally on the Mac that runs exo.
Photos are pulled from iCloud only for the task that needs them and released
afterwards, so the working set never piles up on disk.

## One-time setup (on the Mac with your Photos library)

1. **Tag yourself in Photos.** Photos > People & Pets: name yourself, merge
   duplicates of you, and accept "review more photos" suggestions. Mark the
   photos you like most as Favorites; they rank higher.
2. **Grant Full Disk Access** to your terminal app (System Settings > Privacy &
   Security > Full Disk Access). osxphotos needs it to read the Photos database.
3. **Get the toolkit** in its own folder, so the exo checkout you run day to day
   stays on its current branch:

   ```bash
   git clone -b claude/busy-ramanujan-iomjoc https://github.com/ericmcavey-jpg/exo ~/exo-likeness
   ln -s ~/exo-likeness/scripts/likeness /usr/local/bin/likeness   # optional: run `likeness` anywhere
   ```

   Without the link, run it as `~/exo-likeness/scripts/likeness`. The script works
   from any folder and adds only what each command needs (osxphotos, InsightFace,
   Pillow, mflux) on top of the toolkit's environment. The first run sets that
   environment up, which takes a few minutes. If `/usr/local/bin` does not exist,
   create it first with `sudo mkdir -p /usr/local/bin`.
4. **Run exo with image models enabled** (needed for `enhance`), from your usual
   exo checkout: `EXO_ENABLE_IMAGE_MODELS=true uv run exo`

First runs download model weights: InsightFace `buffalo_l` (about 300 MB) for
the identity check, SeedVR2 3B for restoration, and Z-Image Turbo for training.
InsightFace's pretrained models are licensed for non-commercial use.

## Workflow

```bash
likeness libraries                                # Photos libraries here and on drives
likeness people                                   # your exact name as tagged in Photos
likeness select --task me --person "Your Name"    # rank photos; downloads nothing
likeness pull --task me                           # download only the chosen originals
likeness identity --task me                       # face reference for the identity check
likeness enhance --recipe polish --identity-task me ~/Desktop/photo.jpg
likeness enhance --recipe anime --from-task me    # stylize every pulled photo
likeness train-prepare --task me --trigger-word ohwx --subject-class man
likeness train-run --task me                      # mflux LoRA training
likeness train-export --task me                   # keep just the LoRA adapter
likeness release --task me                        # delete pulled photos and checkpoints
likeness status                                   # disk use per task
```

### Selection (`select`)

Ranking uses only the local Photos database, which keeps faces, People names and
Photos' own aesthetic scores even when "Optimize Mac Storage" leaves originals in
iCloud. It skips screenshots, hidden photos, low resolution, tiny faces, and
photos where someone else's face is at least half the size of yours (people in
the background are fine). By default it only looks at the last 3 years
(`--since YYYY-MM-DD` or `--years N` to change). Re-running `select` keeps any
photos already pulled for the task, and `pull` only copies what is new.

From the rest it keeps up to `--target` photos (default 60), split across
close-up, waist-up and full-body framings. It allows at most 4 per day and skips
burst near-duplicates. The output says how many originals are already on the Mac
and how many must come from iCloud.

Libraries on macOS Ventura and later do not record head angle or closed eyes in
a form osxphotos can read. Head angle then shows as `unknown` and the closed-eye
filter has no effect, so glance through the pulled photos and favorite the good
ones in Photos before re-running `select`.

### A Photos library on another drive

Pass `--library` to `people` and `select` to read a `.photoslibrary` on an
external drive instead of the one Photos is using. `pull` reuses it from the
manifest. Photos on that drive are copied as they are; the toolkit cannot fetch
originals for it from iCloud, so `select` reports how many are missing from it.
osxphotos only reads the library. Do not open an older library in Photos just
for this: Photos would upgrade it in place.

```bash
likeness libraries
likeness people --library "/Volumes/DRIVE/Photos Library.photoslibrary"
```

`libraries` lists mounted drives and every Photos library it finds in
`~/Pictures` and up to three folders deep on each drive, with the date each was
last updated. If the drive is missing, mount it with `diskutil list external`
and then `diskutil mountDisk diskN`.

**Over SSH**, the Terminal app's Full Disk Access does not apply. On the Mac
itself, open System Settings > General > Sharing, click (i) next to Remote
Login, and turn on "Allow full disk access for remote users".

### Storage (`pull` / `release`)

`pull` exports full-resolution copies of just the selected photos into the
task's `originals/` folder, preferring your own edit in Photos when there is one.
`release` deletes `originals/` and `training/`. It keeps the manifest (which
photos), `identity.json` (a few KB), `lora/` and `outputs/`, so `pull` can
re-create the exact same set later.

Photos also keeps its own cached copy of anything downloaded from iCloud. With
"Optimize Mac Storage" on, macOS trims that cache automatically when space runs
low; there is no supported way to force it sooner.

### Enhancement (`enhance`)

Recipes that edit work at about 1 MP, and `clarity` at about 6 MP (`--megapixels`
to change; photos are never enlarged before restoration). Each recipe sharpens and de-noises with
SeedVR2, applies edits through exo's `/v1/images/edits` (default
`exolabs/Qwen-Image-Edit-2509-8bit`, change with `--edit-model`), and finishes
with a SeedVR2 upscale to a 2048 px short edge (`--final-short-edge`, 0 to skip).
Pass `--launch-model` to have exo load the edit model if it is not running.

**Memory.** SeedVR2 runs on the GPU alongside whatever exo has loaded. Running it
under memory pressure can stall the GPU badly enough that macOS restarts (a "SoC
watchdog reset" panic). `enhance` therefore refuses to start SeedVR2 when less
than 32 GB is free (`--min-free-memory-gb`, 0 disables). `--low-ram` runs it in
mflux's slower low-memory mode, and a smaller `--megapixels` lowers the load too.

| Recipe | Style | What it does |
|---|---|---|
| `clarity` | realistic | Sharpen, de-noise, upscale. No retouching. |
| `polish` | realistic | Professional retouch: flattering light, true skin tones, light cleanup. |
| `groomed` | realistic | Well-rested and well-groomed look. |
| `headshot` | realistic | Studio headshot with a grey backdrop. |
| `golden-hour` | realistic | Warm side light with a rim light. |
| `anime` | stylized | Modern anime illustration. |
| `3d-animated` | stylized | 3D animated feature-film character. |
| `comic` | stylized | Comic-book ink and halftone. |
| `watercolor` | stylized | Watercolor portrait. |

**Identity check.** With `--identity-task`, every realistic edit is compared
with your face reference. An edit must stay within 0.10 of the unedited
photo's similarity score, and above 0.45 when the source photo itself reaches
that. An edit that drifts is retried with a new seed, and skipped if it still
drifts. Stylized recipes only report the similarity. `report.json` in each output
folder records every attempt. Pass `--keep-intermediates` to keep the
step-by-step images.

### Training (`train-prepare` / `train-run` / `train-export`)

mflux trains LoRAs for Z-Image and FLUX.2, not FLUX.1 or Qwen-Image.
`train-prepare` writes captioned images ("a close-up portrait photo of ohwx man")
and a config based on mflux's Z-Image Turbo example, sized to
about 2000 training steps. Pass `--quantize 8` to use less memory. After
training (`likeness train-run --task me`), `train-export` copies the newest
adapter to `lora/`. Use it with:

```bash
uvx --from mflux==0.17.5 mflux-generate-z-image-turbo \
  --lora-paths ~/.exo/likeness/me/lora/me-<step>.safetensors \
  --prompt "a photo of ohwx man ..."
```

## Notes

- The osxphotos calls were checked against osxphotos 0.77 but have not yet been
  run against a live library. Run `select` first and check its summary before
  `pull`.
- Workspaces live in `~/.exo/likeness` (override with `EXO_LIKENESS_HOME`).
