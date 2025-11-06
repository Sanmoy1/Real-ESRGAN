# Real-ESRGAN Inference Pipeline and Data Flow

- **Entry points**
  - **Image**: `inference_realesrgan.py`
  - **Video**: `inference_realesrgan_video.py`
  - **Runtime helper**: `realesrgan/utils.py` (`RealESRGANer`)

## 1) Model selection and weight loading

- **Model construction** (CLI scripts): choose generator by `--model_name`.
  - RRDB variants → `RRDBNet` (from BasicSR)
  - SRVGG variants → `SRVGGNetCompact` (`realesrgan/archs/srvgg_arch.py`)

- **RealESRGANer initialization**: `RealESRGANer.__init__(...)` in `realesrgan/utils.py`
  - Downloads `model_path` if it is an https URL (stored under `weights/`).
  - Loads checkpoint: `loadnet = torch.load(..., map_location='cpu')`.
  - Prefers `params_ema` if present; else uses `params`.
  - `model.load_state_dict(loadnet[key], strict=True)`, sets `eval()`, moves to device, optional `half()`.
  - Sets tiling params: `tile_size`, `tile_pad`, `pre_pad`, `scale`, precision, device.

### Optional: DNI blending (denoise strength)

- If `model_path` is a list, `RealESRGANer.__init__` calls `dni(net_a, net_b, dni_weight)`.
- `dni()` loads two checkpoints and interpolates parameters: `w0 * A + w1 * B` (per key).
- Used by `realesr-general-x4v3` to expose `--denoise_strength` via two weights (`clean` and `wdn`).

## 2) Per-image pipeline (`RealESRGANer.enhance`)

Function: `enhance(img, outscale=None, alpha_upsampler='realesrgan')`

- **Input preparation**
  - Convert to `float32`; detect 16-bit vs 8-bit; normalize to [0,1].
  - Handle color:
    - Gray → expand to 3-channel RGB.
    - RGBA → split alpha; run RGB path; alpha handled later.
    - Otherwise BGR→RGB for the network.

- **Pre-process**: `pre_process(img)`
  - HWC→CHW tensor on device; optional half precision.
  - Apply `pre_pad` using reflect padding to reduce border artifacts.
  - Set `mod_scale` (2 if `scale==2`, 4 if `scale==1`) and reflect-pad H,W so they are divisible.

- **Inference**
  - If `tile_size > 0` → `tile_process()` (tiled inference with stitched output).
  - Else → `process()` (single forward): `self.output = self.model(self.img)`.

- **Post-process**: `post_process()`
  - Remove `mod_pad` (scaled by `scale`).
  - Remove `pre_pad` (scaled by `scale`).
  - Convert tensor → HWC numpy, clamp to [0,1], reorder back to BGR for OpenCV.

- **Alpha handling** (if input was RGBA)
  - If `alpha_upsampler == 'realesrgan'` → run the same pipeline on alpha (as 3-channel), then convert back to gray and merge.
  - Else → resize alpha with OpenCV `INTER_LINEAR` to `(w*scale, h*scale)` and merge.

- **Output scale adjustment**
  - If `outscale` is set and differs from the model `scale`, resize final output with Lanczos (`INTER_LANCZOS4`).
  - Convert back to `uint8` or `uint16` depending on input bit depth.

## 3) Tiled inference details (`RealESRGANer.tile_process`)

Goal: avoid OOM by splitting the input (after pre_pad and mod_pad) into overlapping tiles, upscaling each tile, then stitching seamlessly.

Steps (all coordinates are in the preprocessed image space):

1. Read shapes: `B,C,H,W = self.img.shape`. Compute output canvas `self.output` of shape `B,C,H*scale,W*scale` initialized to zeros.
2. Compute tile grid: `tiles_x = ceil(W / tile_size)`, `tiles_y = ceil(H / tile_size)`.
3. For each tile `(x, y)`:
   - Base (unpadded) input region:
     - `input_start_x = x * tile_size`, `input_end_x = min(input_start_x + tile_size, W)`
     - `input_start_y = y * tile_size`, `input_end_y = min(input_start_y + tile_size, H)`
   - Expanded with `tile_pad` (clamped to [0,W]/[0,H]):
     - `input_start_x_pad = max(input_start_x - tile_pad, 0)`
     - `input_end_x_pad = min(input_end_x + tile_pad, W)`
     - `input_start_y_pad = max(input_start_y - tile_pad, 0)`
     - `input_end_y_pad = min(input_end_y + tile_pad, H)`
   - Slice padded tile: `input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]`.
   - Forward pass (no grad): `output_tile = self.model(input_tile)`.
   - Compute destination region on the full output canvas (for the unpadded tile area):
     - `output_start_x = input_start_x * scale`, `output_end_x = input_end_x * scale`
     - `output_start_y = input_start_y * scale`, `output_end_y = input_end_y * scale`
   - Compute crop on `output_tile` to discard the scaled tile padding:
     - `output_start_x_tile = (input_start_x - input_start_x_pad) * scale`
     - `output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * scale`
     - `output_start_y_tile = (input_start_y - input_start_y_pad) * scale`
     - `output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * scale`
   - Paste: assign the cropped `output_tile` region into `self.output[:, :, output_start_y:output_end_y, output_start_x:output_end_x]`.

Result: tiles stitch without seams due to per-tile `tile_pad` overlap + cropping; global `pre_pad` and `mod_pad` are removed in `post_process()`.

## 4) Video inference flow (high level)

- `inference_realesrgan_video.py` builds the same `RealESRGANer` and processes frames from `Reader`.
- Frames go through the exact `enhance()` pipeline (including tiling if enabled).
- `Writer` encodes frames to video; optional audio muxing; optional multi-GPU splitting/concat.

## 5) Quick reference (functions)

- **Weight load & init**: `RealESRGANer.__init__` (`realesrgan/utils.py`)
- **DNI**: `RealESRGANer.dni`
- **Per-image**: `RealESRGANer.enhance`
- **Pre/Tiling/Post**: `pre_process`, `tile_process`, `process`, `post_process`
- **CLI selection**: `inference_realesrgan.py`, `inference_realesrgan_video.py`

## 6) Mermaid overview

```mermaid
flowchart TD
  A[Input image/frame] --> B[enhance]
  B --> C[pre_process\n(pre_pad, mod_pad, to tensor)]
  C --> D{tile_size > 0?}
  D -->|Yes| E[tile_process\n(overlap pad→forward→crop→stitch)]
  D -->|No| F[process\n(single forward)]
  E --> G[post_process\n(remove mod+pre pads)]
  F --> G
  G --> H[alpha handling\n(optional)]
  H --> I[outscale resize\n(optional Lanczos)]
  I --> J[Output image]

  subgraph Init
    K[Build net by model_name]\n--> L[Load checkpoint\nprefer params_ema]\n--> M[Move to device\noptional half]
  end
```

