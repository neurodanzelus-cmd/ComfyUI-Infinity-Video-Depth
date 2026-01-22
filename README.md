# Infinity Video Depth

This is a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) implementation of [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything), re-engineered for consistent, memory-efficient depth estimation on infinite length video sequences.


<img width="3791" height="1466" alt="image" src="https://github.com/user-attachments/assets/71627678-5482-416f-b52d-d408e948d176" />

## Features

- **Infinite Length Support**: Processes video sequences of any length (10,000+ frames) on consumer GPUs without running out of VRAM.
- **Stateful Streaming**: Uses a sliding window approach with CPU offloading to maintain full temporal consistency across the entire video, effectively eliminating the "flickering" or "jumping" depth values common when chunking videos.
- **High-Precision Output**: Save depth maps as **16-bit** or **32-bit** OpenEXR files for professional VFX pipelines.
- **Visualizations**: Built-in colormap support (Inferno, Viridis, Magma, etc.) for easy visualization directly in ComfyUI.
- **Model Support**: Full support for Video-Depth-Anything Small, Base, and Large models (including Metric variants).

## Comparison with Original

| Feature | Original / Standard Nodes | Infinity Video Depth |
| :--- | :--- | :--- |
| **Video Length** | Limited by GPU VRAM (OOM on long clips) | **Unlimited** (Streaming + CPU Offload) |
| **Consistency** | Splitting videos causes depth "jumps" at cuts | **Perfect Consistency** (State Persistence) |
| **Memory Usage** | Linear growth with frames (Explodes VRAM) | **Constant VRAM** (Fixed window size) |
| **Output** | Standard Images | **EXR (16/32-bit float)** & Colorized Maps |

## Algorithm

The "Infinity" streaming mode works by processing the video frame-by-frame while maintaining a "context window" of hidden states from the Transformer backbone. 
1.  **Sliding Window**: A fixed number of past frames (e.g., 32) contribute to the attention mechanism for the current frame.
2.  **CPU Offload**: To prevent VRAM explosion, the cached hidden states are moved to system RAM (CPU) when not in use and pre-fetched to the GPU only when needed for the current inference step.
3.  **Iterative Processing**: Input frames are processed and converted sequentially, minimizing the memory footprint.

## Usage

### Nodes

#### 1. Infinity Video Depth Loader
Loads the model weights.
-   **model**: Select the version (`vitl`, `vits`, etc.).
-   **download_model**: Auto-download from HuggingFace.

#### 2. Infinity Video Depth Run
The core inference node.
-   **use_streaming**: **Enable this for long videos.** It activates the memory-efficient engine.
-   **clean_cache_interval**: How often (in frames) to trigger garbage collection. Default `10` is safe for low-VRAM cards.
-   **colormap**: Select a visualization style (`gray`, `inferno`, `viridis`, etc.) for the preview output. <img width="700" height="260" alt="image" src="https://github.com/user-attachments/assets/3cb9ea9b-85d2-4866-9e0f-1d594660fb8f" />


**Outputs:**
-   **depth_preview**: Colorized RGB image for preview/video.
-   **depth_exr**: Raw float values for EXR saving.

#### 3. Infinity Video Depth Save EXR
Saves high-dynamic-range depth maps.
-   **float_type**: Choose `32-bit` (full float) or `16-bit` (half float) for storage efficiency.

#### 4. Infinity Video Depth Colorizer
A utility node to apply colormaps to existing depth images.

## Installation

1.  Clone into `ComfyUI/custom_nodes`:
    ```bash
    git clone https://github.com/yourusername/ComfyUI-Infinity-Video-Depth.git
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires `OpenEXR` system libraries)*
    

___

    

Guys, I’d really appreciate any support right now. I’m in a tough spot:

[![Boosty](https://img.shields.io/badge/Boosty-Support-orange?style=for-the-badge)](https://boosty.to/danzelus)
[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/danzelus)


