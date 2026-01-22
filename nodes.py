import os
import sys
import torch
import numpy as np
import gc
import cv2
from tqdm import tqdm

# Add current directory to sys.path to allow imports from infinity_video_depth and ivd_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import folder_paths
from infinity_video_depth.video_depth import VideoDepthAnything
from infinity_video_depth.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStream

# Define model URLs
MODEL_URLS = {
    "vits": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth",
    "metric_vits": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth",
    "metric_vitb": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth",
    "metric_vitl": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth",
}

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

COLORMAPS = {
    "gray": None,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "cividis": cv2.COLORMAP_CIVIDIS,
    "heatmap": cv2.COLORMAP_JET,
}

class InfinityVideoDepthLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (list(MODEL_URLS.keys()),),
                "download_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VIDEO_DEPTH_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Infinity Video Depth"

    def load_model(self, model, download_model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine model type and metric
        is_metric = "metric" in model
        encoder = model.replace("metric_", "")
        
        # Define download path
        model_path = os.path.join(folder_paths.models_dir, "infinity_video_depth")
        os.makedirs(model_path, exist_ok=True)
        filename = f"{model}.pth"
        file_path = os.path.join(model_path, filename)
        
        if not os.path.exists(file_path):
            if download_model:
                print(f"Downloading {model} to {file_path}...")
                import requests
                response = requests.get(MODEL_URLS[model], stream=True)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download complete.")
            else:
                raise FileNotFoundError(f"Model {model} not found at {file_path} and download_model is False.")
        
        # Load model state dict
        state_dict = torch.load(file_path, map_location="cpu")
        
        return ({"model_name": model, "state_dict": state_dict, "config": MODEL_CONFIGS[encoder], "is_metric": is_metric},)

class InfinityVideoDepthRun:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_depth_model": ("VIDEO_DEPTH_MODEL",),
                "images": ("IMAGE",),
                "input_size": ("INT", {"default": 518, "min": 224, "max": 2048}),
                "max_len": ("INT", {"default": -1, "min": -1}),
                "fp32": ("BOOLEAN", {"default": False}),
                "use_streaming": ("BOOLEAN", {"default": False}),
                "clean_cache_interval": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1}),
                "colormap": (list(COLORMAPS.keys()),),
                "raw_output_mode": (["Raw (Original)", "Normalized (0-1)", "Logarithmic"], {"default": "Raw (Original)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "EXR_IMAGE")
    RETURN_NAMES = ("depth_preview", "depth_exr")
    FUNCTION = "run_inference"
    CATEGORY = "Infinity Video Depth"

    def run_inference(self, video_depth_model, images, input_size, max_len, fp32, use_streaming, clean_cache_interval=10, colormap="gray", raw_output_mode="Raw (Original)"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        config = video_depth_model["config"]
        state_dict = video_depth_model["state_dict"]
        is_metric = video_depth_model["is_metric"]
        
        model = None
        if use_streaming:
            model = VideoDepthAnythingStream(**config)
        else:
            model = VideoDepthAnything(**config, metric=is_metric)
            
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()
        
        # Determine total frames to process
        total_frames = len(images)
        if max_len > 0:
             total_frames = min(total_frames, max_len)
        
        depths_list = []
        
        if use_streaming:
            # Memory efficient streaming: process one by one
            for i in tqdm(range(total_frames), desc="Processing frames"):
                frame = (images[i].cpu().numpy() * 255).astype(np.uint8)
                depth = model.infer_video_depth_one(frame, input_size=input_size, device=device, fp32=fp32)
                depths_list.append(depth)
                del frame
                if i > 0 and i % clean_cache_interval == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        else:
            # Offline mode
            frames_np = (images[:total_frames].cpu().numpy() * 255).astype(np.uint8)
            with torch.no_grad():
                 depths, _ = model.infer_video_depth(
                     frames_np, 
                     target_fps=30, 
                     input_size=input_size, 
                     device=device, 
                     fp32=fp32
                 )
            depths_list = [d for d in depths]
            del frames_np

        depths = np.array(depths_list)
        depths_tensor = torch.from_numpy(depths).float().unsqueeze(-1) # [N, H, W, 1]
        
        # Process EXR (Raw) output based on selected mode
        if raw_output_mode == "Normalized (0-1)":
            d_min = depths_tensor.min()
            d_max = depths_tensor.max()
            if d_max > d_min:
                depth_exr_processed = (depths_tensor - d_min) / (d_max - d_min)
            else:
                depth_exr_processed = torch.zeros_like(depths_tensor)
        elif raw_output_mode == "Logarithmic":
            # Apply log(1 + x) to keep it positive and handle large ranges
            depth_exr_processed = torch.log1p(depths_tensor)
        else:
            # Raw (Original)
            depth_exr_processed = depths_tensor

        depth_exr = depth_exr_processed.repeat(1, 1, 1, 3)
        
        # Create Preview output (Always Normalized 0-1 for display)
        p_min = depths_tensor.min()
        p_max = depths_tensor.max()
        if p_max > p_min:
            depth_norm = (depths_tensor - p_min) / (p_max - p_min)
        else:
            depth_norm = torch.zeros_like(depths_tensor)
            
        if colormap == "gray":
            depth_preview = depth_norm.repeat(1, 1, 1, 3)
        else:
            # Apply cv2 colormap
            depth_preview_np = (depth_norm.squeeze(-1).numpy() * 255).astype(np.uint8)
            colored_frames = []
            for frame in depth_preview_np:
                colored = cv2.applyColorMap(frame, COLORMAPS[colormap])
                colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB) # OpenCV uses BGR
                colored_frames.append(colored.astype(np.float32) / 255.0)
            depth_preview = torch.from_numpy(np.array(colored_frames))

        return (depth_preview, depth_exr)

class InfinityVideoDepthColorizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "colormap": (list(COLORMAPS.keys()),),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize"
    CATEGORY = "Infinity Video Depth"

    def colorize(self, images, colormap):
        if colormap == "gray":
            # If input is 1 channel, repeat. If 3 channel, assume already gray? 
            # Or convert RGB to Gray? Usually depth maps are 1 channel or 3 repeated.
            if images.shape[-1] == 1:
                return (images.repeat(1, 1, 1, 3),)
            return (images,)
            
        # Images are [B, H, W, C]. Assume depth is in channel 0 or it's grayscale.
        depths_np = (images[..., 0].cpu().numpy() * 255).astype(np.uint8)
        
        colored_frames = []
        for frame in depths_np:
            colored = cv2.applyColorMap(frame, COLORMAPS[colormap])
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            colored_frames.append(colored.astype(np.float32) / 255.0)
            
        return (torch.from_numpy(np.array(colored_frames)),)

class InfinityVideoDepthSaveEXR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "exr_images": ("EXR_IMAGE",),
                "filename_prefix": ("STRING", {"default": "vda_exr"}),
                "float_type": (["32-bit", "16-bit"],),
                "export_channels": (["Z (Depth)", "RGB", "R (Red Only)"],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_exr"
    OUTPUT_NODE = True
    CATEGORY = "Infinity Video Depth"

    def save_exr(self, exr_images, filename_prefix="vda_exr", float_type="32-bit", export_channels="Z (Depth)"):
        try:
            import OpenEXR
            import Imath
        except ImportError:
            raise ImportError("OpenEXR is required to use this node. Please install it with 'pip install OpenEXR'.")

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), exr_images[0].shape[1], exr_images[0].shape[0])
        
        # Determine format
        if float_type == "32-bit":
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            np_type = np.float32
        else:
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
            np_type = np.float16

        for i, image in enumerate(exr_images):
             # Extract simple single channel depth map
             depth = image[..., 0].cpu().numpy().astype(np_type)
             depth_bytes = depth.tobytes()

             file_name = f"{filename}_{counter:05d}.exr"
             full_path = os.path.join(full_output_folder, file_name)
             
             header = OpenEXR.Header(depth.shape[1], depth.shape[0])
             
             # Prepare channel configuration and pixel data based on selection
             header_channels = {}
             pixel_data = {}

             if export_channels == "RGB":
                 # Duplicate depth to R, G, B channels
                 header_channels = {
                     "R": Imath.Channel(pixel_type),
                     "G": Imath.Channel(pixel_type),
                     "B": Imath.Channel(pixel_type)
                 }
                 pixel_data = {
                     "R": depth_bytes,
                     "G": depth_bytes,
                     "B": depth_bytes
                 }
             elif export_channels == "R (Red Only)":
                 # Save only to R channel
                 header_channels = {
                     "R": Imath.Channel(pixel_type)
                 }
                 pixel_data = {
                     "R": depth_bytes
                 }
             else:
                 # Default: Save to Z channel
                 header_channels = {
                     "Z": Imath.Channel(pixel_type)
                 }
                 pixel_data = {
                     "Z": depth_bytes
                 }

             header["channels"] = header_channels
             
             exr_file = OpenEXR.OutputFile(full_path, header)
             exr_file.writePixels(pixel_data)
             exr_file.close()
             
             counter += 1
             
        return {"ui": {"images": []}}

NODE_CLASS_MAPPINGS = {
    "InfinityVideoDepthLoader": InfinityVideoDepthLoader,
    "InfinityVideoDepthRun": InfinityVideoDepthRun,
    "InfinityVideoDepthColorizer": InfinityVideoDepthColorizer,
    "InfinityVideoDepthSaveEXR": InfinityVideoDepthSaveEXR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InfinityVideoDepthLoader": "Infinity Video Depth Loader",
    "InfinityVideoDepthRun": "Infinity Video Depth Run",
    "InfinityVideoDepthColorizer": "Infinity Video Depth Colorizer",
    "InfinityVideoDepthSaveEXR": "Infinity Video Depth Save EXR",
}