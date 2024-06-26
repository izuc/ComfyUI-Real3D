import sys
import os
import logging
import numpy as np
from PIL import Image
import torch
import time
import rembg
import trimesh

# Add the directory containing your custom module to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from folder_paths import get_filename_list, get_full_path, get_save_image_path, get_output_directory
from comfy.model_management import get_torch_device
from tsr.system import TSR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fill_background(image):
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image

def get_rays(image, n_views=1):
    # Placeholder implementation
    height, width = image.size[1], image.size[0]
    rays_o = torch.zeros((n_views, height, width, 3), dtype=torch.float32)
    rays_d = torch.zeros((n_views, height, width, 3), dtype=torch.float32)
    return rays_o, rays_d

class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")

timer = Timer()

class TripoSRModelLoader:
    def __init__(self):
        self.initialized_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (get_filename_list("checkpoints"),),
                "chunk_size": ("INT", {"default": 8192, "min": 1, "max": 10000})
            }
        }

    RETURN_TYPES = ("TRIPOSR_MODEL",)
    FUNCTION = "load"
    CATEGORY = "Real3D TripoSR"

    def load(self, model, chunk_size):
        device = get_torch_device()

        if not torch.cuda.is_available():
            device = "cpu"

        if not self.initialized_model:
            logging.info("Loading TripoSR model")
            model_path = get_full_path("checkpoints", model)
            model_dir = os.path.dirname(model_path)
            model_filename = os.path.basename(model_path)
            self.initialized_model = TSR.from_pretrained(
                pretrained_model_name_or_path=model_dir,
                config_name="config.yaml",
                weight_name=model_filename
            )
            self.initialized_model.renderer.set_chunk_size(chunk_size)
            self.initialized_model.to(device)

        return (self.initialized_model,)

class TripoSRSampler:

    def __init__(self):
        self.initialized_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TRIPOSR_MODEL",),
                "reference_image": ("IMAGE",),
                "geometry_resolution": ("INT", {"default": 256, "min": 128, "max": 12288}),
                "threshold": ("FLOAT", {"default": 25.0, "min": 0.0, "step": 0.01}),
                "model_save_format": ("STRING", {"default": "obj", "choices": ["obj", "glb"]}),
            },
            "optional": {
                "reference_mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "sample"
    CATEGORY = "Real3D TripoSR"

    def sample(self, model, reference_image, geometry_resolution, threshold, model_save_format, reference_mask=None):
        device = get_torch_device()

        if not torch.cuda.is_available():
            device = "cpu"

        image = reference_image[0]

        if reference_mask is not None:
            mask = reference_mask[0].unsqueeze(2)
            image = torch.cat((image, mask), dim=2).detach().cpu().numpy()
        else:
            image = image.detach().cpu().numpy()

        image = Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8))
        if reference_mask is not None:
            image = fill_background(image)
        image = image.convert('RGB')

        timer.start("Running model")
        with torch.no_grad():
            scene_codes = model.get_latent_from_img([image], device=device)
        timer.end("Running model")

        timer.start("Extracting mesh")
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=geometry_resolution, threshold=threshold)
        timer.end("Extracting mesh")

        # Logging for debugging
        if meshes:
            logging.info(f"Number of meshes extracted: {len(meshes)}")
            mesh = meshes[0]
            logging.info(f"Mesh vertices shape: {mesh.vertices.shape}")
            logging.info(f"Mesh faces shape: {mesh.faces.shape}")
            logging.info(f"Mesh vertices: {mesh.vertices}")
            logging.info(f"Mesh faces: {mesh.faces}")
        else:
            logging.error("No meshes were extracted.")

        # Ensure mesh is valid before exporting
        if meshes and len(meshes[0].vertices) > 0 and len(meshes[0].faces) > 0:
            output_filename_base = os.path.join(get_output_directory(), f"mesh_{time.time()}")

            # Export OBJ
            output_filename_obj = output_filename_base + ".obj"
            try:
                meshes[0].export(output_filename_obj)
                logging.info(f"Mesh exported successfully to {output_filename_obj}.")
            except Exception as e:
                logging.error(f"Error exporting OBJ mesh: {e}")

            # Export GLB if specified
            if model_save_format == "glb":
                output_filename_glb = output_filename_base + ".glb"
                try:
                    scene = trimesh.Scene(meshes)
                    scene.export(output_filename_glb)
                    logging.info(f"Mesh exported successfully to {output_filename_glb}.")
                except Exception as e:
                    logging.error(f"Error exporting GLB mesh: {e}")
        else:
            logging.error("No valid mesh extracted.")

        return ([meshes[0]] if meshes else [],)


class TripoSRViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",)
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "display"
    CATEGORY = "Real3D TripoSR"

    def display(self, mesh):
        saved = list()
        full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path("meshsave", get_output_directory())

        for (batch_number, single_mesh) in enumerate(mesh):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.obj"
            single_mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
            single_mesh.export(os.path.join(full_output_folder, file))
            saved.append({
                "filename": file,
                "type": "output",
                "subfolder": subfolder
            })

        return {"ui": {"mesh": saved}}

NODE_CLASS_MAPPINGS = {
    "TripoSRModelLoader": TripoSRModelLoader,
    "TripoSRSampler": TripoSRSampler,
    "TripoSRViewer": TripoSRViewer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoSRModelLoader": "TripoSR Model Loader",
    "TripoSRSampler": "TripoSR Sampler",
    "TripoSRViewer": "TripoSR Viewer"
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
