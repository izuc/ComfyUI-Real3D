import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from skimage import measure

from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
    get_rays,
    get_ray_directions
)

class MarchingCubeHelper(torch.nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution
        self.points_range = [-1.0, 1.0]
        self.grid_vertices = self.create_grid_vertices()

    def create_grid_vertices(self):
        x = np.linspace(self.points_range[0], self.points_range[1], self.resolution)
        y = np.linspace(self.points_range[0], self.points_range[1], self.resolution)
        z = np.linspace(self.points_range[0], self.points_range[1], self.resolution)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        grid_vertices = torch.from_numpy(np.stack([grid_x, grid_y, grid_z], axis=-1).astype(np.float32))
        return grid_vertices

    def update_grid_vertices(self, resolution):
        self.resolution = resolution
        self.grid_vertices = self.create_grid_vertices()

    def forward(self, volume):
        volume_size = volume.numel()
        expected_size = self.resolution ** 3
    
        if volume_size > expected_size:
            volume = volume[:expected_size]  # Take the first expected_size elements
        elif volume_size < expected_size:
            padding_size = expected_size - volume_size
            padding_tensor = torch.zeros(padding_size, dtype=volume.dtype, device=volume.device)  # Create 1D padding tensor
            volume = torch.cat([volume.view(-1), padding_tensor])  # Flatten volume and concatenate with padding tensor
    
        volume = volume.reshape(self.resolution, self.resolution, self.resolution).detach().cpu().numpy()
        verts, faces, _, _ = measure.marching_cubes(volume, level=0, spacing=(1.0, 1.0, 1.0))
        verts = np.array(verts, dtype=np.float32)  # Create a copy of verts with float32 dtype
        faces = np.array(faces, dtype=np.int64)   # Create a copy of faces with int64 dtype
        verts = torch.from_numpy(verts)
        faces = torch.from_numpy(faces)
        return verts, faces
        
class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int

        image_tokenizer_cls: str
        image_tokenizer: dict

        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

    cfg: Config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        
        if any(key.startswith("module.") for key in ckpt.keys()):
            ckpt = {key.replace("module.", ""): item for key, item in ckpt.items()}
        
        model.load_state_dict(ckpt)
        return model

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None

    def forward(self, 
                inputs: torch.FloatTensor, 
                rays_o: torch.FloatTensor,
                rays_d: torch.FloatTensor,
                ):
        # input images in shape [b,1,c,h,w], value range [0,1]
        # rays_o and rays_d in shape [b,Nv,h,w,3]
        batch_size, n_views = rays_o.shape[:2]

        # get triplane
        input_image_tokens: torch.Tensor = self.image_tokenizer(inputs)         # [b,1,c,n]
        input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C')
        tokens: torch.Tensor = self.tokenizer(batch_size)                       # [b,ct,Np*Hp*Wp]
        tokens = self.backbone(tokens, encoder_hidden_states=input_image_tokens)# triplanes in [b,Np,Ct,Hp,Wp]
        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))    # triplanes in [b,Np,Ct',Hp',Wp']
        
        # replicate triplanes
        scene_codes = rearrange(scene_codes.unsqueeze(1).repeat(1,n_views,1,1,1,1),
                                'b Nv Np Ct Hp Wp -> (b Nv) Np Ct Hp Wp')

        # render
        rays_o = rearrange(rays_o, 'b Nv h w c -> (b Nv) h w c')
        rays_d = rearrange(rays_d, 'b Nv h w c -> (b Nv) h w c')
        render_images, render_masks = self.renderer(self.decoder, 
                                                    scene_codes, 
                                                    rays_o, rays_d, 
                                                    return_mask=True)  # [b*Nv,h,w,3], [b*Nv,h,w]
        render_images = rearrange(render_images, '(b Nv) h w c -> b Nv c h w', Nv=n_views)
        render_masks = rearrange(render_masks, '(b Nv) h w c -> b Nv c h w', Nv=n_views)
        
        return {'images_rgb': render_images, 
                'images_weight': render_masks}


    def get_latent_from_img(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(
            device
        )
        batch_size = rgb_cond.shape[0]

        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )

        tokens: torch.Tensor = self.tokenizer(batch_size)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes

    def render_360(
        self,
        scene_codes,
        n_views: int,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                images_.append(process_output(image))
            images.append(images_)
        return images

    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        if self.isosurface_helper is None:
            self.isosurface_helper = MarchingCubeHelper(resolution)
        else:
            self.isosurface_helper.update_grid_vertices(resolution)

    def extract_mesh(self, scene_codes, resolution: int = 256, threshold: float = 25.0, chunk_size: int = 128, batch_size: int = 8):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for scene_code in scene_codes:
            vertices = []
            faces = []
            chunk_batch = []
            for i in range(0, resolution, chunk_size):
                for j in range(0, resolution, chunk_size):
                    for k in range(0, resolution, chunk_size):
                        chunk_batch.append((i, j, k))
                        if len(chunk_batch) == batch_size:
                            # Extract mesh for the current batch of chunks
                            batch_vertices, batch_faces = self.extract_mesh_chunk_batch(scene_code, chunk_batch, chunk_size, threshold)
                            vertices.append(batch_vertices)
                            faces.append(batch_faces)
                            chunk_batch = []

            # Process any remaining chunks
            if chunk_batch:
                batch_vertices, batch_faces = self.extract_mesh_chunk_batch(scene_code, chunk_batch, chunk_size, threshold)
                vertices.append(batch_vertices)
                faces.append(batch_faces)

            # Combine chunk meshes into a single mesh
            vertices = torch.cat(vertices, dim=0)
            faces = torch.cat(faces, dim=0)

            # Create trimesh object
            mesh = trimesh.Trimesh(
                vertices=vertices.cpu().numpy(),
                faces=faces.cpu().numpy(),
            )
            meshes.append(mesh)

        return meshes

    def extract_mesh_chunk_batch(self, scene_code, chunk_batch, chunk_size, threshold):
        batch_vertices = []
        batch_faces = []
        for i, j, k in chunk_batch:
            # Calculate the chunk bounds
            min_x, max_x = i, min(i + chunk_size, self.isosurface_helper.resolution)
            min_y, max_y = j, min(j + chunk_size, self.isosurface_helper.resolution)
            min_z, max_z = k, min(k + chunk_size, self.isosurface_helper.resolution)

            # Get the grid vertices for the current chunk
            grid_vertices = self.isosurface_helper.grid_vertices[min_x:max_x, min_y:max_y, min_z:max_z]
            grid_vertices = grid_vertices.reshape(-1, 3).to(torch.float32).to(scene_code.device)

            # Query the triplane for the current chunk
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        grid_vertices,
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]

            # Apply marching cubes for the current chunk
            v_pos_chunk, t_pos_idx_chunk = self.isosurface_helper(-(density - threshold))

            # Offset the vertex positions based on the chunk bounds
            v_pos_chunk[:, 0] += min_x
            v_pos_chunk[:, 1] += min_y
            v_pos_chunk[:, 2] += min_z

            # Offset the face indices based on the number of vertices in previous chunks
            t_pos_idx_chunk += len(batch_vertices)

            # Scale the vertex positions to the original range
            v_pos_chunk = scale_tensor(
                v_pos_chunk,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )

            batch_vertices.append(v_pos_chunk)
            batch_faces.append(t_pos_idx_chunk)

            # Free up memory
            del density, v_pos_chunk, t_pos_idx_chunk
            torch.cuda.empty_cache()

        # Concatenate the batch vertices and faces
        batch_vertices = torch.cat(batch_vertices, dim=0)
        batch_faces = torch.cat(batch_faces, dim=0)

        return batch_vertices, batch_faces
