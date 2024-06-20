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
        if self.isosurface_helper is not None and self.isosurface_helper.resolution == resolution:
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(self, scene_codes, resolution: int = 256, threshold: float = 25.0):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
    
            logging.info(f"Density shape: {density.shape}, min: {density.min()}, max: {density.max()}")
    
            try:
                v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            except Exception as e:
                logging.error(f"Error during marching cubes: {e}")
                continue
    
            logging.info(f"v_pos shape: {v_pos.shape}")
            logging.info(f"t_pos_idx shape: {t_pos_idx.shape}")
            logging.info(f"First 10 vertices:\n{v_pos[:10]}")
            logging.info(f"First 10 faces:\n{t_pos_idx[:10]}")
    
            if t_pos_idx.max() >= v_pos.shape[0]:
                logging.error(f"Invalid face index found: {t_pos_idx.max()} exceeds number of vertices: {v_pos.shape[0]}")
                continue
    
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
    
            with torch.no_grad():
                color = self.renderer.query_triplane(
                    self.decoder,
                    v_pos,
                    scene_code,
                )["color"]
    
            if color.numel() == 0:
                logging.error("Color tensor is empty.")
                continue
    
            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
                vertex_colors=color.cpu().numpy(),
            )
            meshes.append(mesh)
    
            # Free up memory
            del density, v_pos, t_pos_idx, color
            torch.cuda.empty_cache()
    
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
    
            # Log the shape of density
            logging.info(f"Density tensor shape before squeeze: {density.shape}")
            
            # Squeeze the density tensor to remove the extra dimension
            density = density.squeeze()
            logging.info(f"Density tensor shape after squeeze: {density.shape}")
    
            # Normalize the density values
            min_val, max_val = density.min().item(), density.max().item()
            if max_val - min_val != 0:
                density = (density - min_val) / (max_val - min_val)
            else:
                raise ValueError("Density values have zero range, normalization not possible.")
    
            # Reshape density to expected 3D shape
            if density.numel() == self.isosurface_helper.resolution ** 3:
                density = density.view(self.isosurface_helper.resolution, self.isosurface_helper.resolution, self.isosurface_helper.resolution)
            else:
                raise ValueError(f"Cannot reshape density tensor of shape {density.shape} to {[self.isosurface_helper.resolution, self.isosurface_helper.resolution, self.isosurface_helper.resolution]}")
    
            # Apply marching cubes for the current chunk
            try:
                # Adjust the threshold value if needed
                v_pos_chunk, t_pos_idx_chunk = self.isosurface_helper(-(density - threshold))
            except Exception as e:
                logging.error(f"Error during marching cubes: {e}")
                continue
    
            # Validate vertices and faces
            if v_pos_chunk.size(0) == 0 or t_pos_idx_chunk.size(0) == 0:
                logging.error("Marching cubes returned empty vertices or faces")
                continue
    
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

