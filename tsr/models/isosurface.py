from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchmcubes import marching_cubes


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None or self._grid_vertices.shape[0] != self.resolution:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.stack((x, y, z), dim=-1)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        # Debug print to check the shape of the tensor before reshaping
        print(f"Original level shape: {level.shape}, expected reshape to: {[self.resolution, self.resolution, self.resolution]}")

        # Squeeze to remove the extra dimension
        level = level.squeeze()

        # Normalize the density values
        min_val, max_val = level.min().item(), level.max().item()
        if max_val - min_val != 0:
            level = (level - min_val) / (max_val - min_val)
        else:
            raise ValueError("Density values have zero range, normalization not possible.")
        
        # Adjust reshape based on the actual shape of level tensor
        if level.numel() == self.resolution ** 3:
            level = -level.view(self.resolution, self.resolution, self.resolution)
        else:
            raise ValueError(f"Cannot reshape level tensor of shape {level.shape} to {[self.resolution, self.resolution, self.resolution]}")
        
        try:
            v_pos, t_pos_idx = self.mc_func(level.detach().cpu(), 0.0)
        except Exception as e:
            print(f"Error during marching cubes: {e}")
            raise

        # Validate vertices and faces
        if v_pos.size(0) == 0 or t_pos_idx.size(0) == 0:
            raise ValueError("Marching cubes returned empty vertices or faces")

        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
