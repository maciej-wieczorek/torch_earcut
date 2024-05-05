import torch
from typing import List
from torch_earcut.cpp import triangulate_float_2_int32

def triangulate(contour: torch.Tensor) -> List[torch.Tensor]:
    return triangulate_float_2_int32(contour)