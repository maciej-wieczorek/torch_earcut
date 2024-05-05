# Torch Earcut

PyTorch binding for the Mapbox Earcut library, a C++ library for triangulating 2D polygons. `torch_earcut` provides a batch interface, calls to the original function are parallelized using OMP.

Original code: [earcut.hpp](https://github.com/mapbox/earcut.hpp)

Original description:

> The library implements a modified ear slicing algorithm, optimized by
> [z-order curve](http://en.wikipedia.org/wiki/Z-order_curve) hashing and
> extended to handle holes, twisted polygons, degeneracies and self-intersections
> in a way that doesn't _guarantee_ correctness of triangulation, but attempts to
> always produce acceptable results for practical data like geographical shapes.

Good alternative: [mapbox_earcut_python](https://github.com/skogler/mapbox_earcut_python)

Example:


```py
import torch
from torch_earcut import triangulate

vertices = torch.tensor([
    [[0, 0], [0, 1], [1, 1]],
    [[0, 1], [1, 1], [1, 0]]
], dtype=torch.float32)
indices = triangulate(vertices)
# indices == [tensor([1, 0, 2], dtype=torch.int32), 
#             tensor([1, 0, 2], dtype=torch.int32)]

vertices = torch.tensor([[[0, 0], [0, 1], [1, 1], [1, 0]]], dtype=torch.float32)
indices = triangulate(vertices) 
# indices == [tensor([1, 0, 3, 3, 2, 1], dtype=torch.int32)]

```

TODOs:
- Support more dtypes.
- Support holes in the interface.