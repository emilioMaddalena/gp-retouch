# Usage

Here are some examples of how to use gp-retouch for image processing.

## Basic Usage

```python
from gp_retouch.image import Image
import numpy as np

# Create a new image
data = np.zeros((100, 100))
img = Image(data)

# Plot the image
img.plot()
```

## Checking Image Properties

```python
# Check if image is grayscale
print(img.is_grayscale)

# Get image dimensions
print(f"Width: {img.width}, Height: {img.height}")
```

More usage examples will be added as the project develops.
