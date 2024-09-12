# Diffusion Model from Scratch
## Diffusion Model

The provided code defines a `DiffusionModel` class in PyTorch. It's part of a **denoising diffusion probabilistic model (DDPM)**, typically used for generating images by adding and gradually removing noise in a controlled way. Here's an explanation of each part of the class:

### Initialization (`__init__`):

```python
def __init__(self, start_schedule=0.0001, end_schedule=0.02, timesteps=300):
    self.start_schedule = start_schedule
    self.end_schedule = end_schedule
    self.timesteps = timesteps
    self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
    self.alphas = 1 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
```

1. **`betas`**: A linear schedule of noise levels over `timesteps` from `start_schedule` to `end_schedule`. `betas` represents how much noise is added at each step.
2. **`alphas`**: Defined as `1 - betas`, they represent how much of the original signal is preserved at each step.
3. **`alphas_cumprod`**: This is the cumulative product of `alphas`, which represents the total fraction of the original signal preserved after a given number of steps.

### `forward` method:

```python
def forward(self, x_0, t, device):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)

    mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
    variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

    return mean + variance, noise.to(device)
```

1. **`x_0`**: The original input image (batch of images).
2. **`t`**: The time step index for which noise is being added (can be different for each image in the batch).
3. **`noise`**: Random Gaussian noise, with the same shape as `x_0`, added to the image at step `t`.
4. **`sqrt_alphas_cumprod_t`**: This term scales the original image `x_0`. It retrieves the cumulative product of the alphas for the current time step `t`.
5. **`sqrt_one_minus_alphas_cumprod_t`**: This term scales the noise component. It is the square root of the cumulative noise added up to step `t`.
6. **Mean and variance**:
   - `mean`: This term adjusts the original image.
   - `variance`: This term adjusts the noise.

The method **returns the noisy image** (a combination of the scaled original image and noise) and the **noise** itself.

### `backward` method:

```python
@torch.no_grad()
def backward(self, x, t, model, **kwargs):
    betas_t = self.get_index_from_list(self.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
    sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)

    mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, **kwargs) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = betas_t

    if t == 0:
        return mean
    else:
        noise = torch.randn_like(x)
        variance = torch.sqrt(posterior_variance_t) * noise
        return mean + variance
```

1. **Purpose**: This method denoises the image using the learned noise model (reverse process). It predicts the noise at each step and gradually removes it.
2. **`model(x, t, **kwargs)`**: The model is called to predict the noise at step `t`.
3. **`mean`**: This computes the mean for the reverse step (denoising).
4. **`posterior_variance_t`**: Represents the variance for the noise added at each step.
5. **`if t == 0`**: If it's the final step (`t == 0`), no further noise is added; otherwise, some noise is added back.

### `get_index_from_list` method:

```python
@staticmethod
def get_index_from_list(values, t, x_shape):
    batch_size = t.shape[0]
    result = values.gather(-1, t.cpu())
    return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
```

1. **Purpose**: Extracts the values corresponding to the time step `t` for each image in the batch.
2. **`gather(-1, t.cpu())`**: Retrieves the values from the `values` tensor based on the indices in `t` (time steps).
3. **Reshaping**: Reshapes the output to match the batch size and input image dimensions for broadcasting.

### Summary:

- The `forward` method adds noise to an image, simulating the diffusion process.
- The `backward` method uses the learned model to predict and remove noise, denoising the image step-by-step.
- The `get_index_from_list` method helps retrieve specific time-step values for the batch during the forward and backward processes.

Would you like to implement training loops for this model or any additional features like sampling?

## Sinusodial Position Embeddings

This class defines **Sinusoidal Position Embeddings**, which are often used in transformer models or in diffusion models. These embeddings are based on sine and cosine functions and are designed to encode temporal or positional information into vectors.

Let me explain the parts of the `SinusoidalPositionEmbeddings` class and its `forward` method:

### Breakdown of the Class:

#### 1. **`__init__(self, dim)`**:

```python
def __init__(self, dim):
    super().__init__()
    self.dim = dim
```

- **Purpose**: Initializes the module with the dimension `dim`, which is the size of the output embedding vector. For example, `dim=128` would mean the embedding vector will have 128 elements.
- **`self.dim`**: This stores the dimension size to be used in the embedding computation.

#### 2. **`forward(self, time)`**:

```python
def forward(self, time):
    device = time.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings
```

- **`time`**: A tensor of time steps or positional indices for which you want to generate embeddings. This tensor typically has shape `(B,)` where `B` is the batch size.
- **`half_dim`**: Half of the total embedding dimension (`dim // 2`). We calculate sine and cosine embeddings for half the dimension, and then concatenate them.

- **`math.log(10000) / (half_dim - 1)`**: This sets a scaling factor for different positions in the embedding. The `10000` is a commonly used constant in positional encoding, ensuring that different dimensions are at different frequencies.
- **`torch.exp(torch.arange(half_dim, device=device) * -embeddings)`**: This line creates a range of frequencies that will be applied to the `time` tensor. The `arange(half_dim)` creates a tensor of indices, which are scaled to exponential values using the earlier computed `embeddings`.

- **`time[:, None] * embeddings[None, :]`**: The `time` tensor is reshaped and multiplied element-wise by the exponential frequencies. This results in a `(B, half_dim)` shaped tensor that holds the sine and cosine arguments.

- **`torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)`**: The sine and cosine of the embeddings are computed separately and then concatenated along the last dimension. This results in a `(B, dim)` tensor where the first half of the elements come from `sin` and the second half from `cos`.

- **Return Value**: The resulting sinusoidal position embeddings of shape `(B, dim)`.

### Example of Usage:

```python
import torch
import math
import torch.nn as nn

# Define a time tensor and instantiate the embedding class
time_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)  # Batch of 3 time steps
embedding_dim = 128  # Example embedding dimension

# Initialize the sinusoidal position embeddings module
position_embedding = SinusoidalPositionEmbeddings(dim=embedding_dim)

# Get the embeddings for the input time tensor
embeddings = position_embedding(time_tensor)

print("Shape of the embeddings:", embeddings.shape)  # Should be (3, 128)
print(embeddings)
```

### Output:

- The shape of the returned `embeddings` would be `(3, 128)` (3 time steps, each represented by a 128-dimensional vector).

### Summary:

- This class creates embeddings for time steps using a sinusoidal function (both sine and cosine). These embeddings are often used in models to represent the relative or absolute position of elements (like time steps or positions in a sequence).

**a.** Would you like to visualize these embeddings to see how they behave over time?  
**b.** Should we integrate these embeddings into a model or explore their usage further?
