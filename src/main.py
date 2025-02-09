"""
main.py

This is an implementation of "PuzzleTensor: A Method-Agnostic
Data Transformation for Compact Tensor Factorization"
"""

import time
import math
import random
import numpy as np
import tensorly as tl
import torch
from torch import nn, optim
from tensorly.decomposition import parafac, tucker, matrix_product_state
from tensorly.cp_tensor import cp_to_tensor


class PuzzleTensor(nn.Module):
    """
    PuzzleTensor module that applies multiple layers of frequency-domain shifts
    to the input tensor in order to transform it into a form that yields a lower
    reconstruction error when factorized.

    Attributes:
        batch (int): The size of the batch (first dimension of input tensor).
        shape (tuple): The remaining dimensions of the tensor.
        D (int): Number of dimensions (excluding the batch dimension).
        num_layer (int): The number of frequency shifting layers to apply.
        S (nn.ModuleList): A list of parameter lists for each layer containing learnable shift factors.
        phase (list): List of frequency phase factors for full Fourier components.
        phase_half (list): List of frequency phase factors for half Fourier components.
    """

    def __init__(self, shape, init, num_layer):
        """
        Initialize the PuzzleTensor module.

        Parameters:
            shape (tuple): A tuple specifying the shape of the input tensor, where the first element
                           is the batch size and the rest are dimensions of the tensor.
            init (str): Initialization type. 'r' for random initialization and 'z' for zeros.
            num_layer (int): The number of frequency shifting layers to apply.
        """
        super().__init__()
        self.batch = shape[0]
        self.shape = shape[1:]
        self.D = len(self.shape)
        self.num_layer = num_layer

        if init == 'r':
            init_func = torch.rand
        elif init == 'z':
            init_func = torch.zeros
        else:
            raise ValueError(f'Unknown initialization type: {init}')

        # Create a ModuleList to hold the shift parameter lists for each layer
        self.S = nn.ModuleList()
        for layer in range(num_layer):
            view_shape = (self.batch, ) + tuple(-1 if i == (layer % self.D) else 1 for i in range(self.D))
            params = [nn.Parameter(init_func([self.batch, self.shape[layer % self.D]]).view(view_shape))
                      for _ in range(self.D - 1)]
            self.S.append(nn.ParameterList(params))

        # Precompute phase factors for FFT shifting
        self.phase, self.phase_half = [], []
        for d in range(self.D - 1):
            view_shape = tuple(-1 if i == d else 1 for i in range(self.D))
            sym_arange = (torch.arange(self.shape[d]) + self.shape[d] // 2) % self.shape[d] - self.shape[d] // 2
            sym_arange[self.shape[d] // 2] *= self.shape[d] % 2
            self.phase.append(2.0j * torch.pi * sym_arange.view(view_shape) / self.shape[d])
        for d in range(self.D - 2, self.D):
            view_shape = tuple(-1 if i == d else 1 for i in range(self.D))
            asym_arange = torch.arange(self.shape[d] // 2 + 1)
            asym_arange[-1] *= self.shape[d] % 2
            self.phase_half.append(2.0j * torch.pi * asym_arange.view(view_shape) / self.shape[d])

    def forward(self, x):
        """
        Forward pass of the PuzzleTensor module.
        Applies a sequence of frequency domain transformations on the input tensor x.
        Each layer performs:
            1. FFT transformation.
            2. Outer product with exponential phase factors based on learnable parameters.
            3. Frequency-domain shift multiplication.
            4. Inverse FFT transformation.
        After processing through all layers, the function computes a loss based on the
        nuclear norm of the tensor when reshaped along different modes.

        Parameters:
            x (torch.Tensor): Input tensor, expected to be in the frequency domain for layer 0.

        Returns:
            torch.Tensor: Scalar loss computed from the transformed tensor.
        """
        for layer in range(self.num_layer):
            # Determine the dimensions (axes) to apply FFT/IFFT (skip the current shifting axis)
            dim = tuple(i + 1 for i in range(self.D) if i != (layer % self.D))
            # Calculate the corresponding sizes for inverse FFT
            s = tuple(sh for i, sh in enumerate(self.shape) if i != (layer % self.D))
            # Apply FFT (layer 0 is assumed to be already transformed)
            if layer >= 1:
                x = torch.fft.rfftn(x, dim=dim)
            # Compute the outer product of phase factors
            ph = 1.0
            for k in range(self.D - 2):
                ph = ph * torch.exp(self.S[layer][k] * self.phase[(layer % self.D <= k) + k])
            ph = ph * torch.exp(self.S[layer][self.D - 2] * self.phase_half[(layer % self.D) != (self.D - 1)])
            # Apply the frequency-domain shift
            x = x * ph
            # Apply IFFT
            x = torch.fft.irfftn(x, s=s, dim=dim)

        # Compute the loss
        prod = torch.prod(torch.tensor(self.shape, dtype=torch.int))
        loss_terms = []
        for k in range(self.D):
            perm = [0, k + 1] + [i + 1 for i in range(self.D) if i != k]
            x_perm = x.permute(perm).contiguous()
            x_2d = x_perm.view(self.batch, self.shape[k], prod // self.shape[k])
            norm_k = torch.linalg.matrix_norm(x_2d, ord='nuc')
            loss_terms.append(norm_k / math.sqrt(self.shape[k]))
        loss = sum(loss_terms)
        loss = torch.sum(loss)
        return loss


def solve_puzzle(x, h, num_layer=None, reverse=False):
    """
    Applies a series of frequency domain shifts to a given tensor x,
    using a provided list of shift parameters h.

    Parameters:
        x (torch.Tensor): Input tensor in the spatial domain.
        h (list): List of lists of complex shift parameters obtained from the PuzzleTensor training.
        num_layer (int): Number of layers (shifts) to apply.
        reverse (bool): If True, apply the shifts in reverse order (for inversion).

    Returns:
        torch.Tensor: The tensor after applying the frequency domain shifts.
    """
    shape = x.shape[1:]
    D = len(shape)

    phase, phase_half = [], []
    for d in range(D - 1):
        view_shape = tuple(-1 if i == d else 1 for i in range(D))
        sym_arange = (torch.arange(shape[d]) + shape[d] // 2) % shape[d] - shape[d] // 2
        sym_arange[shape[d] // 2] *= shape[d] % 2
        phase.append(2.0j * torch.pi * sym_arange.view(view_shape) / shape[d])
    for d in range(D - 2, D):
        view_shape = tuple(-1 if i == d else 1 for i in range(D))
        asym_arange = torch.arange(shape[d] // 2 + 1)
        asym_arange[-1] *= shape[d] % 2
        phase_half.append(2.0j * torch.pi * asym_arange.view(view_shape) / shape[d])

    for layer in range(num_layer) if not reverse else reversed(range(num_layer)):
        dim = tuple(i + 1 for i in range(D) if i != (layer % D))
        s = tuple(sh for i, sh in enumerate(shape) if i != (layer % D))
        x = torch.fft.rfftn(x, dim=dim)
        ph = 1.0
        for k in range(D - 2):
            ph = ph * torch.exp(h[layer][k] * phase[(layer % D <= k) + k])
        ph = ph * torch.exp(h[layer][D - 2] * phase_half[(layer % D) != (D - 1)])
        x = x * ph
        x = torch.fft.irfftn(x, s=s, dim=dim)
    return x


def split_tensor(tensor, block_shape):
    """
    Splits an arbitrary n-dimensional tensor into smaller blocks along each axis
    as specified by block_shape, and returns an (n+1)-dimensional tensor stacking these blocks.

    Parameters:
        tensor (torch.Tensor): n-dimensional input tensor with shape [D_0, D_1, ..., D_{n-1}].
        block_shape (list or tuple of ints): Specifies the number of blocks to split each axis into.
            Must have the same length as the number of dimensions of the tensor, and each D_i
            must be divisible by block_shape[i].

    Returns:
        torch.Tensor: A stacked (n+1)-dimensional tensor containing the blocks.
                      The shape is [B_0 * B_1 * ... * B_{n-1}, D_0/B_0, D_1/B_1, ..., D_{n-1}/B_{n-1}],
                      where B_i = block_shape[i].
    """
    if tensor.dim() != len(block_shape):
        raise ValueError("Length of block_shape must match the number of tensor dimensions.")

    blocks = [tensor]
    for axis, num_chunks in enumerate(block_shape):
        new_blocks = []
        for block in blocks:
            new_blocks.extend(torch.chunk(block, num_chunks, dim=axis))
        blocks = new_blocks
    return torch.stack(blocks)


def train(x, shape, epochs=6001, init='z', num_layer=None):
    """
    Trains the PuzzleTensor model on the provided tensor x.

    The function performs staged training by gradually enabling the gradient
    computation for different groups of parameters at specified epochs.

    Parameters:
        x (torch.Tensor): Input tensor (in frequency domain) on which to train the model.
        shape (tuple): Shape of the tensor including the batch dimension.
        epochs (int): Total number of training epochs (default is 6001).
        init (str): Initialization type for the PuzzleTensor parameters ('r' for random, 'z' for zeros).
        num_layer (int): Number of frequency shifting layers in the PuzzleTensor.

    Returns:
        list: A list of lists containing the learned shift parameters for each layer.
    """
    pz = PuzzleTensor(shape, init=init, num_layer=num_layer)
    optimizer = optim.Adam(pz.parameters(), lr=1e-3)

    start = time.time()
    for epoch in range(epochs):
        loss = pz(x)

        # Staged Training
        if epoch == 0:
            for d in range(pz.D - 1):
                pz.S[-3][d].requires_grad = False
                pz.S[-2][d].requires_grad = False
                pz.S[-1][d].requires_grad = False
        elif epoch == epochs // 4:
            for d in range(pz.D - 1):
                pz.S[-3][d].requires_grad = True
        elif epoch == 2 * epochs // 4:
            for d in range(pz.D - 1):
                pz.S[-2][d].requires_grad = True
        elif epoch == 3 * epochs // 4:
            for d in range(pz.D - 1):
                pz.S[-1][d].requires_grad = True
        else:
            pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'epoch: {epoch}\t'
                  f'|\tloss: {loss.item():.4f} \t'
                  f'|\ttime: {time.time() - start:.4f}')
            start = time.time()

    shifts = [[pz.S[i][j].detach() for j in range(pz.D - 1)] for i in range(num_layer)]
    return shifts


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    np.set_printoptions(precision=7, linewidth=999, suppress=True)
    torch.set_printoptions(precision=7, linewidth=999, sci_mode=False)

    # Generate synthetic data
    I, J, K = 32, 32, 64
    mask = torch.rand(I, J, K) > 0.99
    x = torch.zeros(I, J, K)
    x[mask] = 1.0

    # Hyperparameters
    INIT = 'z'
    EPOCHS = 6001
    NUM_LAYER = 9
    block_shape = [1, 1, 2]
    ranks_CP_1 = [27, 67, 133]
    ranks_TK_1 = [11, 17, 22]
    ranks_TT_1 = [8, 13, 19]
    ranks_CP_2 = [21, 61, 127]
    ranks_TK_2 = [9, 16, 22]
    ranks_TT_2 = [7, 12, 19]

    # Preprocessing: Split the tensor into blocks
    D = x.dim()
    x = split_tensor(x, block_shape=block_shape)
    # Apply FFT to convert the data to the frequency domain
    fx = torch.fft.rfftn(x, dim=tuple(i for i in range(1 - D, 0)))
    # Train the PuzzleTensor to learn frequency shifts
    shifts = train(fx, shape=x.shape, epochs=EPOCHS, init=INIT, num_layer=NUM_LAYER)
    x = x.detach()
    # Apply the learned shifts to solve the puzzle
    sol = solve_puzzle(x, shifts, num_layer=NUM_LAYER)

    # (Optional) Measure the distortion induced by shifting
    inv_shifts = [[-tensor for tensor in sublist] for sublist in shifts]
    pred = solve_puzzle(sol, inv_shifts, num_layer=NUM_LAYER, reverse=True)
    error = torch.norm(x - pred) / torch.norm(x)
    print(f"Error induced by shifting: {error.item():.7f}")

    # Set Tensorly backend to PyTorch
    tl.set_backend('pytorch')

    # Evaluate CP decomposition on both tensors
    errors_CP_1, errors_CP_2 = [], []
    for r in ranks_CP_1:
        temp_diff, temp_norm = [], []
        for b in range(x.shape[0]):
            weights, factors = parafac(x[b], rank=r, normalize_factors=False, n_iter_max=100, tol=5e-2)
            reconstructed_tensor = cp_to_tensor((weights, factors))
            temp_diff.append(torch.norm(x[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(x[b]).item())
        errors_CP_1.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))
    for r in ranks_CP_2:
        temp_diff, temp_norm = [], []
        for b in range(sol.shape[0]):
            weights, factors = parafac(sol[b], rank=r, normalize_factors=False, n_iter_max=100, tol=5e-2)
            reconstructed_tensor = cp_to_tensor((weights, factors))
            temp_diff.append(torch.norm(sol[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(sol[b]).item())
        errors_CP_2.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))

    # Evaluate Tucker (TK) decomposition on both tensors
    errors_TK_1, errors_TK_2 = [], []
    for r in ranks_TK_1:
        temp_diff, temp_norm = [], []
        for b in range(x.shape[0]):
            core, factors = tucker(x[b], rank=[int(r)] * D)
            reconstructed_tensor = tl.tucker_to_tensor((core, factors))
            temp_diff.append(torch.norm(x[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(x[b]).item())
        errors_TK_1.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))
    for r in ranks_TK_2:
        temp_diff, temp_norm = [], []
        for b in range(sol.shape[0]):
            core, factors = tucker(sol[b], rank=[int(r)] * D)
            reconstructed_tensor = tl.tucker_to_tensor((core, factors))
            temp_diff.append(torch.norm(sol[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(sol[b]).item())
        errors_TK_2.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))

    # Evaluate Tensor-Train (TT) decomposition on both tensors
    errors_TT_1, errors_TT_2 = [], []
    for r in ranks_TT_1:
        temp_diff, temp_norm = [], []
        for b in range(x.shape[0]):
            tt_factors = matrix_product_state(x[b], rank=int(r))
            reconstructed_tensor = tl.tt_to_tensor(tt_factors)
            temp_diff.append(torch.norm(x[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(x[b]).item())
        errors_TT_1.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))
    for r in ranks_TT_2:
        temp_diff, temp_norm = [], []
        for b in range(x.shape[0]):
            tt_factors = matrix_product_state(sol[b], rank=int(r))
            reconstructed_tensor = tl.tt_to_tensor(tt_factors)
            temp_diff.append(torch.norm(sol[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(sol[b]).item())
        errors_TT_2.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))

    print("\nReconstruction Errors")
    print(f"CP                :", "\t".join(map(lambda a: f"{a:8.4f}", errors_CP_1)))
    print(f"CP + PuzzleTensor :", "\t".join(map(lambda a: f"{a:8.4f}", errors_CP_2)))
    print(f"TK                :", "\t".join(map(lambda a: f"{a:8.4f}", errors_TK_1)))
    print(f"TK + PuzzleTensor :", "\t".join(map(lambda a: f"{a:8.4f}", errors_TK_2)))
    print(f"TT                :", "\t".join(map(lambda a: f"{a:8.4f}", errors_TT_1)))
    print(f"TT + PuzzleTensor :", "\t".join(map(lambda a: f"{a:8.4f}", errors_TT_2)))


if __name__ == '__main__':
    main()
