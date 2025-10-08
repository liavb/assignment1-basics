import torch
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Union

def load_dataset_mmap(file_path: Union[str, Path], dtype: np.dtype = np.uint16) -> np.memmap:
    """
    Load a dataset using memory mapping for efficient lazy loading.

    Args:
        file_path: Path to the dataset file (.npy or binary file)
        dtype: Data type of the array elements (e.g., np.uint16 for token IDs)

    Returns:
        Memory-mapped numpy array that loads data on-demand

    Example:
        # For .npy files saved with np.save
        dataset = np.load('tokens.npy', mmap_mode='r')

        # For binary files
        dataset = load_dataset_mmap('tokens.bin', dtype=np.uint16)
    """
    file_path = Path(file_path)

    if file_path.suffix == '.npy':
        # Use np.load with memory mapping for .npy files
        return np.load(file_path, mmap_mode='r')
    else:
        # Use np.memmap for binary files
        return np.memmap(file_path, dtype=dtype, mode='r')


def get_batch(dataset: npt.NDArray,
              batch_size: int,
              context_length: int,
              device: str='cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of sequences from a dataset for language modeling.

    This function works efficiently with both regular numpy arrays and memory-mapped
    arrays (np.memmap). When using memory-mapped arrays, only the sampled subsequences
    are loaded into memory, making it suitable for very large datasets.

    Args:
        dataset: 1D numpy array or memory-mapped array of token IDs
        batch_size: Number of sequences to sample
        context_length: Length of each sequence
        device: PyTorch device to place tensors on

    Returns:
        Tuple of (input_sequences, target_sequences) where target is input shifted by 1

    Example:
        # Load dataset with memory mapping
        dataset = load_dataset_mmap('large_dataset.bin', dtype=np.uint16)

        # Validate the dataset
        validate_mmap_dataset(dataset, vocab_size=10000)

        # Sample batches efficiently
        x, y = get_batch(dataset, batch_size=32, context_length=128, device='cuda')
    """
    # Calculate the maximum valid starting index
    max_start_idx = len(dataset) - context_length

    if max_start_idx <= 0:
        raise ValueError(f"Dataset too small: length={len(dataset)}, context_length={context_length}")

    # Randomly sample starting indices for the batch
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    # Pre-allocate tensors with correct shape
    batch_input = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    batch_target = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)

    # Fill the batch tensors
    # For memory-mapped arrays, this will only load the specific slices we need
    for i, start_idx in enumerate(start_indices):
        # Load input sequence (lazy loading for mmap)
        input_slice = dataset[start_idx:start_idx+context_length]
        batch_input[i] = torch.tensor(input_slice, dtype=torch.long, device=device)

        # Load target sequence (shifted by 1, lazy loading for mmap)
        target_slice = dataset[start_idx+1:start_idx+context_length+1]
        batch_target[i] = torch.tensor(target_slice, dtype=torch.long, device=device)

    return batch_input, batch_target



def save_checkpoint(model, optimizer, iteration, out):

    """should dump all the state from the
    first three parameters into the file-like object out. You can use the state_dict method of both
    the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump
    obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to
    have obj be a dictionary, but you can use whatever format you want as long as you can load your
    checkpoint later.
    This function expects the following parameters:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    check_point = {'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'iteration': iteration}
    torch.save(check_point, out)


def load_checkpoint(src, model, optimizer):
    """should load a checkpoint from src (path or file-like object), and then recover the model and optimizer states from that checkpoint. Your
    function should return the iteration number that was saved to the checkpoint. You can use
    torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
    load_state_dict method in both the model and optimizers to return them to their previous
    states.
    This function expects the following parameters:
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer"""

    checkpoint = torch.load(src)
    iteration = checkpoint['iteration']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return iteration

