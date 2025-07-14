import torch
import torch.nn.functional as F

def _split_tensor_equal(x: torch.Tensor, n_chunks: int, dim: int = 0,
                        pad_value: float = 0.0):
    """
    Split `x` into `n_chunks` **equal-length** slices along `dim`,
    padding with `pad_value` so every slice has size `ceil(len/n_chunks)`.
    """
    length      = x.size(dim)
    chunk_size  = (length + n_chunks - 1) // n_chunks          # ceil div
    needed_pad  = chunk_size * n_chunks - length               # 0 … chunk_size-1

    if needed_pad:                                             # only if we really need it
        # Build the (left, right, left, right, …) pad spec expected by F.pad
        pad = [0] * (2 * x.dim())
        pad[-(2 * dim + 1)] = needed_pad                       # pad on the *right* side of `dim`
        x = F.pad(x, pad, value=pad_value)

    # Reshape to (..., n_chunks, chunk_size, ...)
    new_shape = list(x.shape)
    new_shape[dim] = n_chunks
    new_shape.insert(dim + 1, chunk_size)
    return x.view(*new_shape).unbind(dim)                      # returns a tuple of tensors

def split_batch(batch, n_chunks, dim=0):
    """Recursively split a nested batch into `n_chunks` along dim."""
    if torch.is_tensor(batch):
        return _split_tensor_equal(batch, n_chunks, dim)   # <= use helper above
    
    elif isinstance(batch, (list, tuple)):
        length   = len(batch)
        step     = (length + n_chunks - 1) // n_chunks    # ceil div
        splits   = [batch[i*step : (i+1)*step] for i in range(n_chunks)]
        # pad with empty lists/tuples if needed so we always have n_chunks
        while len(splits) < n_chunks:
            splits.append(type(batch)())
        return splits
    
    elif isinstance(batch, dict):
        chunks = [dict() for _ in range(n_chunks)]
        for k, v in batch.items():
            v_chunks = split_batch(v, n_chunks, dim)
            for i in range(n_chunks):
                chunks[i][k] = v_chunks[i]
        return chunks
    else:
        raise TypeError(f"Cannot split object of type {type(batch)}")