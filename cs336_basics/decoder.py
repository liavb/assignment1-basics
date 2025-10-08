import torch
import torch.nn.functional as F
from typing import Optional, Union, List
import numpy as np

from cs336_basics.transformer.model_layers import TransformerLM
from cs336_basics.transformer.nn_utils import softmax


def temperature_scaled_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits and then softmax.

    Args:
        logits: Raw logits from the model, shape (..., vocab_size)
        temperature: Temperature parameter τ. Lower values make distribution more peaked.
                    temperature → 0 approaches argmax (greedy), temperature → ∞ approaches uniform

    Returns:
        Temperature-scaled probabilities, same shape as logits
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Apply softmax using our custom implementation (note: takes dimension as second argument, not keyword)
    return softmax(scaled_logits, -1)


def top_p_sampling(probs: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to a probability distribution.

    Args:
        probs: Probability distribution, shape (..., vocab_size)
        p: Threshold for nucleus sampling. Should be between 0 and 1.

    Returns:
        Modified probability distribution with low-probability tokens set to 0
    """
    if not 0 < p <= 1:
        raise ValueError("p must be between 0 and 1")

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cutoff: we want the smallest set V(p) such that sum(V(p)) >= p
    # We keep tokens up to and including the first one that makes cumsum >= p
    cutoff_mask = cumulative_probs <= p

    # Always keep at least the first (highest probability) token
    cutoff_mask[..., 0] = True

    # Zero out probabilities for tokens not in the nucleus
    sorted_probs_filtered = sorted_probs * cutoff_mask.float()

    # Create a tensor to scatter the filtered probabilities back to original order
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs_filtered)

    # Renormalize to ensure probabilities sum to 1
    total_prob = filtered_probs.sum(dim=-1, keepdim=True)
    filtered_probs = filtered_probs / (total_prob + 1e-10)  # Add small epsilon to avoid division by zero

    return filtered_probs


def sample_from_distribution(probs: torch.Tensor) -> torch.Tensor:
    """
    Sample token indices from a probability distribution.

    Args:
        probs: Probability distribution, shape (..., vocab_size)

    Returns:
        Sampled token indices, shape (...)
    """
    # Use torch.multinomial for sampling
    # Reshape to 2D for multinomial, then reshape back
    original_shape = probs.shape[:-1]
    probs_2d = probs.view(-1, probs.size(-1))

    # Sample one token per distribution
    sampled_indices = torch.multinomial(probs_2d, num_samples=1).squeeze(-1)

    # Reshape back to original batch dimensions
    return sampled_indices.view(original_shape)


def decode_one_step(
    model: TransformerLM,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Perform one step of autoregressive decoding.

    Args:
        model: The transformer language model
        input_ids: Input token sequence, shape (batch_size, seq_len)
        temperature: Temperature for scaling logits
        top_p: If provided, apply nucleus sampling with this threshold
        device: Device to run computation on

    Returns:
        Next token predictions, shape (batch_size,)
    """
    model.eval()

    with torch.no_grad():
        # Get logits from model
        logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)

        # Get logits for the last position (next token prediction)
        next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

        # Apply temperature scaling and softmax
        probs = temperature_scaled_softmax(next_token_logits, temperature)

        # Apply top-p sampling if requested
        if top_p is not None:
            probs = top_p_sampling(probs, top_p)

        # Sample next tokens
        next_tokens = sample_from_distribution(probs)

        return next_tokens


def generate_text(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = False
) -> torch.Tensor:
    """
    Generate text autoregressively from a prompt.

    Args:
        model: The transformer language model
        prompt_ids: Initial token sequence(s), shape (batch_size, prompt_len)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for scaling logits (default: 1.0)
        top_p: If provided, apply nucleus sampling with this threshold
        eos_token_id: Token ID for end-of-sequence. Generation stops when this is produced.
        pad_token_id: Token ID for padding (used if sequences have different lengths)
        device: Device to run computation on
        verbose: If True, print generation progress

    Returns:
        Generated sequences including the original prompt, shape (batch_size, prompt_len + generated_len)
    """
    model.eval()

    # Ensure input is on the correct device
    prompt_ids = prompt_ids.to(device)
    model = model.to(device)

    batch_size, prompt_len = prompt_ids.shape

    # Get max context length from the model's transformer blocks
    try:
        max_context_length = model.transformer_blocks[0].mha.max_seq_len
    except (AttributeError, IndexError):
        # Fallback to a reasonable default if we can't get the actual max length
        max_context_length = 512
        if verbose:
            print(f"Warning: Could not determine model's max context length, using default: {max_context_length}")

    if verbose:
        print(f"Starting generation with prompt length {prompt_len}, max new tokens: {max_new_tokens}")
        print(f"Model max context length: {max_context_length}")

    # Initialize the sequence with the prompt
    generated_ids = prompt_ids.clone()

    # Track which sequences have finished (generated EOS token)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        current_length = generated_ids.size(1)

        # Check if we've reached the maximum context length
        if current_length >= max_context_length:
            if verbose:
                print(f"Reached maximum context length ({max_context_length})")
            break

        # Prepare input for the model (use only the last context_length tokens if needed)
        if current_length > max_context_length:
            model_input = generated_ids[:, -max_context_length:]
        else:
            model_input = generated_ids

        # Generate next token
        next_tokens = decode_one_step(
            model=model,
            input_ids=model_input,
            temperature=temperature,
            top_p=top_p,
            device=device
        )

        # For finished sequences, use pad token instead of generated token
        if pad_token_id is not None:
            next_tokens = torch.where(finished, pad_token_id, next_tokens)

        # Append next tokens to the sequence
        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)

        # Check for EOS token
        if eos_token_id is not None:
            finished = finished | (next_tokens == eos_token_id)

            # If all sequences are finished, stop generation
            if finished.all():
                if verbose:
                    print(f"All sequences finished at step {step + 1}")
                break

        if verbose and (step + 1) % 10 == 0:
            print(f"Generated {step + 1} tokens...")

    if verbose:
        final_length = generated_ids.size(1) - prompt_len
        print(f"Generation complete. Generated {final_length} new tokens.")

    return generated_ids


def generate_text_simple(
    model: TransformerLM,
    prompt_ids: Union[torch.Tensor, List[int]],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cpu"
) -> List[int]:
    """
    Simple wrapper for generating text from a single prompt.

    Args:
        model: The transformer language model
        prompt_ids: Initial token sequence, either as tensor or list of ints
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for scaling logits
        top_p: If provided, apply nucleus sampling with this threshold
        eos_token_id: Token ID for end-of-sequence
        device: Device to run computation on

    Returns:
        Generated sequence as a list of token IDs
    """
    # Convert input to tensor if needed
    if isinstance(prompt_ids, list):
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)

    # Add batch dimension if needed
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    # Generate text
    generated = generate_text(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device
    )

    # Return as list of ints (remove batch dimension)
    return generated[0].tolist()


def decode_with_tokenizer(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    device: str = "cpu",
    verbose: bool = False
) -> str:
    """
    High-level function to generate text from a string prompt using a tokenizer.

    Args:
        model: The transformer language model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input text prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for scaling logits
        top_p: If provided, apply nucleus sampling with this threshold
        device: Device to run computation on
        verbose: If True, print generation details

    Returns:
        Generated text as a string
    """
    # Encode the prompt
    prompt_ids = tokenizer.encode(prompt)

    if verbose:
        print(f"Prompt: '{prompt}'")
        print(f"Prompt tokens: {prompt_ids}")

    # Get EOS token ID if available
    eos_token_id = None
    if hasattr(tokenizer, 'special_tokens') and tokenizer.special_tokens:
        # Look for common EOS token names
        eos_candidates = ['<|endoftext|>', '</s>', '<eos>']
        for candidate in eos_candidates:
            if candidate in tokenizer.special_tokens:
                eos_token_id = tokenizer.encode(candidate)[0]
                break

    # Generate text
    generated_ids = generate_text_simple(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device
    )

    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids)

    if verbose:
        new_tokens = generated_ids[len(prompt_ids):]
        print(f"Generated tokens: {new_tokens}")
        print(f"Generated text: '{generated_text}'")

    return generated_text


# Example usage and testing functions
def test_temperature_scaling():
    """Test temperature scaling with different values."""
    print("Testing temperature scaling...")

    # Create some example logits
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]])

    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    for temp in temperatures:
        probs = temperature_scaled_softmax(logits, temp)
        print(f"Temperature {temp}: {probs.squeeze().tolist()}")


def test_top_p_sampling():
    """Test top-p sampling with different thresholds."""
    print("\nTesting top-p sampling...")

    # Create some example probabilities
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.05, 0.025, 0.025]])

    p_values = [0.5, 0.7, 0.9, 1.0]

    for p in p_values:
        filtered_probs = top_p_sampling(probs, p)
        print(f"Top-p {p}: {filtered_probs.squeeze().tolist()}")


if __name__ == "__main__":
    test_temperature_scaling()
    test_top_p_sampling()
