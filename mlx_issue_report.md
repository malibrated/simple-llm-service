# MLX-LM Issue: generate_step() receives unexpected 'temp' parameter (RESOLVED)

**Update**: This issue has been resolved. The MLX developers clarified that temperature parameters should be passed via a sampler object created with `make_sampler()`, not as direct parameters to `generate()`.

## Summary
When using `mlx_lm.generate()` with custom parameters, it raises a `TypeError: generate_step() got an unexpected keyword argument 'temp'`. This occurs even when not explicitly passing any temperature-related parameters.

## Environment
- mlx-lm version: 0.25.3
- Python version: 3.12
- macOS with Apple Silicon

## Steps to Reproduce

```python
from mlx_lm import load, generate

# Load any MLX model
model, tokenizer = load("path/to/mlx/model")

# Try to generate text
response = generate(
    model,
    tokenizer,
    prompt="Hello, world!",
    max_tokens=100,
    verbose=False
)
```

## Expected Behavior
The generate function should produce text output without errors.

## Actual Behavior
The function raises an error:
```
TypeError: generate_step() got an unexpected keyword argument 'temp'
```

## Root Cause Analysis

After investigating the MLX-LM source code, I found that:

1. `generate()` passes all kwargs to `stream_generate()` (generate.py:719)
2. `stream_generate()` passes all kwargs to `generate_step()` (generate.py:642)
3. `generate_step()` function signature (generate.py:291) only accepts these parameters:
   - prompt
   - model
   - max_tokens
   - sampler
   - logits_processors
   - max_kv_size
   - prompt_cache
   - prefill_step_size
   - kv_bits
   - kv_group_size
   - quantized_kv_start
   - prompt_progress_callback
   - input_embeddings

4. However, somewhere in the MLX codebase, a 'temp' parameter is being added to the kwargs before they reach `generate_step()`.

## Impact
This makes it impossible to use the MLX generate function with any custom parameters, as it will always fail with this error. Users cannot control temperature, top_p, or other sampling parameters.

## Suggested Fix

1. **Option 1**: Update `generate_step()` to accept and ignore unknown kwargs:
   ```python
   def generate_step(
       prompt: mx.array,
       model: nn.Module,
       *,
       max_tokens: int = 256,
       sampler: Optional[Callable] = None,
       # ... other parameters ...
       **kwargs  # Accept and ignore unknown kwargs
   ):
   ```

2. **Option 2**: Filter kwargs before passing to `generate_step()` in `stream_generate()`:
   ```python
   # In stream_generate(), before calling generate_step
   allowed_params = {'max_tokens', 'sampler', 'logits_processors', ...}
   filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
   token_generator = generate_step(prompt, model, **filtered_kwargs)
   ```

3. **Option 3**: Document which parameters are actually supported and ensure only those are passed through the call chain.

## Workaround
Currently, we have to wrap the generate function to filter out all parameters except `max_tokens` and `verbose`:

```python
def generate_wrapper(model, tokenizer, prompt, **kwargs):
    from mlx_lm import generate
    # Only pass supported parameters
    supported_kwargs = {
        'max_tokens': kwargs.get('max_tokens', 256),
        'verbose': kwargs.get('verbose', False),
    }
    return generate(model, tokenizer, prompt, **supported_kwargs)
```

This workaround prevents the error but also means we cannot use temperature or other sampling parameters.

## Resolution

The correct way to use temperature and sampling parameters with MLX is:

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("path/to/model")

# Create a sampler with desired parameters
sampler = make_sampler(temp=0.7, top_p=0.95, top_k=40)

# Pass the sampler to generate
result = generate(
    model,
    tokenizer,
    prompt="Hello world",
    max_tokens=100,
    sampler=sampler,
    verbose=False
)
```

This approach properly handles temperature and other sampling parameters without causing the error.