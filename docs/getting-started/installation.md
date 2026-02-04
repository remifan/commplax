# Installation

**Requirements:** Python >= 3.11

Editable install is recommended for easier testing and debugging.

```bash
git clone https://github.com/remifan/commplax.git
cd commplax
```

## CPU only

Works on Windows, Linux, and Mac:

```bash
pip install -e "."
```

## CUDA 12

Linux only (Windows not supported; WSL2 may work but untested):

```bash
pip install -e ".[cuda12]"
```

Verify GPU detection:

```python
import jax
print(jax.devices())
```

!!! note
    CUDA 13 with JAX >= 0.7 is untested. See [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for more details.
