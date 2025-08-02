This project involves CUDA accelerations for cross-Hadamard products. You have to compile this CUDA module right here since original .so file might not match your environment.

To compile CUDA module as standalone pytorch module:

```bash
python setup.py build_ext --inplace
```

It will generate a shared object file (e.g., `cdt_extensions.cpython-xx.so`) in the `cuda_modules` directory.

To use the compiled module in your project, copy `cdt_extensions.pyi` and `cdt_extensions.cpython-xx.so` to your project directory, then import it as follows:

```python
from cdt_extensions import *
```