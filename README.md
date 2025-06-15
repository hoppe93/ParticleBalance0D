# 0D particle balance code
This repository contains a 0D plasma particle balance code, implemented in
Python. The code was used for the calculations presented in
[Hoppe *et al* PPCF **67** (2025)](https://doi.org/10.1088/1361-6587/adbcd5).

## Installation
This code requires some scripts from
[DREAM](https://github.com/chalmersplasmatheory/DREAM). While it is not
necessary to compile DREAM to use this 0D code, it is necessary to download
the ADAS data used by DREAM. To do this, the following steps are needed:

1. Clone the [GitHub](https://github.com/chalmersplasmatheory/DREAM) repository for DREAM.
2. Run the Python script ``tools/get_adas.py``.

Before loading this package, it is also necessary to add the ``tools/``
directory of DREAM to the ``PYTHONPATH`` environment variable:
```bash
export PYTHONPATH="/path/to/DREAM/tools:$PYTHONPATH"


## Usage
A number of examples of how to use the code can be found under the ``examples/``
directory.
```

