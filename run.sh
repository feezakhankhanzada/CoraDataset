#!/bin/sh

pip install torch --no-cache-dir

python -c "import torch; print(torch.__version__)"

python -c "import torch; print(torch.version.cuda)"

pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

pip install torch-geometric
pip install torch_sparse --no-cache-dir
pip install torch_scatter

python run.py