import os

import modular_splicing

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(modular_splicing.__file__))
)
DATA_DIRECTORY = os.path.join(REPO_ROOT, "spliceai_data", "data")
