import os
from huggingface_hub import snapshot_download

snapshot_download(repo_id="uoft-cs/cifar100", repo_type="dataset",
                  cache_dir="./",
                  local_dir_use_symlinks=False, resume_download=True,
                  token="")