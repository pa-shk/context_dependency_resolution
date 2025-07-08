import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Sets the seed for all random number generators to ensure reproducible results.
    
    Args:
        seed: The integer value to use as the seed for all generators
    
    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
