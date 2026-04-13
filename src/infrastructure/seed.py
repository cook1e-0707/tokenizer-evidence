from __future__ import annotations

import os
import random


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, bool]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = True
    except ImportError:
        pass

    torch_seeded = False
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch_seeded = True
    except ImportError:
        pass

    return {
        "python": True,
        "numpy": numpy_seeded,
        "torch": torch_seeded,
    }
