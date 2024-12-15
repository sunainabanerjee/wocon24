import numpy as np
from typing import Callable, Optional, Dict, Tuple, Union, List


class LinearDataGenerator2D:
    def __init__(self,
                 slope: float,
                 intercept: float,
                 ):
        self._slope = slope
        self._intercept = intercept

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._slope * x + self._intercept


def generate_samples_with_uncertainty(number_of_samples: int,
                                      sample_dimension: int = 2,
                                      base_function: Optional[Callable] = None,
                                      uncertainty: float = 0.8,
                                      data_range: Tuple[float, float] = None,
                                      seed: int = 1,
                                      ) -> Dict[str, np.ndarray]:
    if (data_range is None) or not isinstance(data_range, (list, tuple)) or (len(data_range) != 2):
        data_range = 0, 10

    rng = np.random.default_rng(seed=seed)
    if (sample_dimension == 2) and (base_function is None):
        slope = 5 * rng.random() - 2.5
        intercept = 0.
        base_function = LinearDataGenerator2D(slope=slope,
                                              intercept=intercept)
    elif base_function is None:
        raise ValueError(f"Error: please specify data-generator function!")

    x = np.linspace(data_range[0],
                    data_range[1],
                    number_of_samples)[..., np.newaxis]
    y = base_function(x)
    noise = rng.normal(0, scale=uncertainty, size=np.prod(y.shape)).reshape(y.shape)
    y_ = y + noise

    return dict(true_observation=np.concatenate([x, y], axis=-1),
                noisy_observation=np.concatenate([x, y_], axis=-1))


def select_random_data_partition(data_length: int,
                                 partition_sizes: Union[int, List[int]],
                                 seed: int = 1):

    if isinstance(partition_sizes, int):
        partition_sizes = [partition_sizes]

    if any([sz > data_length for sz in partition_sizes]):
        raise ValueError(f"Error: partition size must be smaller than total data size!")

    rng = np.random.default_rng(seed=seed)
    return [np.sort(rng.choice(data_length, sz, replace=False)).tolist()
            for sz in partition_sizes]



