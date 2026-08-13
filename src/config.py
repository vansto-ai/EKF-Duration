from typing import List

import numpy as np

# 默认期限久期配置，单位：年
DEFAULT_DURATION_MAP = {
    "0_1Y": 0.50,
    "1_3Y": 2.00,
    "3_5Y": 4.00,
    "5_7Y": 6.00,
    "7_10Y": 8.50,
    "10_25Y": 15.00,
    "30Y": 25.00,
}

INDEX_COLUMNS: List[str] = [
    "0_1Y",
    "1_3Y",
    "3_5Y",
    "5_7Y",
    "7_10Y",
    "10_25Y",
    "30Y",
]

ALLOCATION_COLUMNS: List[str] = [
    "0_1Y_weight",
    "1_3Y_weight",
    "3_5Y_weight",
    "5_7Y_weight",
    "7_10Y_weight",
    "10_25Y_weight",
    "30Y_weight",
]

# 按照图示默认值设置
DEFAULT_LEVERAGE_PROCESS_NOISE = 0.05
DEFAULT_WEIGHT_PROCESS_NOISE = 0.05
DEFAULT_OBSERVATION_NOISE_MATRIX = np.array(
    [
        [1e-6, 0.0, 0.0],
        [0.0, 0.0025, 0.0],
        [0.0, 0.0, 0.2],
    ],
    dtype=float,
)
