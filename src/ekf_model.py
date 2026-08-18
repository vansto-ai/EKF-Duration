import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter
from scipy.optimize import minimize

from src.config import (
    DEFAULT_DAILY_LEVERAGE_CAP,
    DEFAULT_DAILY_WEIGHT_CAP,
    DEFAULT_DURATION_MAP,
    DEFAULT_LEVERAGE_PROCESS_NOISE,
    DEFAULT_OBSERVATION_NOISE_MATRIX,
    DEFAULT_WEIGHT_PROCESS_NOISE,
)


def solve_entropy_initial_weights(target_duration: float, duration_map: dict, eps: float = 1e-12):
    """
    最小熵初始化：
    在 sum(a)=1, dot(a, D)=target_duration, a_i>=0 的约束下，
    找到最接近均匀分布的期限权重。
    """
    D = np.array(list(duration_map.values()), dtype=float)
    n = len(D)

    if target_duration is None or not np.isfinite(target_duration):
        return np.full(n, 1.0 / n)

    def objective(a):
        a = np.clip(a, eps, None)
        return np.sum(a * np.log(a))

    def constraint_sum(a):
        return np.sum(a) - 1.0

    def constraint_duration(a):
        return np.dot(a, D) - target_duration

    constraints = [
        {"type": "eq", "fun": constraint_sum},
        {"type": "eq", "fun": constraint_duration},
    ]
    bounds = [(eps, None)] * n
    x0 = np.full(n, 1.0 / n)

    try:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if result.success:
            a = np.clip(result.x, eps, None)
            a = a / a.sum()
            return a
    except Exception:
        pass

    return np.full(n, 1.0 / n)


def project_state(
    x: np.ndarray,
    prev_x: np.ndarray | None = None,
    leverage_lower=1.0001,
    leverage_upper=1.3999,
    daily_leverage_cap: float = DEFAULT_DAILY_LEVERAGE_CAP,
    daily_weight_cap: float = DEFAULT_DAILY_WEIGHT_CAP,
):
    """
    约束投影：
    1) 杠杆率限定在 (1, 1.4)
    2) 每日杠杆率变化不能超过 daily_leverage_cap
    3) 每个期限权重每日变化不能超过 daily_weight_cap
    4) 权重非负
    5) 权重归一化
    """
    w = float(x[0])
    if prev_x is not None and prev_x.size > 0:
        prev_w = float(prev_x[0])
        delta = w - prev_w
        if abs(delta) > daily_leverage_cap:
            w = prev_w + np.sign(delta) * daily_leverage_cap
    w = float(np.clip(w, leverage_lower, leverage_upper))

    a = np.asarray(x[1:], dtype=float)
    if prev_x is not None and prev_x.size > 1:
        prev_a = np.asarray(prev_x[1:], dtype=float)
        delta_a = a - prev_a
        mask = np.abs(delta_a) > daily_weight_cap
        if np.any(mask):
            a = np.where(mask, prev_a + np.sign(delta_a) * daily_weight_cap, a)
    a = np.clip(a, 0.0, None)

    if a.sum() <= 0:
        a = np.full_like(a, 1.0 / len(a))
    else:
        a = a / a.sum()

    return np.concatenate(([w], a))


class BondFundEKF:
    """
    约束 EKF：
    状态 x = [w, a1, a2, ..., a7]
    观测包括：
    - 日收益观测：R_fund = w * sum(a_i * R_i) + alpha
    - 季度杠杆观测：w
    - 半年度久期观测：sum(a_i * D_i)
    """

    def __init__(
        self,
        duration_map: dict,
        initial_duration: float = None,
        leverage_process_noise: float = DEFAULT_LEVERAGE_PROCESS_NOISE,
        weight_process_noise: float = DEFAULT_WEIGHT_PROCESS_NOISE,
        observation_noise_matrix=None,
        leverage_daily_cap: float = DEFAULT_DAILY_LEVERAGE_CAP,
        weight_daily_cap: float = DEFAULT_DAILY_WEIGHT_CAP,
    ):
        self.duration_map = np.array(list(duration_map.values()), dtype=float)
        self.keys = list(duration_map.keys())
        self.leverage_process_noise = float(leverage_process_noise)
        self.weight_process_noise = float(weight_process_noise)
        self.leverage_daily_cap = float(leverage_daily_cap)
        self.weight_daily_cap = float(weight_daily_cap)

        if observation_noise_matrix is None:
            observation_noise_matrix = DEFAULT_OBSERVATION_NOISE_MATRIX.copy()
        observation_noise_matrix = np.asarray(observation_noise_matrix, dtype=float)
        if observation_noise_matrix.size == 3:
            observation_noise_matrix = np.diag(observation_noise_matrix)
        if observation_noise_matrix.shape != (3, 3):
            raise ValueError("观测噪声矩阵必须为 3x3 矩阵或长度为 3 的对角向量。")
        self.observation_noise_matrix = observation_noise_matrix

        base_weight = solve_entropy_initial_weights(
            target_duration=initial_duration if initial_duration is not None else np.median(self.duration_map),
            duration_map=duration_map,
        )

        init_x = np.concatenate(([1.05], base_weight))

        self.ekf = ExtendedKalmanFilter(dim_x=8, dim_z=1)
        self.ekf.x = np.asarray(init_x, dtype=float)
        self.ekf.P = np.eye(8, dtype=float) * 1e-2
        self.ekf.Q = np.diag([self.leverage_process_noise] + [self.weight_process_noise] * 7)
        self.ekf.R = np.eye(1, dtype=float) * 1e-4
        self.ekf.F = np.eye(8, dtype=float)

    def _project_state(self, prev_x=None):
        self.ekf.x = project_state(
            self.ekf.x,
            prev_x=prev_x,
            daily_leverage_cap=self.leverage_daily_cap,
            daily_weight_cap=self.weight_daily_cap,
        )

    def _set_observation_noise(self, idx: int):
        self.ekf.R = np.eye(1, dtype=float) * float(self.observation_noise_matrix[idx, idx])

    def predict(self):
        """
        状态转移：
        x_t = x_{t-1} + u_t
        采样随机游走。这里简单用常量状态转移矩阵 F = I，
        进程噪声 Q 包含杠杆和权重扰动。
        """
        prev_x = self.ekf.x.copy()
        self.ekf.predict()
        self._project_state(prev_x=prev_x)

    def update_return(self, fund_return: float, index_returns: np.ndarray):
        """
        基金收益观测更新：
        z_t = w_t * sum(a_i * R_i,t) + alpha + eps
        """
        self._set_observation_noise(0)
        prev_x = self.ekf.x.copy()
        index_returns = np.asarray(index_returns, dtype=float)

        def hx(x):
            w = x[0]
            a = x[1:]
            return np.array([w * np.dot(a, index_returns)], dtype=float)

        def HJacobian(x):
            H = np.zeros((1, 8), dtype=float)
            w = x[0]
            a = x[1:]
            H[0, 0] = np.dot(a, index_returns)
            H[0, 1:] = w * index_returns
            return H

        z = np.array([fund_return], dtype=float)
        self.ekf.update(z, HJacobian, hx)
        self._project_state(prev_x=prev_x)

    def update_leverage(self, leverage_obs: float):
        """
        季度杠杆率观测更新：
        z_w = w_t + eta
        """
        self._set_observation_noise(1)
        prev_x = self.ekf.x.copy()

        def hx(x):
            return np.array([x[0]], dtype=float)

        def HJacobian(x):
            H = np.zeros((1, 8), dtype=float)
            H[0, 0] = 1.0
            return H

        z = np.array([leverage_obs], dtype=float)
        self.ekf.update(z, HJacobian, hx)
        self._project_state(prev_x=prev_x)

    def update_duration(self, duration_obs: float):
        """
        半年度久期观测更新：
        z_D = sum(a_i * D_i) + eta
        """
        self._set_observation_noise(2)
        prev_x = self.ekf.x.copy()

        def hx(x):
            a = x[1:]
            return np.array([np.dot(a, self.duration_map)], dtype=float)

        def HJacobian(x):
            H = np.zeros((1, 8), dtype=float)
            H[0, 1:] = self.duration_map
            return H

        z = np.array([duration_obs], dtype=float)
        self.ekf.update(z, HJacobian, hx)
        self._project_state(prev_x=prev_x)

    def current_state(self):
        x = self.ekf.x.copy()
        leverage = float(x[0])
        weights = x[1:]
        asset_duration = float(np.dot(weights, self.duration_map))
        nav_duration = asset_duration if leverage <= 1.0 else float(leverage * asset_duration)
        return {
            "leverage": leverage,
            "weights": weights,
            "asset_duration": asset_duration,
            "nav_duration": nav_duration,
        }


def estimate_daily_fund_states(
    fund_df: pd.DataFrame,
    duration_map: dict,
    report_df: pd.DataFrame,
    leverage_process_noise: float = DEFAULT_LEVERAGE_PROCESS_NOISE,
    weight_process_noise: float = DEFAULT_WEIGHT_PROCESS_NOISE,
    observation_noise_matrix=None,
    leverage_daily_cap: float = DEFAULT_DAILY_LEVERAGE_CAP,
    weight_daily_cap: float = DEFAULT_DAILY_WEIGHT_CAP,
):
    """
    对单个基金执行完整的日度 EKF 滤波估计。
    重点：
    - 先 predict
    - 再 update 今日收益
    - 若当天有季度报告或半年久期，则额外执行观测更新
    """
    model = BondFundEKF(
        duration_map=duration_map,
        initial_duration=get_initial_duration_from_report(report_df, duration_map),
        leverage_process_noise=leverage_process_noise,
        weight_process_noise=weight_process_noise,
        observation_noise_matrix=observation_noise_matrix,
        leverage_daily_cap=leverage_daily_cap,
        weight_daily_cap=weight_daily_cap,
    )

    rows = []

    for _, row in fund_df.iterrows():
        model.predict()

        if np.isfinite(row.get("return", 0.0)):
            index_vector = np.array(
                [
                    row.get("0_1Y_ret", 0.0),
                    row.get("1_3Y_ret", 0.0),
                    row.get("3_5Y_ret", 0.0),
                    row.get("5_7Y_ret", 0.0),
                    row.get("7_10Y_ret", 0.0),
                    row.get("10_25Y_ret", 0.0),
                    row.get("30Y_ret", 0.0),
                ],
                dtype=float,
            )
            model.update_return(row["return"], index_vector)

        leverage_obs = row.get("leverage_obs")
        if pd.notna(leverage_obs):
            model.update_leverage(float(leverage_obs))

        duration_obs = row.get("duration_obs")
        if pd.notna(duration_obs):
            model.update_duration(float(duration_obs))

        state = model.current_state()
        rows.append(
            {
                "fund_id": row["fund_id"],
                "date": row["date"],
                "leverage": state["leverage"],
                "asset_duration": state["asset_duration"],
                "nav_duration": state["nav_duration"],
                "0_1Y_weight": state["weights"][0],
                "1_3Y_weight": state["weights"][1],
                "3_5Y_weight": state["weights"][2],
                "5_7Y_weight": state["weights"][3],
                "7_10Y_weight": state["weights"][4],
                "10_25Y_weight": state["weights"][5],
                "30Y_weight": state["weights"][6],
            }
        )

    return pd.DataFrame(rows)


def get_initial_duration_from_report(report_df: pd.DataFrame, duration_map: dict):
    """
    从报告中取第一个有效久期，作为 EKF 初始长期目标。
    """
    valid = report_df["duration"].dropna()
    if not valid.empty:
        return float(valid.iloc[0])
    return float(np.median(list(duration_map.values())))
