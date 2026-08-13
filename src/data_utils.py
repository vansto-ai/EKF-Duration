import numpy as np
import pandas as pd

from src.config import DEFAULT_DURATION_MAP, INDEX_COLUMNS


def compute_daily_return(series: pd.Series) -> pd.Series:
    """基于序列的日收益率，适用于净值或指数。"""
    return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_uploaded_data(nav_file, index_file, report_file):
    """
    读取上传的三个 CSV 文件，并进行基本清理与类型转换。
    """
    nav_df = pd.read_csv(nav_file)
    index_df = pd.read_csv(index_file)
    report_df = pd.read_csv(report_file)

    for df in [nav_df, index_df, report_df]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    # 处理基金净值
    nav_df = nav_df[["fund_id", "date", "nav"]].dropna().sort_values(["fund_id", "date"]).reset_index(drop=True)

    # 处理债券指数
    index_df = index_df[["date"] + INDEX_COLUMNS].dropna().sort_values("date").reset_index(drop=True)

    # 处理披露数据
    report_df = report_df[["fund_id", "date", "leverage", "duration"]].dropna(subset=["fund_id", "date"]).sort_values(["fund_id", "date"]).reset_index(drop=True)
    # duration=0 表示无披露，必须转成 NaN
    report_df["duration"] = report_df["duration"].replace({0: np.nan})

    return nav_df, index_df, report_df


def prepare_fund_data(fund_id: str, nav_df: pd.DataFrame, index_df: pd.DataFrame, report_df: pd.DataFrame):
    """
    为单只基金构造日度样本，并联结债券指数收益与季度/半年度披露观测。
    """
    fund_nav = nav_df[nav_df["fund_id"] == fund_id].copy()
    fund_nav["return"] = compute_daily_return(fund_nav["nav"])

    # 合并指数收益
    index_returns = index_df.copy()
    for col in INDEX_COLUMNS:
        index_returns[col] = compute_daily_return(index_returns[col])
    index_returns = index_returns.rename(columns={c: f"{c}_ret" for c in INDEX_COLUMNS})

    df = fund_nav.merge(index_returns, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # 过滤无效的指数收益
    for col in [f"{c}_ret" for c in INDEX_COLUMNS]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 合并披露观测：杠杆每季度更新，久期仅半年有效
    fund_report = report_df[report_df["fund_id"] == fund_id].copy()
    fund_report = fund_report.sort_values("date")

    # 处理季度/半年度披露为日度观测索引
    df = df.merge(fund_report[["date", "leverage", "duration"]], on="date", how="left")
    df["leverage_obs"] = df["leverage"]
    df["duration_obs"] = df["duration"]

    return df


def get_initial_duration_for_state(report_df: pd.DataFrame, default_duration_map: dict):
    """
    为 EKF 初始状态提供一个合理的起始组合久期。
    优先使用首个有效的披露久期；若没有，则使用默认期限权重时的中位久期水平。
    """
    valid = report_df["duration"].dropna()
    if not valid.empty:
        return float(valid.iloc[0])
    return float(np.median(list(default_duration_map.values())))
