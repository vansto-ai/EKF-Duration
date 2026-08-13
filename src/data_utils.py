import numpy as np
import pandas as pd

from src.config import DEFAULT_DURATION_MAP, INDEX_COLUMNS


def _normalize_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """兼容常见债券指数列名写法，把其标准化为项目要求的字段名。"""
    rename_map = {
        "0-1Y": "0_1Y",
        "0_1Y": "0_1Y",
        "0-1年": "0_1Y",
        "0_1年": "0_1Y",
        "1-3Y": "1_3Y",
        "1_3Y": "1_3Y",
        "1-3年": "1_3Y",
        "1_3年": "1_3Y",
        "3-5Y": "3_5Y",
        "3_5Y": "3_5Y",
        "3-5年": "3_5Y",
        "3_5年": "3_5Y",
        "5-7Y": "5_7Y",
        "5_7Y": "5_7Y",
        "5-7年": "5_7Y",
        "5_7年": "5_7Y",
        "7-10Y": "7_10Y",
        "7_10Y": "7_10Y",
        "7-10年": "7_10Y",
        "7_10年": "7_10Y",
        "10-25Y": "10_25Y",
        "10_25Y": "10_25Y",
        "10-25年": "10_25Y",
        "10_25年": "10_25Y",
        "30Y": "30Y",
        "30年": "30Y",
    }

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.rename(columns={c: rename_map.get(c, c) for c in out.columns})

    missing = [c for c in INDEX_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"债券指数文件缺少列：{missing}，请检查列名是否为 {INDEX_COLUMNS}")

    return out


def _clean_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """清洗日期列中的非日期说明行（例如“数据来源：Wind”），并统一转换成 datetime。"""
    if "date" not in df.columns:
        return df

    out = df.copy()
    out["date"] = out["date"].astype(str).str.strip()

    # 仅保留真正包含日期的行，过滤说明类文本
    valid_mask = out["date"].str.contains(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{4}/\d{1,2}/\d{1,2}$|^\d{8}$",
        regex=True,
        na=False,
    )
    out = out[valid_mask].copy()

    try:
        out["date"] = pd.to_datetime(out["date"], format="mixed", errors="coerce")
    except TypeError:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


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

    # 标准化列名，并删除非数据行（如“数据来源：Wind”）
    nav_df = _clean_date_column(nav_df)
    index_df = _clean_date_column(_normalize_index_columns(index_df))
    report_df = _clean_date_column(report_df)

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
