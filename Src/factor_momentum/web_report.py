from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from .config import (
    COST_GRID_METRICS_FILE,
    DEFAULT_UNIVERSE_SLUG,
    EQUITY_CURVE_FILE,
    RESULTS_DIR,
    SELECTION_FILE,
    SELECTION_SMA_WINDOW,
    SELECTION_VOL_WINDOW,
    SITE_BASE_COST_BPS,
    SITE_CHART_POINTS,
    SITE_DIR,
    SITE_INDEX_FILE,
    STRATEGY_RETURNS_FILE,
    UNIVERSES,
)
from .evaluate import compute_drawdown


SignalKind = Literal["primary", "risk", "watch", "avoid"]


class SelectionRecord(TypedDict):
    signal_date: str
    ticker: str
    rank: int
    mom12_1: float
    latest_price: float
    sma: float
    annualized_vol: float
    above_sma: bool
    high_volatility: bool
    strategy_signal: bool
    conservative_entry_signal: bool
    strategy_equal_weight: float
    conservative_equal_weight: float
    reason: str
    status: str
    signal_kind: SignalKind


class MetricRecord(TypedDict):
    cost_bps: int
    periods: int
    total_return: float
    CAGR: float
    annual_vol: float
    Sharpe: float
    max_drawdown: float


class UniverseComparisonRecord(TypedDict):
    universe_slug: str
    universe_name: str
    requested_tickers: int
    used_tickers: int
    dropped_tickers: int
    signal_date: str
    conservative_candidates: int
    strategy_candidates: int
    cost_bps: int
    total_return: float
    CAGR: float
    annual_vol: float
    Sharpe: float
    max_drawdown: float


class ChartPoint(TypedDict):
    date: str
    value: float


class DashboardPayload(TypedDict):
    universe_slug: str
    universe_name: str
    signal_date: str
    generated_at: str
    base_cost_bps: int
    top_candidate_count: int
    strategy_count: int
    rows: list[SelectionRecord]
    metrics: list[MetricRecord]
    base_metric: MetricRecord
    universe_comparison: list[UniverseComparisonRecord]
    equity_points: list[ChartPoint]
    drawdown_points: list[ChartPoint]


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _missing_columns(frame: pd.DataFrame, required_columns: set[str]) -> list[str]:
    return sorted(required_columns.difference(set(frame.columns)))


def _load_selection(path: Path, sma_window: int, vol_window: int) -> pd.DataFrame:
    _assert_exists(path, "latest selection report")
    selection: pd.DataFrame = pd.read_csv(path)
    sma_column: str = f"sma_{sma_window}"
    vol_column: str = f"annualized_vol_{vol_window}d"
    required_columns: set[str] = {
        "signal_date",
        "ticker",
        "rank",
        "mom12_1",
        "latest_price",
        sma_column,
        vol_column,
        "above_sma",
        "high_volatility",
        "strategy_signal",
        "conservative_entry_signal",
        "strategy_equal_weight",
        "conservative_equal_weight",
        "reason",
    }
    missing: list[str] = _missing_columns(selection, required_columns)
    if missing:
        raise ValueError(f"Selection report is missing columns: {missing}")

    return selection.sort_values("rank").reset_index(drop=True)


def _load_metrics(path: Path) -> pd.DataFrame:
    _assert_exists(path, "cost grid metrics")
    metrics: pd.DataFrame = pd.read_csv(path)
    required_columns: set[str] = {
        "cost_bps",
        "periods",
        "total_return",
        "CAGR",
        "annual_vol",
        "Sharpe",
        "max_drawdown",
    }
    missing: list[str] = _missing_columns(metrics, required_columns)
    if missing:
        raise ValueError(f"Cost metrics report is missing columns: {missing}")

    return metrics.sort_values("cost_bps").reset_index(drop=True)


def _load_universe_comparison(path: Path | None) -> list[UniverseComparisonRecord]:
    if path is None:
        return []

    _assert_exists(path, "universe comparison")
    comparison: pd.DataFrame = pd.read_csv(path)
    required_columns: set[str] = {
        "universe_slug",
        "universe_name",
        "requested_tickers",
        "used_tickers",
        "dropped_tickers",
        "signal_date",
        "conservative_candidates",
        "strategy_candidates",
        "cost_bps",
        "total_return",
        "CAGR",
        "annual_vol",
        "Sharpe",
        "max_drawdown",
    }
    missing: list[str] = _missing_columns(comparison, required_columns)
    if missing:
        raise ValueError(f"Universe comparison is missing columns: {missing}")

    return [
        {
            "universe_slug": str(row["universe_slug"]),
            "universe_name": str(row["universe_name"]),
            "requested_tickers": int(row["requested_tickers"]),
            "used_tickers": int(row["used_tickers"]),
            "dropped_tickers": int(row["dropped_tickers"]),
            "signal_date": str(row["signal_date"]),
            "conservative_candidates": int(row["conservative_candidates"]),
            "strategy_candidates": int(row["strategy_candidates"]),
            "cost_bps": int(row["cost_bps"]),
            "total_return": float(row["total_return"]),
            "CAGR": float(row["CAGR"]),
            "annual_vol": float(row["annual_vol"]),
            "Sharpe": float(row["Sharpe"]),
            "max_drawdown": float(row["max_drawdown"]),
        }
        for _, row in comparison.iterrows()
    ]


def _load_series(path: Path, column: str, label: str) -> pd.Series:
    _assert_exists(path, label)
    frame: pd.DataFrame = pd.read_parquet(path)
    if column not in frame.columns:
        raise ValueError(f"{label} must contain column '{column}'. Found: {list(frame.columns)}")

    series: pd.Series = frame[column].dropna().astype(float)
    if series.empty:
        raise ValueError(f"{label} has no usable values in column '{column}'.")

    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _to_bool(value: object, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized: str = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False

    raise TypeError(f"Column '{column}' contains a non-boolean value: {value!r}")


def _signal_kind(row: pd.Series) -> SignalKind:
    conservative_signal: bool = _to_bool(row["conservative_entry_signal"], "conservative_entry_signal")
    strategy_signal: bool = _to_bool(row["strategy_signal"], "strategy_signal")
    high_volatility: bool = _to_bool(row["high_volatility"], "high_volatility")
    above_sma: bool = _to_bool(row["above_sma"], "above_sma")
    momentum: float = float(row["mom12_1"])

    if conservative_signal:
        return "primary"
    if strategy_signal and (high_volatility or not above_sma):
        return "risk"
    if momentum > 0.0 and above_sma:
        return "watch"
    return "avoid"


def _status_text(signal_kind: SignalKind) -> str:
    status_map: dict[SignalKind, str] = {
        "primary": "优先研究",
        "risk": "动量强但风险高",
        "watch": "观察",
        "avoid": "暂不考虑",
    }
    return status_map[signal_kind]


def _selection_record(row: pd.Series, sma_window: int, vol_window: int) -> SelectionRecord:
    sma_column: str = f"sma_{sma_window}"
    vol_column: str = f"annualized_vol_{vol_window}d"
    kind: SignalKind = _signal_kind(row)
    return {
        "signal_date": str(row["signal_date"]),
        "ticker": str(row["ticker"]),
        "rank": int(row["rank"]),
        "mom12_1": float(row["mom12_1"]),
        "latest_price": float(row["latest_price"]),
        "sma": float(row[sma_column]),
        "annualized_vol": float(row[vol_column]),
        "above_sma": _to_bool(row["above_sma"], "above_sma"),
        "high_volatility": _to_bool(row["high_volatility"], "high_volatility"),
        "strategy_signal": _to_bool(row["strategy_signal"], "strategy_signal"),
        "conservative_entry_signal": _to_bool(
            row["conservative_entry_signal"], "conservative_entry_signal"
        ),
        "strategy_equal_weight": float(row["strategy_equal_weight"]),
        "conservative_equal_weight": float(row["conservative_equal_weight"]),
        "reason": str(row["reason"]),
        "status": _status_text(kind),
        "signal_kind": kind,
    }


def _selection_records(selection: pd.DataFrame, sma_window: int, vol_window: int) -> list[SelectionRecord]:
    return [
        _selection_record(row, sma_window, vol_window)
        for _, row in selection.iterrows()
    ]


def _metric_record(row: pd.Series) -> MetricRecord:
    return {
        "cost_bps": int(row["cost_bps"]),
        "periods": int(row["periods"]),
        "total_return": float(row["total_return"]),
        "CAGR": float(row["CAGR"]),
        "annual_vol": float(row["annual_vol"]),
        "Sharpe": float(row["Sharpe"]),
        "max_drawdown": float(row["max_drawdown"]),
    }


def _metric_records(metrics: pd.DataFrame) -> list[MetricRecord]:
    return [_metric_record(row) for _, row in metrics.iterrows()]


def _base_metric(metrics: pd.DataFrame, cost_bps: int) -> MetricRecord:
    matched: pd.DataFrame = metrics[metrics["cost_bps"] == cost_bps]
    if len(matched) != 1:
        raise ValueError(
            f"Expected exactly one metrics row for cost_bps={cost_bps}; found {len(matched)}."
        )

    return _metric_record(matched.iloc[0])


def _chart_points(series: pd.Series, max_points: int) -> list[ChartPoint]:
    sorted_series: pd.Series = series.dropna().astype(float).sort_index()
    if sorted_series.empty:
        raise ValueError("Cannot build chart from an empty series.")
    if max_points <= 4:
        raise ValueError(f"max_points must be greater than 4. Received: {max_points}")

    point_count: int = min(max_points, len(sorted_series))
    required_positions: set[int] = {
        0,
        len(sorted_series) - 1,
        int(np.argmin(sorted_series.values)),
        int(np.argmax(sorted_series.values)),
    }
    sampled_positions: np.ndarray = np.unique(
        np.linspace(0, len(sorted_series) - 1, point_count, dtype=int)
    )
    remaining_slots: int = point_count - len(required_positions)
    if remaining_slots < 0:
        raise ValueError(
            f"max_points={max_points} is too small for required chart extrema positions."
        )

    optional_positions: list[int] = [
        int(position)
        for position in sampled_positions
        if int(position) not in required_positions
    ]
    positions: list[int] = sorted(
        required_positions.union(set(optional_positions[:remaining_slots]))
    )
    return [
        {
            "date": pd.Timestamp(sorted_series.index[position]).date().isoformat(),
            "value": float(sorted_series.iloc[position]),
        }
        for position in positions
    ]


def _dashboard_payload(
    universe_slug: str,
    universe_name: str,
    selection: pd.DataFrame,
    metrics: pd.DataFrame,
    universe_comparison: list[UniverseComparisonRecord],
    equity: pd.Series,
    cost_bps: int,
    chart_points: int,
    sma_window: int,
    vol_window: int,
) -> DashboardPayload:
    records: list[SelectionRecord] = _selection_records(selection, sma_window, vol_window)
    signal_dates: list[str] = sorted({record["signal_date"] for record in records})
    if len(signal_dates) != 1:
        raise ValueError(f"Expected one signal date in selection report. Found: {signal_dates}")

    drawdown: pd.Series = compute_drawdown(equity)
    return {
        "universe_slug": universe_slug,
        "universe_name": universe_name,
        "signal_date": signal_dates[0],
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        "base_cost_bps": cost_bps,
        "top_candidate_count": len([row for row in records if row["signal_kind"] == "primary"]),
        "strategy_count": len([row for row in records if row["strategy_signal"]]),
        "rows": records,
        "metrics": _metric_records(metrics),
        "base_metric": _base_metric(metrics, cost_bps),
        "universe_comparison": universe_comparison,
        "equity_points": _chart_points(equity, chart_points),
        "drawdown_points": _chart_points(drawdown, chart_points),
    }


def _html_template() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>动量选股分析台</title>
  <style>
    :root {
      --ink: #17211f;
      --muted: #5d6764;
      --line: #d6ddd8;
      --paper: #f5f6f1;
      --panel: #ffffff;
      --teal: #087969;
      --teal-soft: #d8eee9;
      --amber: #a9651f;
      --amber-soft: #f4e5ce;
      --coral: #bd3d4f;
      --coral-soft: #f5dbe0;
      --blue: #3556a4;
      --blue-soft: #dfe7fb;
      --shadow: 0 16px 40px rgba(23, 33, 31, 0.08);
    }

    * {
      box-sizing: border-box;
      letter-spacing: 0;
    }

    body {
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(8, 121, 105, 0.08) 1px, transparent 1px),
        linear-gradient(180deg, rgba(53, 86, 164, 0.06) 1px, transparent 1px),
        var(--paper);
      background-size: 28px 28px;
      font-family: "Avenir Next", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    }

    button,
    input {
      font: inherit;
    }

    .shell {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 42px;
    }

    .topbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 20px;
      align-items: end;
      padding: 22px 0 18px;
      border-bottom: 2px solid var(--ink);
    }

    .eyebrow {
      margin: 0 0 8px;
      color: var(--teal);
      font-size: 0.82rem;
      font-weight: 700;
    }

    h1 {
      margin: 0;
      max-width: 760px;
      font-family: "Baskerville", "Songti SC", "STSong", serif;
      font-size: 2.45rem;
      line-height: 1.08;
      font-weight: 700;
    }

    .stamp {
      display: grid;
      gap: 6px;
      justify-items: end;
      color: var(--muted);
      font-size: 0.9rem;
      white-space: nowrap;
    }

    .stamp strong {
      color: var(--ink);
      font-size: 1.1rem;
    }

    .notice {
      margin: 18px 0;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-left: 5px solid var(--amber);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.72);
      color: var(--muted);
      line-height: 1.7;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 20px 0;
    }

    .kpi {
      min-height: 116px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .kpi span {
      display: block;
      color: var(--muted);
      font-size: 0.84rem;
      font-weight: 700;
    }

    .kpi strong {
      display: block;
      margin-top: 12px;
      font-size: 2rem;
      line-height: 1;
    }

    .kpi small {
      display: block;
      margin-top: 10px;
      color: var(--muted);
      line-height: 1.4;
    }

    .grid-two {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
      gap: 16px;
      margin-top: 18px;
    }

    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.9);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
    }

    .panel-title {
      margin: 0;
      font-size: 1.05rem;
      line-height: 1.2;
    }

    .panel-subtitle {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.5;
    }

    .list {
      display: grid;
      gap: 10px;
      padding: 14px;
    }

    .stock-row {
      display: grid;
      grid-template-columns: 84px 1fr auto;
      gap: 14px;
      align-items: center;
      min-height: 76px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }

    .ticker {
      font-size: 1.35rem;
      font-weight: 800;
    }

    .meta {
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.55;
    }

    .weight {
      min-width: 82px;
      text-align: right;
      font-weight: 800;
      color: var(--teal);
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 28px;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 800;
      white-space: nowrap;
    }

    .badge.primary {
      color: var(--teal);
      background: var(--teal-soft);
    }

    .badge.risk {
      color: var(--amber);
      background: var(--amber-soft);
    }

    .badge.watch {
      color: var(--blue);
      background: var(--blue-soft);
    }

    .badge.avoid {
      color: var(--coral);
      background: var(--coral-soft);
    }

    .chart-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
      margin-top: 16px;
    }

    .chart-box {
      min-height: 330px;
      padding: 16px;
    }

    .chart {
      width: 100%;
      min-height: 250px;
    }

    .chart svg {
      width: 100%;
      height: auto;
      display: block;
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }

    .filter-button {
      min-height: 36px;
      padding: 7px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--ink);
      background: #fff;
      cursor: pointer;
    }

    .filter-button.active {
      border-color: var(--ink);
      color: #fff;
      background: var(--ink);
    }

    .search {
      width: 180px;
      min-height: 36px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }

    .table-wrap {
      width: 100%;
      overflow-x: auto;
    }

    table {
      width: 100%;
      min-width: 820px;
      border-collapse: collapse;
      background: #fff;
    }

    th,
    td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
      font-size: 0.9rem;
    }

    th {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      background: #fafbf8;
    }

    td.numeric,
    th.numeric {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .rules {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }

    .rule {
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.86);
    }

    .rule strong {
      display: block;
      margin-bottom: 8px;
    }

    .rule p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }

    .empty {
      padding: 18px;
      color: var(--muted);
      line-height: 1.6;
    }

    @media (max-width: 880px) {
      .topbar,
      .grid-two,
      .chart-grid,
      .rules {
        grid-template-columns: 1fr;
      }

      .stamp {
        justify-items: start;
      }

      .kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      h1 {
        font-size: 2rem;
      }
    }

    @media (max-width: 560px) {
      .shell {
        width: min(100% - 20px, 1180px);
        padding-top: 16px;
      }

      .kpi-grid {
        grid-template-columns: 1fr;
      }

      .stock-row {
        grid-template-columns: 1fr;
      }

      .weight {
        text-align: left;
      }

      .panel-header {
        align-items: flex-start;
        flex-direction: column;
      }

      .controls {
        width: 100%;
      }

      .search {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">12-1 动量因子 · 自动选股面板</p>
        <h1>用规则筛出值得进一步研究的入手候选</h1>
        <p class="panel-subtitle" id="universeName">--</p>
      </div>
      <div class="stamp">
        <span>信号日期</span>
        <strong id="signalDate">--</strong>
        <span id="generatedAt">--</span>
      </div>
    </header>

    <section class="notice">
      这个页面只做规则化筛选：先找动量排名靠前的股票，再排除跌破 200 日均线或近期波动过高的标的。它不是投资建议，也不能替代仓位控制、止损和你自己的基本面判断。
    </section>

    <section class="panel" id="universeComparisonPanel" style="margin-top:16px;">
      <div class="panel-header">
        <div>
          <h2 class="panel-title">股票池对比</h2>
          <p class="panel-subtitle">同一套 12-1 动量策略，在不同股票池上的结果。重点看 Sharpe、回撤、交易成本后收益和候选股数量。</p>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>股票池</th>
              <th class="numeric">使用股票</th>
              <th class="numeric">优先候选</th>
              <th class="numeric">累计收益</th>
              <th class="numeric">年化收益</th>
              <th class="numeric">Sharpe</th>
              <th class="numeric">最大回撤</th>
            </tr>
          </thead>
          <tbody id="universeTable"></tbody>
        </table>
      </div>
    </section>

    <section class="kpi-grid" aria-label="核心指标">
      <div class="kpi">
        <span>优先研究候选</span>
        <strong id="candidateCount">--</strong>
        <small>通过动量、趋势和波动三层过滤</small>
      </div>
      <div class="kpi">
        <span>原始动量组合</span>
        <strong id="strategyCount">--</strong>
        <small>仅按 12-1 动量排名前列筛选</small>
      </div>
      <div class="kpi">
        <span id="sharpeLabel">回测 Sharpe</span>
        <strong id="sharpeValue">--</strong>
        <small>交易成本已计入指定基准情景</small>
      </div>
      <div class="kpi">
        <span>历史最大回撤</span>
        <strong id="drawdownValue">--</strong>
        <small>用于理解策略曾经承受的下跌</small>
      </div>
    </section>

    <section class="grid-two">
      <article class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">优先研究</h2>
            <p class="panel-subtitle">这些股票同时满足：动量前列、价格在 200 日均线上方、近期波动不在高风险组。</p>
          </div>
        </div>
        <div class="list" id="candidateList"></div>
      </article>

      <article class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">动量强但风险高</h2>
            <p class="panel-subtitle">它们进入了原始动量组合，但被趋势或波动过滤器挡下。</p>
          </div>
        </div>
        <div class="list" id="riskList"></div>
      </article>
    </section>

    <section class="chart-grid">
      <article class="panel chart-box">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">策略净值曲线</h2>
            <p class="panel-subtitle" id="equitySubtitle">--</p>
          </div>
        </div>
        <div class="chart" id="equityChart"></div>
      </article>

      <article class="panel chart-box">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">历史回撤</h2>
            <p class="panel-subtitle">越接近 0 表示回撤越浅，向下越多表示历史压力越大。</p>
          </div>
        </div>
        <div class="chart" id="drawdownChart"></div>
      </article>
    </section>

    <section class="panel" style="margin-top:16px;">
      <div class="panel-header">
        <div>
          <h2 class="panel-title">完整股票排名</h2>
          <p class="panel-subtitle">可以按标签筛选，也可以直接搜索股票代码。</p>
        </div>
        <div class="controls" aria-label="排名筛选">
          <button class="filter-button active" data-filter="all" type="button">全部</button>
          <button class="filter-button" data-filter="primary" type="button">优先研究</button>
          <button class="filter-button" data-filter="risk" type="button">风险高</button>
          <button class="filter-button" data-filter="watch" type="button">观察</button>
          <button class="filter-button" data-filter="avoid" type="button">暂不考虑</button>
          <input class="search" id="tickerSearch" type="search" placeholder="搜索代码">
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th class="numeric">排名</th>
              <th>股票</th>
              <th>信号</th>
              <th class="numeric">动量分数</th>
              <th>趋势</th>
              <th class="numeric">63 日年化波动</th>
              <th class="numeric">建议等权</th>
            </tr>
          </thead>
          <tbody id="rankTable"></tbody>
        </table>
      </div>
    </section>

    <section class="panel" style="margin-top:16px;">
      <div class="panel-header">
        <div>
          <h2 class="panel-title">交易成本压力测试</h2>
          <p class="panel-subtitle">同一套原始动量策略，在不同交易成本假设下的表现。</p>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th class="numeric">成本</th>
              <th class="numeric">累计收益</th>
              <th class="numeric">年化收益</th>
              <th class="numeric">年化波动</th>
              <th class="numeric">Sharpe</th>
              <th class="numeric">最大回撤</th>
            </tr>
          </thead>
          <tbody id="costTable"></tbody>
        </table>
      </div>
    </section>

    <section class="rules">
      <div class="rule">
        <strong>1. 先看“优先研究”</strong>
        <p>它是更保守的候选池，适合拿去做下一步基本面、估值和仓位判断。</p>
      </div>
      <div class="rule">
        <strong>2. 不追高波动</strong>
        <p>“动量强但风险高”说明过去涨得强，但近期波动已经偏大，直接追入风险更高。</p>
      </div>
      <div class="rule">
        <strong>3. 每次更新后再看</strong>
        <p>如果当前月份还没结束，信号会随着行情变化；完整月末信号更适合严肃决策。</p>
      </div>
    </section>
  </main>

  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
    const pctFormatter = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 1 });
    const pct2Formatter = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 2 });
    const num = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 });
    let activeFilter = "all";

    function setText(id, value) {
      document.getElementById(id).textContent = value;
    }

    function pct(value) {
      return pctFormatter.format(value);
    }

    function pct2(value) {
      return pct2Formatter.format(value);
    }

    function badge(row) {
      return `<span class="badge ${row.signal_kind}">${row.status}</span>`;
    }

    function trendText(row) {
      return row.above_sma ? "高于 200 日均线" : "低于 200 日均线";
    }

    function rowMeta(row) {
      return `排名 #${row.rank} · 动量 ${pct(row.mom12_1)} · ${trendText(row)} · 波动 ${pct(row.annualized_vol)}`;
    }

    function renderStockList(id, rows, emptyText) {
      const el = document.getElementById(id);
      if (rows.length === 0) {
        el.innerHTML = `<div class="empty">${emptyText}</div>`;
        return;
      }
      el.innerHTML = rows.map((row) => `
        <div class="stock-row">
          <div class="ticker">${row.ticker}</div>
          <div>
            ${badge(row)}
            <div class="meta">${rowMeta(row)}</div>
          </div>
          <div class="weight">${row.conservative_entry_signal ? pct(row.conservative_equal_weight) : "暂不配权"}</div>
        </div>
      `).join("");
    }

    function renderRankTable() {
      const query = document.getElementById("tickerSearch").value.trim().toUpperCase();
      const rows = DATA.rows.filter((row) => {
        const filterOk = activeFilter === "all" || row.signal_kind === activeFilter;
        const queryOk = query.length === 0 || row.ticker.includes(query);
        return filterOk && queryOk;
      });
      const table = document.getElementById("rankTable");
      if (rows.length === 0) {
        table.innerHTML = `<tr><td colspan="7" class="empty">没有匹配的股票。</td></tr>`;
        return;
      }
      table.innerHTML = rows.map((row) => `
        <tr>
          <td class="numeric">${row.rank}</td>
          <td><strong>${row.ticker}</strong></td>
          <td>${badge(row)}</td>
          <td class="numeric">${pct(row.mom12_1)}</td>
          <td>${trendText(row)}</td>
          <td class="numeric">${pct(row.annualized_vol)}</td>
          <td class="numeric">${row.conservative_entry_signal ? pct(row.conservative_equal_weight) : "-"}</td>
        </tr>
      `).join("");
    }

    function renderCostTable() {
      document.getElementById("costTable").innerHTML = DATA.metrics.map((row) => `
        <tr>
          <td class="numeric">${row.cost_bps} bps</td>
          <td class="numeric">${pct(row.total_return)}</td>
          <td class="numeric">${pct(row.CAGR)}</td>
          <td class="numeric">${pct(row.annual_vol)}</td>
          <td class="numeric">${num.format(row.Sharpe)}</td>
          <td class="numeric">${pct(row.max_drawdown)}</td>
        </tr>
      `).join("");
    }

    function renderUniverseTable() {
      const panel = document.getElementById("universeComparisonPanel");
      if (DATA.universe_comparison.length === 0) {
        panel.style.display = "none";
        return;
      }
      document.getElementById("universeTable").innerHTML = DATA.universe_comparison.map((row) => {
        const current = row.universe_slug === DATA.universe_slug;
        const name = current ? `<strong>${row.universe_name}</strong>` : row.universe_name;
        return `
          <tr>
            <td>${name}</td>
            <td class="numeric">${row.used_tickers}/${row.requested_tickers}</td>
            <td class="numeric">${row.conservative_candidates}</td>
            <td class="numeric">${pct(row.total_return)}</td>
            <td class="numeric">${pct(row.CAGR)}</td>
            <td class="numeric">${num.format(row.Sharpe)}</td>
            <td class="numeric">${pct(row.max_drawdown)}</td>
          </tr>
        `;
      }).join("");
    }

    function scalePoints(points, width, height, pad) {
      const values = points.map((point) => point.value);
      let min = Math.min(...values);
      let max = Math.max(...values);
      if (min === max) {
        min = min - 1;
        max = max + 1;
      }
      const xStep = points.length > 1 ? (width - pad * 2) / (points.length - 1) : 0;
      return points.map((point, index) => {
        const x = pad + index * xStep;
        const y = height - pad - ((point.value - min) / (max - min)) * (height - pad * 2);
        return { x, y, value: point.value, date: point.date };
      });
    }

    function renderLineChart(id, points, color, fillColor, formatter) {
      const width = 760;
      const height = 260;
      const pad = 34;
      const scaled = scalePoints(points, width, height, pad);
      const line = scaled.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
      const area = `${pad},${height - pad} ${line} ${width - pad},${height - pad}`;
      const first = points[0];
      const last = points[points.length - 1];
      const min = points.reduce((a, b) => a.value < b.value ? a : b);
      const max = points.reduce((a, b) => a.value > b.value ? a : b);
      document.getElementById(id).innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="趋势图">
          <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d6ddd8" />
          <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d6ddd8" />
          <polygon points="${area}" fill="${fillColor}" opacity="0.72"></polygon>
          <polyline points="${line}" fill="none" stroke="${color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>
          <text x="${pad}" y="${height - 8}" fill="#5d6764" font-size="13">${first.date}</text>
          <text x="${width - pad}" y="${height - 8}" fill="#5d6764" font-size="13" text-anchor="end">${last.date}</text>
          <text x="${pad}" y="20" fill="#5d6764" font-size="13">${formatter(max.value)}</text>
          <text x="${width - pad}" y="20" fill="#5d6764" font-size="13" text-anchor="end">${formatter(min.value)}</text>
        </svg>
      `;
    }

    function wireFilters() {
      document.querySelectorAll(".filter-button").forEach((button) => {
        button.addEventListener("click", () => {
          activeFilter = button.dataset.filter;
          document.querySelectorAll(".filter-button").forEach((item) => item.classList.remove("active"));
          button.classList.add("active");
          renderRankTable();
        });
      });
      document.getElementById("tickerSearch").addEventListener("input", renderRankTable);
    }

    function render() {
      const candidates = DATA.rows.filter((row) => row.signal_kind === "primary");
      const riskRows = DATA.rows.filter((row) => row.signal_kind === "risk");
      setText("universeName", `当前股票池：${DATA.universe_name}`);
      setText("signalDate", DATA.signal_date);
      setText("generatedAt", `生成时间 ${DATA.generated_at}`);
      setText("candidateCount", `${DATA.top_candidate_count} 只`);
      setText("strategyCount", `${DATA.strategy_count} 只`);
      setText("sharpeLabel", `${DATA.base_cost_bps} bps 成本 Sharpe`);
      setText("sharpeValue", num.format(DATA.base_metric.Sharpe));
      setText("drawdownValue", pct(DATA.base_metric.max_drawdown));
      setText("equitySubtitle", `按 ${DATA.base_cost_bps} bps 交易成本计算，累计收益 ${pct2(DATA.base_metric.total_return)}。`);
      renderStockList("candidateList", candidates, "当前没有通过保守过滤的候选。");
      renderStockList("riskList", riskRows, "当前没有被风险过滤挡下的强动量股票。");
      renderUniverseTable();
      renderLineChart("equityChart", DATA.equity_points, "#087969", "#d8eee9", (value) => num.format(value));
      renderLineChart("drawdownChart", DATA.drawdown_points, "#bd3d4f", "#f5dbe0", (value) => pct(value));
      renderRankTable();
      renderCostTable();
      wireFilters();
    }

    render();
  </script>
</body>
</html>
"""


def render_html(payload: DashboardPayload) -> str:
    payload_json: str = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return _html_template().replace("__DASHBOARD_DATA__", payload_json)


def save_site(html: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def build_site(
    results_dir: Path,
    site_dir: Path,
    selection_file: str,
    metrics_file: str,
    equity_file: str,
    returns_file: str,
    index_file: str,
    cost_bps: int,
    chart_points: int,
    sma_window: int,
    vol_window: int,
    comparison_path: Path | None,
    universe_slug: str,
    universe_name: str,
) -> Path:
    selection_path: Path = results_dir / selection_file
    metrics_path: Path = results_dir / metrics_file
    equity_path: Path = results_dir / f"cost_{cost_bps}bps" / equity_file
    returns_path: Path = results_dir / f"cost_{cost_bps}bps" / returns_file

    selection: pd.DataFrame = _load_selection(selection_path, sma_window, vol_window)
    metrics: pd.DataFrame = _load_metrics(metrics_path)
    universe_comparison: list[UniverseComparisonRecord] = _load_universe_comparison(comparison_path)
    equity: pd.Series = _load_series(equity_path, "equity", "base cost equity curve")
    _load_series(returns_path, "strategy_return", "base cost strategy returns")

    payload: DashboardPayload = _dashboard_payload(
        universe_slug,
        universe_name,
        selection,
        metrics,
        universe_comparison,
        equity,
        cost_bps,
        chart_points,
        sma_window,
        vol_window,
    )
    out_path: Path = site_dir / index_file
    save_site(render_html(payload), out_path)
    return out_path


def main() -> None:
    universe_name: str = UNIVERSES[DEFAULT_UNIVERSE_SLUG]["name"]
    out_path: Path = build_site(
        RESULTS_DIR,
        SITE_DIR,
        SELECTION_FILE,
        COST_GRID_METRICS_FILE,
        EQUITY_CURVE_FILE,
        STRATEGY_RETURNS_FILE,
        SITE_INDEX_FILE,
        SITE_BASE_COST_BPS,
        SITE_CHART_POINTS,
        SELECTION_SMA_WINDOW,
        SELECTION_VOL_WINDOW,
        None,
        DEFAULT_UNIVERSE_SLUG,
        universe_name,
    )
    print(f"Saved local website: {out_path}")


if __name__ == "__main__":
    main()
