from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from .config import (
    COST_GRID_METRICS_FILE,
    EQUITY_CURVE_FILE,
    LOW_VOL_QUALITY_MOMENTUM_NAME,
    LOW_VOL_QUALITY_RESULTS_DIR,
    LOW_VOL_QUALITY_SELECTION_FILE,
    LOW_VOL_QUALITY_SITE_DIR,
    LOW_VOL_QUALITY_SNAPSHOT_FILE,
    SITE_BASE_COST_BPS,
    SITE_CHART_POINTS,
    SITE_INDEX_FILE,
    STRATEGY_RETURNS_FILE,
)
from .evaluate import compute_drawdown


class MetricRecord(TypedDict):
    cost_bps: int
    periods: int
    total_return: float
    CAGR: float
    annual_vol: float
    Sharpe: float
    max_drawdown: float


class SelectionRecord(TypedDict):
    signal_date: str
    ticker: str
    sector: str
    final_rank: float | None
    status: str
    defensive_entry_signal: bool
    target_weight: float | None
    cash_weight: float | None
    final_score: float | None
    price_composite_score: float | None
    quality_score: float | None
    quality_pass: bool
    momentum_12_1: float | None
    annualized_volatility: float | None
    latest_price: float | None
    above_sma: bool
    volatility_ok: bool
    price_eligible: bool
    return_on_equity: float | None
    profit_margin: float | None
    free_cash_flow_yield: float | None
    debt_to_market_cap: float | None
    reason: str


class ChartPoint(TypedDict):
    date: str
    value: float


class QualitySummary(TypedDict):
    total_tickers: int
    available_tickers: int
    quality_pass_tickers: int
    quality_threshold: float


class LowVolQualityDashboardPayload(TypedDict):
    strategy_name: str
    signal_date: str
    generated_at: str
    base_cost_bps: int
    candidate_count: int
    cash_weight: float
    base_metric: MetricRecord
    metrics: list[MetricRecord]
    quality_summary: QualitySummary
    candidates: list[SelectionRecord]
    rows: list[SelectionRecord]
    equity_points: list[ChartPoint]
    drawdown_points: list[ChartPoint]


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _missing_columns(frame: pd.DataFrame, required_columns: set[str]) -> list[str]:
    return sorted(required_columns.difference(set(frame.columns)))


def _load_selection(path: Path) -> pd.DataFrame:
    _assert_exists(path, "low-vol quality latest selection")
    selection: pd.DataFrame = pd.read_csv(path)
    required_columns: set[str] = {
        "signal_date",
        "ticker",
        "sector",
        "final_rank",
        "status",
        "defensive_entry_signal",
        "target_weight",
        "cash_weight",
        "final_score",
        "price_composite_score",
        "quality_score",
        "quality_pass",
        "momentum_12_1",
        "annualized_volatility",
        "latest_price",
        "above_sma",
        "volatility_ok",
        "price_eligible",
        "return_on_equity",
        "profit_margin",
        "free_cash_flow_yield",
        "debt_to_market_cap",
        "reason",
    }
    missing: list[str] = _missing_columns(selection, required_columns)
    if missing:
        raise ValueError(f"Low-vol quality selection is missing columns: {missing}")

    return selection.sort_values(["defensive_entry_signal", "final_score"], ascending=[False, False]).reset_index(
        drop=True
    )


def _load_quality_snapshot(path: Path) -> pd.DataFrame:
    _assert_exists(path, "quality snapshot")
    quality: pd.DataFrame = pd.read_csv(path)
    required_columns: set[str] = {
        "ticker",
        "quality_score",
        "quality_threshold",
        "quality_pass",
        "quality_data_available",
    }
    missing: list[str] = _missing_columns(quality, required_columns)
    if missing:
        raise ValueError(f"Quality snapshot is missing columns: {missing}")

    return quality


def _load_metrics(path: Path) -> pd.DataFrame:
    _assert_exists(path, "low-vol quality cost metrics")
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
        raise ValueError(f"Low-vol quality cost metrics are missing columns: {missing}")

    return metrics.sort_values("cost_bps").reset_index(drop=True)


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
    if isinstance(value, (int, np.integer)):
        if int(value) == 1:
            return True
        if int(value) == 0:
            return False

    raise TypeError(f"Column '{column}' contains a non-boolean value: {value!r}")


def _json_float(value: object) -> float | None:
    try:
        numeric_value: float = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _selection_record(row: pd.Series) -> SelectionRecord:
    return {
        "signal_date": str(row["signal_date"]),
        "ticker": str(row["ticker"]),
        "sector": str(row["sector"]),
        "final_rank": _json_float(row["final_rank"]),
        "status": str(row["status"]),
        "defensive_entry_signal": _to_bool(row["defensive_entry_signal"], "defensive_entry_signal"),
        "target_weight": _json_float(row["target_weight"]),
        "cash_weight": _json_float(row["cash_weight"]),
        "final_score": _json_float(row["final_score"]),
        "price_composite_score": _json_float(row["price_composite_score"]),
        "quality_score": _json_float(row["quality_score"]),
        "quality_pass": _to_bool(row["quality_pass"], "quality_pass"),
        "momentum_12_1": _json_float(row["momentum_12_1"]),
        "annualized_volatility": _json_float(row["annualized_volatility"]),
        "latest_price": _json_float(row["latest_price"]),
        "above_sma": _to_bool(row["above_sma"], "above_sma"),
        "volatility_ok": _to_bool(row["volatility_ok"], "volatility_ok"),
        "price_eligible": _to_bool(row["price_eligible"], "price_eligible"),
        "return_on_equity": _json_float(row["return_on_equity"]),
        "profit_margin": _json_float(row["profit_margin"]),
        "free_cash_flow_yield": _json_float(row["free_cash_flow_yield"]),
        "debt_to_market_cap": _json_float(row["debt_to_market_cap"]),
        "reason": str(row["reason"]),
    }


def _selection_records(selection: pd.DataFrame) -> list[SelectionRecord]:
    return [_selection_record(row) for _, row in selection.iterrows()]


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
        raise ValueError(f"max_points must be greater than 4. Received: {max_points}.")

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


def _quality_summary(quality_snapshot: pd.DataFrame) -> QualitySummary:
    threshold_values: pd.Series = pd.to_numeric(quality_snapshot["quality_threshold"], errors="coerce").dropna()
    if threshold_values.empty:
        raise ValueError("Quality snapshot has no quality_threshold values.")

    available_tickers: int = sum(
        1
        for value in quality_snapshot["quality_data_available"]
        if _to_bool(value, "quality_data_available")
    )
    quality_pass_tickers: int = sum(
        1
        for value in quality_snapshot["quality_pass"]
        if _to_bool(value, "quality_pass")
    )
    return {
        "total_tickers": int(len(quality_snapshot)),
        "available_tickers": available_tickers,
        "quality_pass_tickers": quality_pass_tickers,
        "quality_threshold": float(threshold_values.iloc[0]),
    }


def _dashboard_payload(
    selection: pd.DataFrame,
    quality_snapshot: pd.DataFrame,
    metrics: pd.DataFrame,
    equity: pd.Series,
    cost_bps: int,
    chart_points: int,
    strategy_name: str,
) -> LowVolQualityDashboardPayload:
    records: list[SelectionRecord] = _selection_records(selection)
    signal_dates: list[str] = sorted({record["signal_date"] for record in records})
    if len(signal_dates) != 1:
        raise ValueError(f"Expected one signal date in selection report. Found: {signal_dates}.")

    candidates: list[SelectionRecord] = [
        record
        for record in records
        if record["defensive_entry_signal"]
    ]
    cash_weight_values: set[float | None] = {record["cash_weight"] for record in records}
    if None in cash_weight_values:
        raise ValueError(f"Selection report contains missing cash_weight values: {cash_weight_values}.")
    cash_weights: list[float] = sorted(float(value) for value in cash_weight_values if value is not None)
    if len(cash_weights) != 1:
        raise ValueError(f"Expected one cash_weight in selection report. Found: {cash_weights}.")

    drawdown: pd.Series = compute_drawdown(equity)
    return {
        "strategy_name": strategy_name,
        "signal_date": signal_dates[0],
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        "base_cost_bps": cost_bps,
        "candidate_count": len(candidates),
        "cash_weight": cash_weights[0],
        "base_metric": _base_metric(metrics, cost_bps),
        "metrics": _metric_records(metrics),
        "quality_summary": _quality_summary(quality_snapshot),
        "candidates": candidates,
        "rows": records,
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
  <title>低波动质量动量看板</title>
  <style>
    :root {
      --ink: #17201d;
      --muted: #5f6862;
      --paper: #f4f5ef;
      --panel: #ffffff;
      --line: #d7ddd4;
      --green: #0a735f;
      --green-soft: #dcefe8;
      --blue: #31589a;
      --blue-soft: #e1e8f8;
      --gold: #9a641c;
      --gold-soft: #f2e4ce;
      --red: #b33d4b;
      --red-soft: #f4dce0;
      --shadow: 0 16px 42px rgba(23, 32, 29, 0.08);
    }

    * { box-sizing: border-box; letter-spacing: 0; }

    body {
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(10, 115, 95, 0.08) 1px, transparent 1px),
        linear-gradient(180deg, rgba(49, 88, 154, 0.06) 1px, transparent 1px),
        var(--paper);
      background-size: 30px 30px;
      font-family: "Avenir Next", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    }

    .shell {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 30px 0 44px;
    }

    .topbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 20px;
      align-items: end;
      padding-bottom: 18px;
      border-bottom: 2px solid var(--ink);
    }

    .eyebrow {
      margin: 0 0 8px;
      color: var(--green);
      font-size: 0.82rem;
      font-weight: 800;
    }

    h1 {
      margin: 0;
      max-width: 820px;
      font-family: "Baskerville", "Songti SC", "STSong", serif;
      font-size: 2.45rem;
      line-height: 1.08;
    }

    .stamp {
      display: grid;
      gap: 6px;
      justify-items: end;
      color: var(--muted);
      white-space: nowrap;
    }

    .stamp strong {
      color: var(--ink);
      font-size: 1.12rem;
    }

    .notice {
      margin: 18px 0;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-left: 5px solid var(--gold);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.76);
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
      min-height: 118px;
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
      font-weight: 800;
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
      line-height: 1.45;
    }

    .grid-two {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
      gap: 16px;
      margin-top: 16px;
    }

    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
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
      font-size: 0.9rem;
      line-height: 1.55;
    }

    .candidate-list {
      display: grid;
      gap: 10px;
      padding: 14px;
    }

    .stock-row {
      display: grid;
      grid-template-columns: 86px 1fr auto;
      gap: 14px;
      align-items: center;
      min-height: 78px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }

    .ticker {
      font-size: 1.32rem;
      font-weight: 900;
    }

    .meta {
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.55;
    }

    .weight {
      min-width: 86px;
      text-align: right;
      font-weight: 900;
      color: var(--green);
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
      color: var(--green);
      background: var(--green-soft);
    }

    .badge.watch {
      color: var(--blue);
      background: var(--blue-soft);
    }

    .badge.risk {
      color: var(--gold);
      background: var(--gold-soft);
    }

    .badge.avoid {
      color: var(--red);
      background: var(--red-soft);
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

    .table-wrap {
      width: 100%;
      overflow-x: auto;
    }

    table {
      width: 100%;
      min-width: 840px;
      border-collapse: collapse;
      background: #fff;
    }

    th, td {
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

    td.numeric, th.numeric {
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

      h1 {
        font-size: 2rem;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">低波动质量动量 · 月度防守选股</p>
        <h1>先排除高波动和趋势破坏，再用质量因子筛出更稳的强势股</h1>
      </div>
      <div class="stamp">
        <span>信号日期</span>
        <strong id="signalDate">--</strong>
        <span id="generatedAt">--</span>
      </div>
    </header>

    <section class="notice">
      历史回测只使用当时已经可见的价格数据；最新候选名单再叠加当前基本面质量快照。这样可以避免把今天看到的基本面数据回填到过去。
    </section>

    <section class="kpi-grid" aria-label="核心指标">
      <div class="kpi">
        <span>当前优先研究</span>
        <strong id="candidateCount">--</strong>
        <small>通过低波动、趋势、动量和质量过滤</small>
      </div>
      <div class="kpi">
        <span>现金预留</span>
        <strong id="cashWeight">--</strong>
        <small>候选股不足时保留现金，避免硬凑满仓</small>
      </div>
      <div class="kpi">
        <span id="sharpeLabel">回测 Sharpe</span>
        <strong id="sharpeValue">--</strong>
        <small>历史段为价格规则防守版</small>
      </div>
      <div class="kpi">
        <span>历史最大回撤</span>
        <strong id="drawdownValue">--</strong>
        <small>用于衡量历史压力</small>
      </div>
    </section>

    <section class="grid-two">
      <article class="panel">
        <div class="panel-header">
          <h2 class="panel-title">当前防守候选</h2>
          <p class="panel-subtitle">参考仓位按候选股数量自动计算，候选不足时剩余部分保持现金。</p>
        </div>
        <div class="candidate-list" id="candidateList"></div>
      </article>

      <article class="panel">
        <div class="panel-header">
          <h2 class="panel-title">质量覆盖</h2>
          <p class="panel-subtitle" id="qualitySubtitle">--</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th class="numeric">排名</th>
                <th>股票</th>
                <th>行业</th>
                <th class="numeric">质量分</th>
                <th class="numeric">动量</th>
                <th class="numeric">波动</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody id="rankingTable"></tbody>
          </table>
        </div>
      </article>
    </section>

    <section class="chart-grid">
      <article class="panel chart-box">
        <div class="panel-header">
          <h2 class="panel-title">策略净值曲线</h2>
          <p class="panel-subtitle" id="equitySubtitle">--</p>
        </div>
        <div class="chart" id="equityChart"></div>
      </article>

      <article class="panel chart-box">
        <div class="panel-header">
          <h2 class="panel-title">历史回撤</h2>
          <p class="panel-subtitle">越接近 0 表示回撤越浅，向下越多表示历史压力越大。</p>
        </div>
        <div class="chart" id="drawdownChart"></div>
      </article>
    </section>

    <section class="grid-two">
      <article class="panel">
        <div class="panel-header">
          <h2 class="panel-title">交易成本压力测试</h2>
          <p class="panel-subtitle">同一套价格防守规则，在不同交易成本假设下的表现。</p>
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
      </article>

      <article class="panel">
        <div class="panel-header">
          <h2 class="panel-title">规则边界</h2>
          <p class="panel-subtitle">这套防守版优先控制风险，再寻找仍有趋势的股票。</p>
        </div>
        <div class="rules">
          <div class="rule">
            <strong>趋势未破</strong>
            <p>只考虑 12-1 动量为正、并且价格站上 200 日均线的股票。</p>
          </div>
          <div class="rule">
            <strong>波动不过高</strong>
            <p>剔除近期波动处于股票池高位的股票，减少追涨回撤风险。</p>
          </div>
          <div class="rule">
            <strong>质量过线</strong>
            <p>用盈利能力、现金流收益和负债压力组成质量分，当前名单必须过线。</p>
          </div>
        </div>
      </article>
    </section>
  </main>

  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
    const pctFormatter = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 1 });
    const pct2Formatter = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 2 });
    const num = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 });
    const score = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 3 });

    function setText(id, value) {
      document.getElementById(id).textContent = value;
    }

    function pct(value) {
      if (!Number.isFinite(value)) return "--";
      return pctFormatter.format(value);
    }

    function pct2(value) {
      if (!Number.isFinite(value)) return "--";
      return pct2Formatter.format(value);
    }

    function scoreText(value) {
      if (!Number.isFinite(value)) return "--";
      return score.format(value);
    }

    function badgeClass(row) {
      if (row.defensive_entry_signal) return "badge primary";
      if (row.price_eligible && !row.quality_pass) return "badge risk";
      if (row.quality_pass && !row.price_eligible) return "badge watch";
      return "badge avoid";
    }

    function renderCandidates() {
      const container = document.getElementById("candidateList");
      if (DATA.candidates.length === 0) {
        container.innerHTML = `<div class="stock-row"><div class="ticker">现金</div><div class="meta">当前没有股票同时通过价格和质量过滤。</div><div class="weight">${pct(DATA.cash_weight)}</div></div>`;
        return;
      }
      container.innerHTML = DATA.candidates.map((row) => `
        <div class="stock-row">
          <div class="ticker">${row.ticker}</div>
          <div>
            <span class="badge primary">${row.status}</span>
            <div class="meta">${row.sector} · 质量分 ${scoreText(row.quality_score)} · 动量 ${pct(row.momentum_12_1)} · 波动 ${pct(row.annualized_volatility)}</div>
          </div>
          <div class="weight">${pct(row.target_weight)}</div>
        </div>
      `).join("");
    }

    function renderRanking() {
      document.getElementById("rankingTable").innerHTML = DATA.rows.slice(0, 30).map((row) => `
        <tr>
          <td class="numeric">${Number.isFinite(row.final_rank) ? Math.round(row.final_rank) : "--"}</td>
          <td><strong>${row.ticker}</strong></td>
          <td>${row.sector || "--"}</td>
          <td class="numeric">${scoreText(row.quality_score)}</td>
          <td class="numeric">${pct(row.momentum_12_1)}</td>
          <td class="numeric">${pct(row.annualized_volatility)}</td>
          <td><span class="${badgeClass(row)}">${row.status}</span></td>
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
          <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d7ddd4" />
          <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d7ddd4" />
          <polygon points="${area}" fill="${fillColor}" opacity="0.72"></polygon>
          <polyline points="${line}" fill="none" stroke="${color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>
          <text x="${pad}" y="${height - 8}" fill="#5f6862" font-size="13">${first.date}</text>
          <text x="${width - pad}" y="${height - 8}" fill="#5f6862" font-size="13" text-anchor="end">${last.date}</text>
          <text x="${pad}" y="20" fill="#5f6862" font-size="13">${formatter(max.value)}</text>
          <text x="${width - pad}" y="20" fill="#5f6862" font-size="13" text-anchor="end">${formatter(min.value)}</text>
        </svg>
      `;
    }

    function render() {
      setText("signalDate", DATA.signal_date);
      setText("generatedAt", `生成时间 ${DATA.generated_at}`);
      setText("candidateCount", `${DATA.candidate_count} 只`);
      setText("cashWeight", pct(DATA.cash_weight));
      setText("sharpeLabel", `${DATA.base_cost_bps} bps 成本 Sharpe`);
      setText("sharpeValue", num.format(DATA.base_metric.Sharpe));
      setText("drawdownValue", pct(DATA.base_metric.max_drawdown));
      setText("qualitySubtitle", `${DATA.quality_summary.available_tickers}/${DATA.quality_summary.total_tickers} 只有足够基本面字段，${DATA.quality_summary.quality_pass_tickers} 只通过质量线，质量线 ${scoreText(DATA.quality_summary.quality_threshold)}。`);
      setText("equitySubtitle", `按 ${DATA.base_cost_bps} bps 交易成本计算，累计收益 ${pct2(DATA.base_metric.total_return)}。`);
      renderCandidates();
      renderRanking();
      renderCostTable();
      renderLineChart("equityChart", DATA.equity_points, "#0a735f", "#dcefe8", (value) => num.format(value));
      renderLineChart("drawdownChart", DATA.drawdown_points, "#b33d4b", "#f4dce0", (value) => pct(value));
    }

    render();
  </script>
</body>
</html>
"""


def render_html(payload: LowVolQualityDashboardPayload) -> str:
    payload_json: str = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    return _html_template().replace("__DASHBOARD_DATA__", payload_json)


def save_site(html: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def build_low_vol_quality_site(
    results_dir: Path,
    site_dir: Path,
    selection_file: str,
    quality_snapshot_file: str,
    metrics_file: str,
    equity_file: str,
    returns_file: str,
    index_file: str,
    cost_bps: int,
    chart_points: int,
    strategy_name: str,
) -> Path:
    selection_path: Path = results_dir / selection_file
    quality_path: Path = results_dir / quality_snapshot_file
    metrics_path: Path = results_dir / metrics_file
    equity_path: Path = results_dir / f"cost_{cost_bps}bps" / equity_file
    returns_path: Path = results_dir / f"cost_{cost_bps}bps" / returns_file

    selection: pd.DataFrame = _load_selection(selection_path)
    quality_snapshot: pd.DataFrame = _load_quality_snapshot(quality_path)
    metrics: pd.DataFrame = _load_metrics(metrics_path)
    equity: pd.Series = _load_series(equity_path, "equity", "low-vol quality equity curve")
    _load_series(returns_path, "strategy_return", "low-vol quality strategy returns")

    payload: LowVolQualityDashboardPayload = _dashboard_payload(
        selection,
        quality_snapshot,
        metrics,
        equity,
        cost_bps,
        chart_points,
        strategy_name,
    )
    out_path: Path = site_dir / index_file
    save_site(render_html(payload), out_path)
    return out_path


def main() -> None:
    out_path: Path = build_low_vol_quality_site(
        LOW_VOL_QUALITY_RESULTS_DIR,
        LOW_VOL_QUALITY_SITE_DIR,
        LOW_VOL_QUALITY_SELECTION_FILE,
        LOW_VOL_QUALITY_SNAPSHOT_FILE,
        COST_GRID_METRICS_FILE,
        EQUITY_CURVE_FILE,
        STRATEGY_RETURNS_FILE,
        SITE_INDEX_FILE,
        SITE_BASE_COST_BPS,
        SITE_CHART_POINTS,
        LOW_VOL_QUALITY_MOMENTUM_NAME,
    )
    print(f"Saved low-vol quality website: {out_path}")


if __name__ == "__main__":
    main()
