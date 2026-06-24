from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from .config import (
    COST_GRID_METRICS_FILE,
    EQUITY_CURVE_FILE,
    ETF_DUAL_MOMENTUM_LOOKBACK_MONTHS,
    ETF_DUAL_MOMENTUM_NAME,
    ETF_DUAL_MOMENTUM_RANKING_FILE,
    ETF_DUAL_MOMENTUM_RESULTS_DIR,
    ETF_DUAL_MOMENTUM_SIGNALS_FILE,
    ETF_DUAL_MOMENTUM_SITE_DIR,
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


class RankingRecord(TypedDict):
    ticker: str
    rank: int
    momentum: float
    selected: bool
    target_weight: float
    role: str


class RotationRecord(TypedDict):
    signal_date: str
    selected_ticker: str
    selected_score: float
    defensive_signal: bool


class ChartPoint(TypedDict):
    date: str
    value: float


class EtfDashboardPayload(TypedDict):
    strategy_name: str
    signal_date: str
    generated_at: str
    lookback_months: int
    selected_ticker: str
    selected_score: float
    defensive_signal: bool
    base_cost_bps: int
    base_metric: MetricRecord
    metrics: list[MetricRecord]
    ranking: list[RankingRecord]
    rotations: list[RotationRecord]
    equity_points: list[ChartPoint]
    drawdown_points: list[ChartPoint]


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _missing_columns(frame: pd.DataFrame, required_columns: set[str]) -> list[str]:
    return sorted(required_columns.difference(set(frame.columns)))


def _load_signals(path: Path) -> pd.DataFrame:
    _assert_exists(path, "ETF dual momentum signals")
    signals: pd.DataFrame = pd.read_csv(path)
    required_columns: set[str] = {
        "signal_date",
        "selected_ticker",
        "selected_score",
        "defensive_signal",
    }
    missing: list[str] = _missing_columns(signals, required_columns)
    if missing:
        raise ValueError(f"ETF dual momentum signals are missing columns: {missing}")

    return signals.sort_values("signal_date").reset_index(drop=True)


def _load_ranking(path: Path, lookback_months: int) -> pd.DataFrame:
    _assert_exists(path, "ETF dual momentum latest ranking")
    ranking: pd.DataFrame = pd.read_csv(path)
    momentum_column: str = f"momentum_{lookback_months}m"
    required_columns: set[str] = {
        "ticker",
        "rank",
        momentum_column,
        "selected",
        "target_weight",
        "role",
    }
    missing: list[str] = _missing_columns(ranking, required_columns)
    if missing:
        raise ValueError(f"ETF dual momentum ranking is missing columns: {missing}")

    return ranking.sort_values("rank").reset_index(drop=True)


def _load_metrics(path: Path) -> pd.DataFrame:
    _assert_exists(path, "ETF dual momentum cost metrics")
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
        raise ValueError(f"ETF dual momentum cost metrics are missing columns: {missing}")

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


def _ranking_records(ranking: pd.DataFrame, lookback_months: int) -> list[RankingRecord]:
    momentum_column: str = f"momentum_{lookback_months}m"
    return [
        {
            "ticker": str(row["ticker"]),
            "rank": int(row["rank"]),
            "momentum": float(row[momentum_column]),
            "selected": _to_bool(row["selected"], "selected"),
            "target_weight": float(row["target_weight"]),
            "role": str(row["role"]),
        }
        for _, row in ranking.iterrows()
    ]


def _rotation_records(signals: pd.DataFrame, count: int) -> list[RotationRecord]:
    if count <= 0:
        raise ValueError(f"count must be positive. Received: {count}.")

    recent: pd.DataFrame = signals.tail(count).iloc[::-1].reset_index(drop=True)
    return [
        {
            "signal_date": str(row["signal_date"]),
            "selected_ticker": str(row["selected_ticker"]),
            "selected_score": float(row["selected_score"]),
            "defensive_signal": _to_bool(row["defensive_signal"], "defensive_signal"),
        }
        for _, row in recent.iterrows()
    ]


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


def _dashboard_payload(
    signals: pd.DataFrame,
    ranking: pd.DataFrame,
    metrics: pd.DataFrame,
    equity: pd.Series,
    cost_bps: int,
    chart_points: int,
    lookback_months: int,
    strategy_name: str,
) -> EtfDashboardPayload:
    latest_signal: pd.Series = signals.iloc[-1]
    drawdown: pd.Series = compute_drawdown(equity)
    return {
        "strategy_name": strategy_name,
        "signal_date": str(latest_signal["signal_date"]),
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        "lookback_months": lookback_months,
        "selected_ticker": str(latest_signal["selected_ticker"]),
        "selected_score": float(latest_signal["selected_score"]),
        "defensive_signal": _to_bool(latest_signal["defensive_signal"], "defensive_signal"),
        "base_cost_bps": cost_bps,
        "base_metric": _base_metric(metrics, cost_bps),
        "metrics": _metric_records(metrics),
        "ranking": _ranking_records(ranking, lookback_months),
        "rotations": _rotation_records(signals, 12),
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
  <title>ETF 双动量轮动看板</title>
  <style>
    :root {
      --ink: #18201d;
      --muted: #5f6a64;
      --paper: #f3f5ee;
      --panel: #ffffff;
      --line: #d8ddd5;
      --green: #0b7464;
      --green-soft: #dcefe9;
      --blue: #284f9f;
      --blue-soft: #e1e8fb;
      --red: #b83e4f;
      --red-soft: #f4dce0;
      --gold: #a5691f;
      --shadow: 0 16px 42px rgba(24, 32, 29, 0.08);
    }

    * { box-sizing: border-box; letter-spacing: 0; }

    body {
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(11, 116, 100, 0.09) 1px, transparent 1px),
        linear-gradient(180deg, rgba(40, 79, 159, 0.07) 1px, transparent 1px),
        var(--paper);
      background-size: 30px 30px;
      font-family: "Avenir Next", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    }

    .shell {
      width: min(1160px, calc(100% - 32px));
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
      max-width: 760px;
      font-family: "Baskerville", "Songti SC", "STSong", serif;
      font-size: 2.5rem;
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

    .allocation {
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 18px;
      align-items: center;
      padding: 22px;
    }

    .allocation-code {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 150px;
      height: 150px;
      border: 2px solid var(--ink);
      border-radius: 50%;
      background: var(--green-soft);
      color: var(--green);
      font-size: 2.2rem;
      font-weight: 900;
    }

    .allocation h2 {
      margin: 0 0 10px;
      font-size: 1.7rem;
    }

    .allocation p {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 28px;
      margin-bottom: 10px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 800;
      color: var(--green);
      background: var(--green-soft);
    }

    .badge.defensive {
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
      min-width: 680px;
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
      .rules,
      .allocation {
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
        width: min(100% - 20px, 1160px);
        padding-top: 16px;
      }

      .kpi-grid {
        grid-template-columns: 1fr;
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
        <p class="eyebrow">ETF 双动量 · 月度资产轮动</p>
        <h1>在股票、债券、黄金和防守资产之间选择最强趋势</h1>
      </div>
      <div class="stamp">
        <span>信号日期</span>
        <strong id="signalDate">--</strong>
        <span id="generatedAt">--</span>
      </div>
    </header>

    <section class="notice">
      这套策略每月比较 SPY、QQQ、IWM、TLT、GLD 的 6 个月收益，选择最强资产；如果最强资产的 6 个月收益不为正，则切到 SHY。它用于研究资产轮动，不是投资建议。
    </section>

    <section class="kpi-grid" aria-label="核心指标">
      <div class="kpi">
        <span>当前目标资产</span>
        <strong id="selectedTicker">--</strong>
        <small id="selectedType">--</small>
      </div>
      <div class="kpi">
        <span>6 个月动量</span>
        <strong id="selectedScore">--</strong>
        <small>当前被选资产对应的趋势分数</small>
      </div>
      <div class="kpi">
        <span id="sharpeLabel">回测 Sharpe</span>
        <strong id="sharpeValue">--</strong>
        <small>交易成本已计入指定基准情景</small>
      </div>
      <div class="kpi">
        <span>历史最大回撤</span>
        <strong id="drawdownValue">--</strong>
        <small>用于衡量策略曾经承受的最大下跌</small>
      </div>
    </section>

    <section class="grid-two">
      <article class="panel">
        <div class="allocation">
          <div class="allocation-code" id="allocationCode">--</div>
          <div>
            <div class="badge" id="allocationBadge">趋势持有</div>
            <h2 id="allocationTitle">--</h2>
            <p id="allocationText">--</p>
          </div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-header">
          <h2 class="panel-title">最新资产排名</h2>
          <p class="panel-subtitle">按 6 个月收益排序。只有最高资产为正收益时才持有风险资产，否则进入防守。</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th class="numeric">排名</th>
                <th>资产</th>
                <th class="numeric">6 个月动量</th>
                <th class="numeric">目标仓位</th>
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
          <h2 class="panel-title">最近 12 次月度选择</h2>
          <p class="panel-subtitle">用来观察策略是否频繁切换，以及是否多次进入防守状态。</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>信号日期</th>
                <th>选择资产</th>
                <th class="numeric">分数</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody id="rotationTable"></tbody>
          </table>
        </div>
      </article>

      <article class="panel">
        <div class="panel-header">
          <h2 class="panel-title">交易成本压力测试</h2>
          <p class="panel-subtitle">同一套 ETF 双动量规则，在不同交易成本假设下的表现。</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th class="numeric">成本</th>
                <th class="numeric">累计收益</th>
                <th class="numeric">年化收益</th>
                <th class="numeric">Sharpe</th>
                <th class="numeric">最大回撤</th>
              </tr>
            </thead>
            <tbody id="costTable"></tbody>
          </table>
        </div>
      </article>
    </section>

    <section class="rules">
      <div class="rule">
        <strong>1. 先选相对最强</strong>
        <p>每月只在几个高流动性 ETF 之间比较，不做单只股票判断。</p>
      </div>
      <div class="rule">
        <strong>2. 再看绝对趋势</strong>
        <p>如果最强资产自己也没有正收益，说明风险资产趋势不足，切到 SHY。</p>
      </div>
      <div class="rule">
        <strong>3. 每月调一次</strong>
        <p>这套策略不适合日内频繁交易，重点是中期资产轮动。</p>
      </div>
    </section>
  </main>

  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
    const pctFormatter = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 1 });
    const pct2Formatter = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 2 });
    const num = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 });

    function setText(id, value) {
      document.getElementById(id).textContent = value;
    }

    function pct(value) {
      return pctFormatter.format(value);
    }

    function pct2(value) {
      return pct2Formatter.format(value);
    }

    function renderRanking() {
      document.getElementById("rankingTable").innerHTML = DATA.ranking.map((row) => `
        <tr>
          <td class="numeric">${row.rank}</td>
          <td><strong>${row.ticker}</strong>${row.selected ? " · 当前选择" : ""}</td>
          <td class="numeric">${pct(row.momentum)}</td>
          <td class="numeric">${pct(row.target_weight)}</td>
        </tr>
      `).join("");
    }

    function renderRotations() {
      document.getElementById("rotationTable").innerHTML = DATA.rotations.map((row) => `
        <tr>
          <td>${row.signal_date}</td>
          <td><strong>${row.selected_ticker}</strong></td>
          <td class="numeric">${pct(row.selected_score)}</td>
          <td>${row.defensive_signal ? "防守" : "趋势持有"}</td>
        </tr>
      `).join("");
    }

    function renderCostTable() {
      document.getElementById("costTable").innerHTML = DATA.metrics.map((row) => `
        <tr>
          <td class="numeric">${row.cost_bps} bps</td>
          <td class="numeric">${pct(row.total_return)}</td>
          <td class="numeric">${pct(row.CAGR)}</td>
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
          <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d8ddd5" />
          <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d8ddd5" />
          <polygon points="${area}" fill="${fillColor}" opacity="0.72"></polygon>
          <polyline points="${line}" fill="none" stroke="${color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>
          <text x="${pad}" y="${height - 8}" fill="#5f6a64" font-size="13">${first.date}</text>
          <text x="${width - pad}" y="${height - 8}" fill="#5f6a64" font-size="13" text-anchor="end">${last.date}</text>
          <text x="${pad}" y="20" fill="#5f6a64" font-size="13">${formatter(max.value)}</text>
          <text x="${width - pad}" y="20" fill="#5f6a64" font-size="13" text-anchor="end">${formatter(min.value)}</text>
        </svg>
      `;
    }

    function render() {
      const defensiveText = DATA.defensive_signal ? "防守仓位" : "趋势持有";
      setText("signalDate", DATA.signal_date);
      setText("generatedAt", `生成时间 ${DATA.generated_at}`);
      setText("selectedTicker", DATA.selected_ticker);
      setText("selectedType", defensiveText);
      setText("selectedScore", pct(DATA.selected_score));
      setText("sharpeLabel", `${DATA.base_cost_bps} bps 成本 Sharpe`);
      setText("sharpeValue", num.format(DATA.base_metric.Sharpe));
      setText("drawdownValue", pct(DATA.base_metric.max_drawdown));
      setText("allocationCode", DATA.selected_ticker);
      setText("allocationTitle", `本月目标：100% ${DATA.selected_ticker}`);
      setText("allocationText", DATA.defensive_signal
        ? "风险资产的 6 个月动量没有转正，因此策略进入防守资产。"
        : `当前最强资产的 ${DATA.lookback_months} 个月动量为 ${pct(DATA.selected_score)}，策略选择继续持有趋势资产。`
      );
      document.getElementById("allocationBadge").textContent = defensiveText;
      document.getElementById("allocationBadge").className = DATA.defensive_signal ? "badge defensive" : "badge";
      setText("equitySubtitle", `按 ${DATA.base_cost_bps} bps 交易成本计算，累计收益 ${pct2(DATA.base_metric.total_return)}。`);
      renderRanking();
      renderRotations();
      renderCostTable();
      renderLineChart("equityChart", DATA.equity_points, "#0b7464", "#dcefe9", (value) => num.format(value));
      renderLineChart("drawdownChart", DATA.drawdown_points, "#b83e4f", "#f4dce0", (value) => pct(value));
    }

    render();
  </script>
</body>
</html>
"""


def render_html(payload: EtfDashboardPayload) -> str:
    payload_json: str = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return _html_template().replace("__DASHBOARD_DATA__", payload_json)


def save_site(html: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def build_etf_dual_momentum_site(
    results_dir: Path,
    site_dir: Path,
    signals_file: str,
    ranking_file: str,
    metrics_file: str,
    equity_file: str,
    returns_file: str,
    index_file: str,
    cost_bps: int,
    chart_points: int,
    lookback_months: int,
    strategy_name: str,
) -> Path:
    signals_path: Path = results_dir / signals_file
    ranking_path: Path = results_dir / ranking_file
    metrics_path: Path = results_dir / metrics_file
    equity_path: Path = results_dir / f"cost_{cost_bps}bps" / equity_file
    returns_path: Path = results_dir / f"cost_{cost_bps}bps" / returns_file

    signals: pd.DataFrame = _load_signals(signals_path)
    ranking: pd.DataFrame = _load_ranking(ranking_path, lookback_months)
    metrics: pd.DataFrame = _load_metrics(metrics_path)
    equity: pd.Series = _load_series(equity_path, "equity", "ETF dual momentum equity curve")
    _load_series(returns_path, "strategy_return", "ETF dual momentum strategy returns")

    payload: EtfDashboardPayload = _dashboard_payload(
        signals,
        ranking,
        metrics,
        equity,
        cost_bps,
        chart_points,
        lookback_months,
        strategy_name,
    )
    out_path: Path = site_dir / index_file
    save_site(render_html(payload), out_path)
    return out_path


def main() -> None:
    out_path: Path = build_etf_dual_momentum_site(
        ETF_DUAL_MOMENTUM_RESULTS_DIR,
        ETF_DUAL_MOMENTUM_SITE_DIR,
        ETF_DUAL_MOMENTUM_SIGNALS_FILE,
        ETF_DUAL_MOMENTUM_RANKING_FILE,
        COST_GRID_METRICS_FILE,
        EQUITY_CURVE_FILE,
        STRATEGY_RETURNS_FILE,
        SITE_INDEX_FILE,
        SITE_BASE_COST_BPS,
        SITE_CHART_POINTS,
        ETF_DUAL_MOMENTUM_LOOKBACK_MONTHS,
        ETF_DUAL_MOMENTUM_NAME,
    )
    print(f"Saved ETF dual momentum website: {out_path}")


if __name__ == "__main__":
    main()
