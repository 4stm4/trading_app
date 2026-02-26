import { useEffect, useMemo, useRef, useState } from 'react';
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
} from 'lightweight-charts';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
const DEFAULT_MODEL = 'balanced';
const DEFAULT_DEPOSIT = 100000;
const DEFAULT_COMMISSION_BPS = 4;
const DEFAULT_SLIPPAGE_BPS = 6;
const DEFAULT_PATTERN_MIN_SAMPLE = 30;
const DEFAULT_MARKET_LIMIT = 300;
const CANDIDATE_SCAN_LIMIT = 24;
const CANDIDATE_SCAN_CONCURRENCY = 4;
const DEFAULT_BACKTEST_LIMIT = 1200;
const DEFAULT_BACKTEST_LOOKBACK = 300;
const DEFAULT_BACKTEST_MAX_HOLDING = 50;
const DEFAULT_ROBUSTNESS_LIMIT = 1500;
const DEFAULT_MONTE_CARLO_SIMULATIONS = 300;
const DEFAULT_TRAIN_RATIO = 0.7;
const MARKET_OPTIONS = [
  {
    key: 'shares',
    label: 'MOEX Акции',
    engine: 'stock',
    market: 'shares',
    board: 'TQBR',
  },
  {
    key: 'futures',
    label: 'MOEX Фьючерсы',
    engine: 'futures',
    market: 'forts',
    board: 'RFUD',
  },
];
const TIMEFRAME_OPTIONS = ['1m', '10m', '1h', '1d'];
const SCREEN_OPTIONS = [
  { key: 'portfolio', label: 'Portfolio & Signals' },
  { key: 'analysis', label: 'Technical Analysis' },
  { key: 'testing', label: 'System Testing' },
  { key: 'system', label: 'System' },
];
const TESTING_TAB_OPTIONS = [
  { key: 'backtest', label: 'Backtest' },
  { key: 'robustness', label: 'Robustness' },
];
const MODEL_OPTIONS = ['balanced', 'aggressive', 'scalp', 'high_rr', 'conservative'];
const SCANNER_TIMEFRAMES = Object.freeze([...TIMEFRAME_OPTIONS]);
const SCANNER_MODELS = Object.freeze([...MODEL_OPTIONS]);
const AUTH_TOKEN_STORAGE_KEY = 'trading.authToken';
const USER_EMAIL_STORAGE_KEY = 'trading.activeUserEmail';
const USER_EMAILS_STORAGE_KEY = 'trading.userEmails';
const CONFIDENCE_RANK = {
  high: 3,
  medium: 2,
  low: 1,
  none: 0,
};
const SYSTEM_DEFAULTS = Object.freeze({
  deposit: DEFAULT_DEPOSIT,
  commissionBps: DEFAULT_COMMISSION_BPS,
  slippageBps: DEFAULT_SLIPPAGE_BPS,
  patternMinSample: DEFAULT_PATTERN_MIN_SAMPLE,
  marketLimit: DEFAULT_MARKET_LIMIT,
  candidateScanLimit: CANDIDATE_SCAN_LIMIT,
  candidateScanConcurrency: CANDIDATE_SCAN_CONCURRENCY,
  backtestLimit: DEFAULT_BACKTEST_LIMIT,
  backtestLookbackWindow: DEFAULT_BACKTEST_LOOKBACK,
  backtestMaxHoldingCandles: DEFAULT_BACKTEST_MAX_HOLDING,
  robustnessLimit: DEFAULT_ROBUSTNESS_LIMIT,
  monteCarloSimulations: DEFAULT_MONTE_CARLO_SIMULATIONS,
  trainRatio: DEFAULT_TRAIN_RATIO,
});

function normalizeSystemConfig(rawConfig) {
  const raw = rawConfig ?? {};
  const pickNumber = (key, fallback) => {
    const value = Number(raw[key]);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    deposit: Math.max(0, pickNumber("deposit", SYSTEM_DEFAULTS.deposit)),
    commissionBps: Math.max(0, pickNumber("commissionBps", SYSTEM_DEFAULTS.commissionBps)),
    slippageBps: Math.max(0, pickNumber("slippageBps", SYSTEM_DEFAULTS.slippageBps)),
    patternMinSample: Math.max(1, Math.floor(pickNumber("patternMinSample", SYSTEM_DEFAULTS.patternMinSample))),
    marketLimit: Math.max(1, Math.floor(pickNumber("marketLimit", SYSTEM_DEFAULTS.marketLimit))),
    candidateScanLimit: Math.max(1, Math.floor(pickNumber("candidateScanLimit", SYSTEM_DEFAULTS.candidateScanLimit))),
    candidateScanConcurrency: Math.max(
      1,
      Math.floor(pickNumber("candidateScanConcurrency", SYSTEM_DEFAULTS.candidateScanConcurrency)),
    ),
    backtestLimit: Math.max(1, Math.floor(pickNumber("backtestLimit", SYSTEM_DEFAULTS.backtestLimit))),
    backtestLookbackWindow: Math.max(
      1,
      Math.floor(pickNumber("backtestLookbackWindow", SYSTEM_DEFAULTS.backtestLookbackWindow)),
    ),
    backtestMaxHoldingCandles: Math.max(
      1,
      Math.floor(pickNumber("backtestMaxHoldingCandles", SYSTEM_DEFAULTS.backtestMaxHoldingCandles)),
    ),
    robustnessLimit: Math.max(1, Math.floor(pickNumber("robustnessLimit", SYSTEM_DEFAULTS.robustnessLimit))),
    monteCarloSimulations: Math.max(
      1,
      Math.floor(pickNumber("monteCarloSimulations", SYSTEM_DEFAULTS.monteCarloSimulations)),
    ),
    trainRatio: clamp(pickNumber("trainRatio", SYSTEM_DEFAULTS.trainRatio), 0.1, 0.95),
  };
}

function normalizeUserEmail(rawValue) {
  const normalized = String(rawValue ?? '').trim().toLowerCase();
  return normalized;
}

function readStoredUserEmails() {
  if (typeof window === 'undefined') {
    return [];
  }
  const fallback = [];
  try {
    const raw = window.localStorage.getItem(USER_EMAILS_STORAGE_KEY);
    if (!raw) {
      return fallback;
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return fallback;
    }
    const normalized = Array.from(
      new Set(parsed.map((value) => normalizeUserEmail(value)).filter(Boolean)),
    );
    return normalized.length > 0 ? normalized : fallback;
  } catch (_error) {
    return fallback;
  }
}

function buildMoexInstrumentsUrl(marketConfig) {
  const params = new URLSearchParams({
    limit: '80',
    engine: marketConfig.engine,
    market: marketConfig.market,
    board: marketConfig.board,
  });
  return `${API_BASE_URL}/api/moex/instruments?${params.toString()}`;
}

function buildDashboardMarketUrl({
  ticker,
  exchange,
  timeframe,
  engine,
  market,
  board,
  model,
  deposit,
  limit,
  commissionBps,
  slippageBps,
  patternMinSample,
}) {
  const params = new URLSearchParams({
    ticker,
    exchange,
    timeframe,
    engine,
    market,
    board,
    model,
    deposit: String(deposit),
    limit: String(limit ?? 300),
    commission_bps: String(commissionBps ?? DEFAULT_COMMISSION_BPS),
    slippage_bps: String(slippageBps ?? DEFAULT_SLIPPAGE_BPS),
    pattern_min_sample: String(patternMinSample ?? DEFAULT_PATTERN_MIN_SAMPLE),
  });
  return `${API_BASE_URL}/api/dashboard/market?${params.toString()}`;
}

function buildDashboardBacktestUrl() {
  return `${API_BASE_URL}/api/dashboard/backtest`;
}

function buildDashboardRobustnessUrl() {
  return `${API_BASE_URL}/api/dashboard/robustness`;
}

function buildAuthRegisterUrl() {
  return `${API_BASE_URL}/api/auth/register`;
}

function buildAuthLoginUrl() {
  return `${API_BASE_URL}/api/auth/login`;
}

function buildAuthMeUrl() {
  return `${API_BASE_URL}/api/auth/me`;
}

function buildSystemsUrl(userEmail) {
  const normalizedEmail = String(userEmail ?? '').trim();
  if (!normalizedEmail) {
    return `${API_BASE_URL}/api/systems`;
  }
  const params = new URLSearchParams({ user_email: normalizedEmail });
  return `${API_BASE_URL}/api/systems?${params.toString()}`;
}

function buildSetCurrentSystemUrl() {
  return `${API_BASE_URL}/api/systems/current`;
}

function buildUpdateSystemConfigUrl(systemId) {
  return `${API_BASE_URL}/api/systems/${systemId}/config`;
}

function buildPortfolioUrl(userEmail) {
  const normalizedEmail = String(userEmail ?? '').trim();
  if (!normalizedEmail) {
    return `${API_BASE_URL}/api/portfolio`;
  }
  const params = new URLSearchParams({ user_email: normalizedEmail });
  return `${API_BASE_URL}/api/portfolio?${params.toString()}`;
}

function buildScansUrl(userEmail, options = {}) {
  const normalizedEmail = String(userEmail ?? '').trim();
  const params = new URLSearchParams();
  if (normalizedEmail) {
    params.set('user_email', normalizedEmail);
  }
  const systemId = Number(options.systemId);
  if (Number.isInteger(systemId) && systemId > 0) {
    params.set('system_id', String(systemId));
  }
  const scanKey = String(options.scanKey ?? '').trim();
  if (scanKey) {
    params.set('scan_key', scanKey);
  }
  const limit = Number(options.limit);
  if (Number.isInteger(limit) && limit > 0) {
    params.set('limit', String(limit));
  }
  const tradableOnly = options.tradableOnly === true;
  if (tradableOnly) {
    params.set('tradable_only', 'true');
  }
  const query = params.toString();
  return query ? `${API_BASE_URL}/api/scans?${query}` : `${API_BASE_URL}/api/scans`;
}

function buildCreateScansUrl() {
  return `${API_BASE_URL}/api/scans`;
}

function buildAuthHeaders(authToken, headers = {}) {
  const nextHeaders = { ...headers };
  const token = String(authToken ?? '').trim();
  if (token) {
    nextHeaders.Authorization = `Bearer ${token}`;
  }
  return nextHeaders;
}

async function mapWithConcurrency(items, limit, mapper, onProgress) {
  const total = Array.isArray(items) ? items.length : 0;
  if (total === 0) {
    return [];
  }
  const safeLimit = Math.max(1, Math.min(Number(limit) || 1, total));
  const results = new Array(total);
  let cursor = 0;
  let completed = 0;

  async function worker() {
    while (true) {
      const currentIndex = cursor;
      cursor += 1;
      if (currentIndex >= total) {
        break;
      }
      try {
        results[currentIndex] = await mapper(items[currentIndex], currentIndex);
      } catch (_error) {
        results[currentIndex] = null;
      } finally {
        completed += 1;
        if (typeof onProgress === 'function') {
          onProgress(completed, total);
        }
      }
    }
  }

  await Promise.all(Array.from({ length: safeLimit }, () => worker()));
  return results;
}

function formatPrice(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return value.toLocaleString('ru-RU', { maximumFractionDigits: 4 });
}

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return `${value.toFixed(2)}%`;
}

function formatSignedPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
}

function formatNumber(value, digits = 2) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(digits);
}

function formatDateTime(value) {
  const normalized = String(value ?? '').trim();
  if (!normalized) {
    return '—';
  }
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) {
    return normalized;
  }
  return parsed.toLocaleString('ru-RU', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function toOptionalNumber(value) {
  if (value === null || value === undefined || value === '') {
    return NaN;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : NaN;
}

function normalizeLineSeries(data) {
  if (!Array.isArray(data) || data.length === 0) {
    return [];
  }

  const prepared = data
    .map((point) => ({
      time: Number(point.time),
      value: Number(point.value),
    }))
    .filter((point) => isFiniteNumber(point.time) && isFiniteNumber(point.value))
    .sort((left, right) => left.time - right.time);

  if (prepared.length <= 1) {
    return prepared;
  }

  const normalized = [prepared[0]];
  for (let index = 1; index < prepared.length; index += 1) {
    const current = prepared[index];
    const previous = normalized[normalized.length - 1];
    if (current.time === previous.time) {
      // Keep latest value for the same candle timestamp to satisfy chart constraints.
      normalized[normalized.length - 1] = current;
      continue;
    }
    normalized.push(current);
  }
  return normalized;
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function buildTrendLine(candles, windowSize = 120) {
  if (!Array.isArray(candles) || candles.length < 2) {
    return { data: [], direction: 'flat', slopePercent: 0 };
  }

  const sample = candles.slice(-Math.min(windowSize, candles.length));
  if (sample.length < 2) {
    return { data: [], direction: 'flat', slopePercent: 0 };
  }

  const closes = sample.map((candle) => Number(candle.close));
  if (closes.some((value) => !isFiniteNumber(value))) {
    return { data: [], direction: 'flat', slopePercent: 0 };
  }

  const n = sample.length;
  const meanX = (n - 1) / 2;
  const meanY = closes.reduce((sum, value) => sum + value, 0) / n;

  let numerator = 0;
  let denominator = 0;
  for (let index = 0; index < n; index += 1) {
    const x = index - meanX;
    const y = closes[index] - meanY;
    numerator += x * y;
    denominator += x * x;
  }
  const slope = denominator === 0 ? 0 : numerator / denominator;
  const intercept = meanY - slope * meanX;

  const first = sample[0];
  const last = sample[n - 1];
  const firstValue = intercept;
  const lastValue = intercept + slope * (n - 1);
  const base = Math.abs(firstValue) > 1e-9 ? Math.abs(firstValue) : 1.0;
  const slopePercent = ((lastValue - firstValue) / base) * 100;
  const direction = Math.abs(slopePercent) < 0.2 ? 'flat' : slopePercent > 0 ? 'up' : 'down';

  return {
    data: [
      { time: first.time, value: firstValue },
      { time: last.time, value: lastValue },
    ],
    direction,
    slopePercent,
  };
}

function buildRegimeBands(candles) {
  if (!Array.isArray(candles) || candles.length === 0) {
    return [];
  }
  const regimeColor = {
    trend: 'rgba(38, 184, 125, 0.10)',
    range: 'rgba(66, 103, 178, 0.08)',
    high_volatility: 'rgba(212, 134, 28, 0.10)',
    unknown: 'rgba(108, 122, 138, 0.08)',
  };
  return candles.map((candle) => {
    const regime = String(candle.regime ?? 'unknown');
    return {
      time: candle.time,
      value: 1,
      color: regimeColor[regime] ?? regimeColor.unknown,
    };
  });
}

function validateTradePlan(signal) {
  const signalType = String(signal?.signal ?? 'none').toLowerCase();
  const entry = Number(signal?.entry);
  const stop = Number(signal?.stop);
  const target = Number(signal?.target);
  const rr = Number(signal?.rr);
  const issues = [];

  if (signalType !== 'long' && signalType !== 'short') {
    return {
      status: 'no_signal',
      tradable: false,
      signal: signalType,
      issues: ['No actionable signal'],
      entry,
      stop,
      target,
      rr,
      confidence: String(signal?.confidence ?? 'none'),
    };
  }

  if (!isFiniteNumber(entry) || entry <= 0) {
    issues.push('entry_missing_or_non_positive');
  }
  if (!isFiniteNumber(stop) || stop <= 0) {
    issues.push('stop_missing_or_non_positive');
  }
  if (!isFiniteNumber(target) || target <= 0) {
    issues.push('target_missing_or_non_positive');
  }
  if (issues.length === 0 && signalType === 'long' && !(stop < entry && entry < target)) {
    issues.push('invalid_price_order_for_long');
  }
  if (issues.length === 0 && signalType === 'short' && !(target < entry && entry < stop)) {
    issues.push('invalid_price_order_for_short');
  }
  if (!isFiniteNumber(rr) || rr <= 0) {
    issues.push('rr_non_positive');
  }

  return {
    status: issues.length === 0 ? 'valid' : 'invalid',
    tradable: issues.length === 0,
    signal: signalType,
    issues,
    entry,
    stop,
    target,
    rr,
    confidence: String(signal?.confidence ?? 'none'),
  };
}

function buildPatternInsights(candles, options = {}) {
  const horizon = Number(options.horizon ?? 8);
  const breakoutLookback = Number(options.breakoutLookback ?? 20);
  const roundTripCostPercent = Number(options.roundTripCostPercent ?? 0);
  const minSample = Number(options.minSample ?? DEFAULT_PATTERN_MIN_SAMPLE);
  if (!Array.isArray(candles) || candles.length < 3) {
    return { events: [], stats: [] };
  }

  const events = [];
  for (let index = 1; index < candles.length; index += 1) {
    const current = candles[index];
    const previous = candles[index - 1];
    if (!isFiniteNumber(current.open) || !isFiniteNumber(current.high) || !isFiniteNumber(current.low) || !isFiniteNumber(current.close)) {
      continue;
    }
    if (!isFiniteNumber(previous.open) || !isFiniteNumber(previous.high) || !isFiniteNumber(previous.low) || !isFiniteNumber(previous.close)) {
      continue;
    }

    const range = Math.max(current.high - current.low, 1e-9);
    const body = Math.abs(current.close - current.open);
    const upperWick = current.high - Math.max(current.open, current.close);
    const lowerWick = Math.min(current.open, current.close) - current.low;
    const trendBias =
      isFiniteNumber(current.ma50) && isFiniteNumber(current.ma200)
        ? Math.sign(current.ma50 - current.ma200)
        : 0;
    const regime = String(current.regime ?? 'range');
    const futureIndex = Math.min(index + horizon, candles.length - 1);
    const futureClose = candles[futureIndex]?.close;
    const hasForwardWindow = index + horizon < candles.length && isFiniteNumber(futureClose);
    const rawReturnPct = hasForwardWindow ? ((futureClose - current.close) / current.close) * 100 : null;

    const candidates = [];
    if (
      previous.close < previous.open &&
      current.close > current.open &&
      current.open <= previous.close &&
      current.close >= previous.open
    ) {
      candidates.push({ key: 'bull_engulfing', label: 'Bull Engulf', code: 'BE', direction: 'long' });
    }
    if (
      previous.close > previous.open &&
      current.close < current.open &&
      current.open >= previous.close &&
      current.close <= previous.open
    ) {
      candidates.push({ key: 'bear_engulfing', label: 'Bear Engulf', code: 'SE', direction: 'short' });
    }
    if (body / range < 0.35 && lowerWick > body * 2.2 && upperWick < body * 1.4) {
      candidates.push({ key: 'bull_pinbar', label: 'Bull Pin', code: 'BP', direction: 'long' });
    }
    if (body / range < 0.35 && upperWick > body * 2.2 && lowerWick < body * 1.4) {
      candidates.push({ key: 'bear_pinbar', label: 'Bear Pin', code: 'SP', direction: 'short' });
    }
    if (current.high < previous.high && current.low > previous.low) {
      candidates.push({ key: 'inside_bar', label: 'Inside Bar', code: 'IB', direction: 'none' });
    }

    const start = Math.max(0, index - breakoutLookback);
    const lookbackSlice = candles.slice(start, index);
    const maxHigh = lookbackSlice.reduce((maxValue, candle) => Math.max(maxValue, candle.high), Number.NEGATIVE_INFINITY);
    const minLow = lookbackSlice.reduce((minValue, candle) => Math.min(minValue, candle.low), Number.POSITIVE_INFINITY);
    if (isFiniteNumber(maxHigh) && current.close > maxHigh) {
      candidates.push({ key: 'breakout_up', label: 'Breakout Up', code: 'BO', direction: 'long' });
    }
    if (isFiniteNumber(minLow) && current.close < minLow) {
      candidates.push({ key: 'breakout_down', label: 'Breakout Down', code: 'BD', direction: 'short' });
    }

    for (const pattern of candidates.slice(0, 2)) {
      const signedReturnPct =
        hasForwardWindow && rawReturnPct !== null
          ? pattern.direction === 'short'
            ? -rawReturnPct
            : rawReturnPct
          : null;
      const trendAligned =
        pattern.direction === 'none' ||
        (pattern.direction === 'long' && trendBias >= 0) ||
        (pattern.direction === 'short' && trendBias <= 0);
      const regimeBonus = regime === 'trend' ? 6 : regime === 'high_volatility' ? -4 : 0;
      const baseConfidence =
        52 +
        (signedReturnPct !== null ? signedReturnPct * 6 : 0) +
        (trendAligned ? 8 : -6) +
        (current.isImpulse ? 5 : 0) +
        regimeBonus;
      const confidence = clamp(Math.round(baseConfidence), 5, 99);

      events.push({
        time: current.time,
        pattern: pattern.key,
        label: pattern.label,
        code: pattern.code,
        direction: pattern.direction,
        confidence,
        success: signedReturnPct !== null ? signedReturnPct > 0 : null,
        signedReturnPct,
      });
    }
  }

  const statsMap = new Map();
  for (const event of events) {
    if (!statsMap.has(event.pattern)) {
      statsMap.set(event.pattern, {
        pattern: event.pattern,
        label: event.label,
        count: 0,
        evaluated: 0,
        wins: 0,
        confidenceSum: 0,
        returnSum: 0,
      });
    }
    const bucket = statsMap.get(event.pattern);
    bucket.count += 1;
    bucket.confidenceSum += event.confidence;
    if (event.success !== null && event.signedReturnPct !== null && event.direction !== 'none') {
      bucket.evaluated += 1;
      if (event.success) {
        bucket.wins += 1;
      }
      bucket.returnSum += event.signedReturnPct;
    }
  }

  const stats = Array.from(statsMap.values())
    .map((bucket) => ({
      pattern: bucket.pattern,
      label: bucket.label,
      count: bucket.count,
      winrate: bucket.evaluated > 0 ? (bucket.wins / bucket.evaluated) * 100 : null,
      avgReturnPct: bucket.evaluated > 0 ? bucket.returnSum / bucket.evaluated : null,
      avgReturnNetPct:
        bucket.evaluated > 0 ? bucket.returnSum / bucket.evaluated - roundTripCostPercent : null,
      avgConfidence: bucket.count > 0 ? bucket.confidenceSum / bucket.count : null,
      minSample,
      sampleQualified: bucket.count >= minSample,
      edgeQualified:
        bucket.count >= minSample &&
        bucket.evaluated > 0 &&
        bucket.returnSum / bucket.evaluated - roundTripCostPercent > 0,
    }))
    .sort((left, right) => right.count - left.count);

  return {
    events: events.slice(-120),
    stats: stats.slice(0, 8),
  };
}

function parseInstruments(payload) {
  if (!payload || !Array.isArray(payload.instruments)) {
    return [];
  }
  return payload.instruments
    .map((item) => {
      const symbol = String(item.symbol ?? '')
        .trim()
        .toUpperCase();
      if (!symbol) {
        return null;
      }
      const prevPrice = item.prevPrice === null || item.prevPrice === undefined ? NaN : Number(item.prevPrice);
      const currency = String(item.currency ?? '').trim() || 'RUB';
      return {
        symbol,
        name: String(item.shortName ?? item.name ?? symbol).trim() || symbol,
        price: formatPrice(prevPrice),
        change: currency,
      };
    })
    .filter(Boolean);
}

function parseMarketResponse(payload) {
  if (!payload || !Array.isArray(payload.candles)) {
    return null;
  }

  const assumptions = payload.execution_assumptions ?? {};
  const roundTripCostPercent = toOptionalNumber(assumptions.round_trip_cost_percent);
  const patternMinSample = Number(assumptions.pattern_min_sample);

  const candles = [];
  const volume = [];
  const ma50 = [];
  const ma200 = [];
  const rsi = [];
  const atr = [];
  const volumeRatio = [];
  for (const item of payload.candles) {
    const time = Number(item.time);
    const open = Number(item.open);
    const high = Number(item.high);
    const low = Number(item.low);
    const close = Number(item.close);
    const vol = Number(item.volume);
    if (
      !isFiniteNumber(time) ||
      !isFiniteNumber(open) ||
      !isFiniteNumber(high) ||
      !isFiniteNumber(low) ||
      !isFiniteNumber(close) ||
      !isFiniteNumber(vol)
    ) {
      continue;
    }
    const ma50Value = toOptionalNumber(item.ma50);
    const ma200Value = toOptionalNumber(item.ma200);
    const rsiValue = toOptionalNumber(item.rsi);
    const atrValue = toOptionalNumber(item.atr);
    const volumeRatioValue = toOptionalNumber(item.volume_ratio);
    const regime = String(item.regime ?? 'unknown');
    const isImpulse = Boolean(item.is_impulse);
    candles.push({
      time,
      open,
      high,
      low,
      close,
      regime,
      isImpulse,
      ma50: ma50Value,
      ma200: ma200Value,
    });
    volume.push({
      time,
      value: vol,
      color: close >= open ? 'rgba(38, 184, 125, 0.50)' : 'rgba(209, 79, 101, 0.50)',
    });
    if (isFiniteNumber(ma50Value)) {
      ma50.push({ time, value: ma50Value });
    }
    if (isFiniteNumber(ma200Value)) {
      ma200.push({ time, value: ma200Value });
    }
    if (isFiniteNumber(rsiValue)) {
      rsi.push({ time, value: rsiValue });
    }
    if (isFiniteNumber(atrValue)) {
      atr.push({ time, value: atrValue });
    }
    if (isFiniteNumber(volumeRatioValue)) {
      volumeRatio.push({ time, value: volumeRatioValue });
    }
  }

  const trendLine = buildTrendLine(candles);
  const regimeBands = buildRegimeBands(candles);
  const patternInsights = buildPatternInsights(candles, {
    roundTripCostPercent: isFiniteNumber(roundTripCostPercent) ? roundTripCostPercent : 0,
    minSample: Number.isFinite(patternMinSample) && patternMinSample > 0 ? patternMinSample : DEFAULT_PATTERN_MIN_SAMPLE,
  });
  const signal = payload.signal ?? null;
  const tradePlan = payload.trade_plan ?? validateTradePlan(signal);

  return {
    candles,
    volume,
    ma50,
    ma200,
    rsi,
    atr,
    volumeRatio,
    trendLine,
    regimeBands,
    patternEvents: patternInsights.events,
    patternStats: patternInsights.stats,
    signal,
    tradePlan,
    executionAssumptions: {
      commissionBpsPerSide: toOptionalNumber(assumptions.commission_bps_per_side),
      slippageBpsPerSide: toOptionalNumber(assumptions.slippage_bps_per_side),
      roundTripCostPercent: isFiniteNumber(roundTripCostPercent) ? roundTripCostPercent : 0,
      patternMinSample:
        Number.isFinite(patternMinSample) && patternMinSample > 0 ? patternMinSample : DEFAULT_PATTERN_MIN_SAMPLE,
    },
    structure: payload.structure ?? null,
    indicatorSummary: payload.indicator_summary ?? null,
    meta: {
      ticker: payload.ticker,
      timeframe: payload.timeframe,
      period: payload.period,
    },
  };
}

function parseBacktestResponse(payload) {
  if (!payload || typeof payload !== 'object') {
    return null;
  }
  const equityCurve = normalizeLineSeries(
    Array.isArray(payload.equity_curve)
      ? payload.equity_curve.map((point) => ({
          time: point.time,
          value: point.equity,
        }))
      : [],
  );
  const drawdownCurve = normalizeLineSeries(
    Array.isArray(payload.drawdown_curve)
      ? payload.drawdown_curve.map((point) => ({
          time: point.time,
          value: point.drawdown_percent,
        }))
      : [],
  );
  const trades = Array.isArray(payload.trades) ? payload.trades : [];
  return {
    summary: payload.summary ?? {},
    trades,
    equityCurve,
    drawdownCurve,
    filterFunnel: payload.filter_funnel ?? null,
  };
}

function parseRobustnessResponse(payload) {
  if (!payload || typeof payload !== 'object') {
    return null;
  }
  const regimeTimeline = Array.isArray(payload.regime_timeline)
    ? payload.regime_timeline
        .map((point) => ({
          time: Number(point.time),
          regime: String(point.regime ?? 'range'),
        }))
        .filter((point) => isFiniteNumber(point.time))
    : [];
  return {
    train: payload.train ?? {},
    test: payload.test ?? {},
    robustness: payload.robustness ?? {},
    regimePerformance: payload.market_regime_performance ?? {},
    monteCarlo: payload.monte_carlo ?? null,
    risk: payload.risk ?? {},
    admission: payload.admission ?? {},
    edgeFound: Boolean(payload.edge_found),
    regimeTimeline,
    trainPeriod: payload.train_period ?? null,
    testPeriod: payload.test_period ?? null,
  };
}

function DashboardMainChart({
  candles,
  volume,
  ma50,
  ma200,
  signal,
  structure,
  trendLine,
  regimeBands,
  patternEvents,
  layers,
}) {
  const chartContainerRef = useRef(null);

  useEffect(() => {
    if (!chartContainerRef.current || candles.length === 0) {
      return undefined;
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#243240',
      },
      grid: {
        vertLines: { color: '#eff2f8' },
        horzLines: { color: '#eff2f8' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 460,
      rightPriceScale: {
        borderColor: '#d3deee',
      },
      timeScale: {
        borderColor: '#d3deee',
      },
    });

    if (layers.showTrend && Array.isArray(regimeBands) && regimeBands.length > 0) {
      const regimeSeries = chart.addSeries(HistogramSeries, {
        priceScaleId: 'regime',
        base: 0,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      regimeSeries.setData(regimeBands);
      chart.priceScale('regime').applyOptions({
        scaleMargins: {
          top: 0,
          bottom: 0,
        },
        visible: false,
        borderVisible: false,
      });
    }

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#26b87d',
      borderUpColor: '#26b87d',
      wickUpColor: '#26b87d',
      downColor: '#d14f65',
      borderDownColor: '#d14f65',
      wickDownColor: '#d14f65',
    });
    candleSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.05,
        bottom: 0.18,
      },
    });
    candleSeries.setData(
      candles.map((item) => ({
        time: item.time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      })),
    );

    if (layers.showTrend) {
      const ma50Series = chart.addSeries(LineSeries, {
        color: '#3568d4',
        lineWidth: 2,
        title: 'MA50',
      });
      ma50Series.setData(ma50);

      const ma200Series = chart.addSeries(LineSeries, {
        color: '#8a57db',
        lineWidth: 2,
        title: 'MA200',
      });
      ma200Series.setData(ma200);

      if (Array.isArray(trendLine?.data) && trendLine.data.length >= 2) {
        const trendColor =
          trendLine.direction === 'up' ? '#16976b' : trendLine.direction === 'down' ? '#c24d65' : '#5e6f83';
        const trendSeries = chart.addSeries(LineSeries, {
          color: trendColor,
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          title: `Trend ${String(trendLine.direction ?? 'flat').toUpperCase()}`,
          priceLineVisible: false,
        });
        trendSeries.setData(trendLine.data);
      }
    }

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      scaleMargins: {
        top: 0.86,
        bottom: 0,
      },
    });
    volumeSeries.setData(volume);
    chart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.86,
        bottom: 0,
      },
      visible: false,
      borderVisible: false,
    });

    const markerPlugin = createSeriesMarkers(candleSeries, []);
    const lastCandle = candles[candles.length - 1];
    const signalType = String(signal?.signal ?? 'none').toLowerCase();
    const markers = [];
    if (lastCandle && signalType) {
      const markerColor = signalType === 'long' ? '#26b87d' : signalType === 'short' ? '#d14f65' : '#6c7a8a';
      const markerShape = signalType === 'long' ? 'arrowUp' : signalType === 'short' ? 'arrowDown' : 'circle';
      const markerPosition = signalType === 'long' ? 'belowBar' : signalType === 'short' ? 'aboveBar' : 'inBar';
      markers.push({
        time: lastCandle.time,
        position: markerPosition,
        color: markerColor,
        shape: markerShape,
        text: `${signalType.toUpperCase()} ${String(signal?.confidence ?? 'none')}`,
      });
    }

    if (layers.showPatterns && Array.isArray(patternEvents) && patternEvents.length > 0) {
      for (const event of patternEvents.slice(-80)) {
        const markerDirection = String(event.direction ?? 'none');
        const markerColor =
          markerDirection === 'long' ? '#189567' : markerDirection === 'short' ? '#c45064' : '#5072b7';
        const markerShape = markerDirection === 'long' ? 'arrowUp' : markerDirection === 'short' ? 'arrowDown' : 'circle';
        const markerPosition = markerDirection === 'long' ? 'belowBar' : markerDirection === 'short' ? 'aboveBar' : 'inBar';
        const confidence = Number(event.confidence);
        markers.push({
          time: Number(event.time),
          position: markerPosition,
          color: markerColor,
          shape: markerShape,
          text: `${String(event.code ?? 'PT')} ${isFiniteNumber(confidence) ? `${confidence}%` : ''}`.trim(),
        });
      }
    }

    if (layers.showStructure && Boolean(structure?.breakout) && lastCandle) {
      markers.push({
        time: lastCandle.time,
        position: 'aboveBar',
        color: '#df9931',
        shape: 'square',
        text: 'BREAKOUT',
      });
    }
    markerPlugin.setMarkers(markers.sort((left, right) => Number(left.time) - Number(right.time)));

    const levelConfigs = [
      { key: 'entry', color: '#3568d4', title: 'ENTRY', style: LineStyle.Dashed },
      { key: 'stop', color: '#d14f65', title: 'STOP', style: LineStyle.Dotted },
      { key: 'target', color: '#26b87d', title: 'TARGET', style: LineStyle.Dashed },
    ];
    for (const level of levelConfigs) {
      const value = Number(signal?.[level.key]);
      if (!isFiniteNumber(value) || value <= 0 || signalType === 'none') {
        continue;
      }
      candleSeries.createPriceLine({
        price: value,
        color: level.color,
        lineWidth: 2,
        lineStyle: level.style,
        axisLabelVisible: true,
        title: level.title,
      });
    }

    if (layers.showStructure) {
      const swingLevels = [
        { value: Number(structure?.last_swing_high), title: 'SWING HIGH', color: '#e39a2f', style: LineStyle.Dashed },
        { value: Number(structure?.last_swing_low), title: 'SWING LOW', color: '#5374be', style: LineStyle.Dashed },
      ];
      for (const level of swingLevels) {
        if (!isFiniteNumber(level.value)) {
          continue;
        }
        candleSeries.createPriceLine({
          price: level.value,
          color: level.color,
          lineWidth: 1,
          lineStyle: level.style,
          axisLabelVisible: true,
          title: level.title,
        });
      }
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (!chartContainerRef.current) {
        return;
      }
      chart.applyOptions({ width: chartContainerRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [candles, volume, ma50, ma200, signal, structure, trendLine, regimeBands, patternEvents, layers]);

  return <div className="chart chart-main" ref={chartContainerRef} />;
}

function CompactLineChart({ data, title, color, minValue, maxValue }) {
  const chartContainerRef = useRef(null);

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) {
      return undefined;
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#5c6c80',
      },
      grid: {
        vertLines: { color: '#f3f6fb' },
        horzLines: { color: '#f3f6fb' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 160,
      rightPriceScale: {
        borderColor: '#e0e8f4',
      },
      timeScale: {
        borderColor: '#e0e8f4',
        timeVisible: true,
      },
    });

    const lineSeries = chart.addSeries(LineSeries, {
      color,
      lineWidth: 2,
      title,
    });
    lineSeries.setData(data);

    if (isFiniteNumber(minValue)) {
      lineSeries.createPriceLine({
        price: Number(minValue),
        color: '#c7d2e4',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `MIN ${minValue}`,
      });
    }
    if (isFiniteNumber(maxValue)) {
      lineSeries.createPriceLine({
        price: Number(maxValue),
        color: '#c7d2e4',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `MAX ${maxValue}`,
      });
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (!chartContainerRef.current) {
        return;
      }
      chart.applyOptions({ width: chartContainerRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data, title, color, minValue, maxValue]);

  return <div className="chart chart-compact" ref={chartContainerRef} />;
}

function MetricCard({ label, value, tone = 'neutral' }) {
  return (
    <article className={`metricCard ${tone}`}>
      <span className="metricLabel">{label}</span>
      <strong className="metricValue">{value}</strong>
    </article>
  );
}

function PortfolioSignalsScreen({
  currentSystemName,
  scanTimeframes,
  scanModels,
  candidateLimit,
  portfolioAmount,
  onPortfolioAmountChange,
  onPortfolioAmountCommit,
  isSavingPortfolio,
  portfolioError,
  candidateScan,
  onScanCandidates,
  scanHistory,
  isLoadingScanHistory,
  scanHistoryError,
}) {
  const scanUniverseLabel = `all TF (${scanTimeframes.length}) x all models (${scanModels.length})`;
  const sessions = Array.isArray(scanHistory?.sessions) ? scanHistory.sessions : [];
  const scans = Array.isArray(scanHistory?.scans) ? scanHistory.scans : [];

  return (
    <div className="screenStack portfolioScreen">
      <section className="funnelPanel portfolioSummaryPanel">
        <h3>Portfolio Controls</h3>
        <div className="portfolioControlRow">
          <label className="portfolioControl">
            <span>Portfolio Amount</span>
            <input
              className="portfolioControlInput"
              type="number"
              min="0"
              step="1000"
              value={String(portfolioAmount)}
              onChange={(event) => onPortfolioAmountChange(event.target.value)}
              onBlur={onPortfolioAmountCommit}
            />
          </label>
          <div className="portfolioHint">
            Эта сумма используется в поиске сигналов и расчетах риска.
            {isSavingPortfolio ? ' Сохранение портфеля...' : ''}
          </div>
        </div>
        {portfolioError ? <div className="instrumentStatus error">{portfolioError}</div> : null}
        <div className="funnelGrid">
          <MetricCard label="Portfolio Amount" value={formatPrice(Number(portfolioAmount))} />
          <MetricCard label="System" value={String(currentSystemName || '—')} />
          <MetricCard label="Scan Universe" value={scanUniverseLabel} />
          <MetricCard label="Scan Top" value={String(candidateLimit)} />
          <MetricCard label="LONG Candidates" value={String(candidateScan.long.length)} tone="positive" />
          <MetricCard label="SHORT Candidates" value={String(candidateScan.short.length)} tone="negative" />
          <MetricCard label="Last Scan" value={candidateScan.lastUpdated || '—'} />
        </div>
      </section>

      <section className="candidatePanel candidatePanelWide portfolioCandidatesPanel">
        <div className="candidatePanelHeader">
          <h3>Signal Candidates</h3>
          <button type="button" className="scanButton" onClick={onScanCandidates} disabled={candidateScan.isScanning}>
            {candidateScan.isScanning ? 'Scanning...' : 'Scan'}
          </button>
        </div>
        <p className="candidateMeta">
          {scanUniverseLabel} / top {candidateLimit} / deposit {formatPrice(Number(portfolioAmount))}
          {` / system ${String(currentSystemName || '—')}`}
          {candidateScan.lastUpdated ? ` / updated ${candidateScan.lastUpdated}` : ''}
        </p>
        {candidateScan.isScanning ? (
          <div className="instrumentStatus">
            Сканирование: {candidateScan.scanned}/{candidateScan.total}
          </div>
        ) : null}
        {candidateScan.error ? <div className="instrumentStatus error">{candidateScan.error}</div> : null}
        <div className="candidateColumns candidateColumnsWide">
          <div className="candidateColumn">
            <h4>LONG ({candidateScan.long.length})</h4>
            <ul className="candidateList">
              {candidateScan.long.length === 0 ? <li className="candidateEmpty">Нет LONG кандидатов</li> : null}
              {candidateScan.long.map((item) => (
                <li key={`long-${item.scanKey}`}>
                  <div className="candidateItem long">
                    <span className="candidateTop">
                      <strong>{item.symbol}</strong>
                      <span>{String(item.confidence).toUpperCase()}</span>
                    </span>
                    <span className="candidateBottom">
                      {item.timeframe} / {item.model} / {String(currentSystemName || '—')} | RR{' '}
                      {formatNumber(Number(item.rr), 2)} | {item.regime}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
          <div className="candidateColumn">
            <h4>SHORT ({candidateScan.short.length})</h4>
            <ul className="candidateList">
              {candidateScan.short.length === 0 ? <li className="candidateEmpty">Нет SHORT кандидатов</li> : null}
              {candidateScan.short.map((item) => (
                <li key={`short-${item.scanKey}`}>
                  <div className="candidateItem short">
                    <span className="candidateTop">
                      <strong>{item.symbol}</strong>
                      <span>{String(item.confidence).toUpperCase()}</span>
                    </span>
                    <span className="candidateBottom">
                      {item.timeframe} / {item.model} / {String(currentSystemName || '—')} | RR{' '}
                      {formatNumber(Number(item.rr), 2)} | {item.regime}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="funnelPanel portfolioHistoryPanel">
        <h3>Scan History</h3>
        {isLoadingScanHistory ? <div className="instrumentStatus">Загрузка истории сканирования...</div> : null}
        {scanHistoryError ? <div className="instrumentStatus error">{scanHistoryError}</div> : null}
        {!isLoadingScanHistory && !scanHistoryError && sessions.length === 0 ? (
          <div className="instrumentStatus">История сканирования пуста</div>
        ) : null}
        {sessions.length > 0 ? (
          <div className="tradesTableWrap">
            <table className="tradesTable">
              <thead>
                <tr>
                  <th>Когда</th>
                  <th>Система</th>
                  <th>Модели</th>
                  <th>Кандидаты</th>
                  <th>Tradable</th>
                  <th>Scan Key</th>
                </tr>
              </thead>
              <tbody>
                {sessions.slice(0, 20).map((item) => (
                  <tr key={String(item.scan_key)}>
                    <td>{formatDateTime(item.created_at)}</td>
                    <td>{Array.isArray(item.systems) && item.systems.length > 0 ? item.systems.join(', ') : '—'}</td>
                    <td>{Array.isArray(item.models) && item.models.length > 0 ? item.models.join(', ') : '—'}</td>
                    <td>{String(item.count ?? 0)}</td>
                    <td>{String(item.tradable_count ?? 0)}</td>
                    <td>{String(item.scan_key ?? '')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}

        {scans.length > 0 ? (
          <div className="tradesTableWrap">
            <table className="tradesTable">
              <thead>
                <tr>
                  <th>Когда</th>
                  <th>Тикер</th>
                  <th>Signal</th>
                  <th>TF</th>
                  <th>Model</th>
                  <th>System</th>
                  <th>Entry</th>
                  <th>RR</th>
                </tr>
              </thead>
              <tbody>
                {scans.slice(0, 80).map((item) => (
                  <tr key={String(item.id ?? `${item.scan_key}-${item.symbol}-${item.timeframe}-${item.model_name}`)}>
                    <td>{formatDateTime(item.created_at)}</td>
                    <td>{String(item.symbol ?? '—')}</td>
                    <td>{String(item.signal ?? 'none').toUpperCase()}</td>
                    <td>{String(item.timeframe ?? '—')}</td>
                    <td>{String(item.model_name ?? item.model ?? '—')}</td>
                    <td>{String(item.system_name ?? '—')}</td>
                    <td>{formatPrice(Number(item.entry))}</td>
                    <td>{formatNumber(Number(item.rr), 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function TechnicalAnalysisScreen({ marketData, isLoading, error, timeframe, layers, onToggleLayer }) {
  if (isLoading) {
    return <div className="screenStatus">Загрузка market-данных...</div>;
  }
  if (error) {
    return <div className="screenStatus error">{error}</div>;
  }
  if (!marketData || marketData.candles.length === 0) {
    return <div className="screenStatus">Нет данных для market-экрана</div>;
  }

  const signal = marketData.signal ?? {};
  const structure = marketData.structure ?? {};
  const indicatorSummary = marketData.indicatorSummary ?? {};
  const signalClass = String(signal.signal ?? 'none').toLowerCase();
  const signalTone = signalClass === 'long' ? 'positive' : signalClass === 'short' ? 'negative' : 'neutral';
  const patternStats = Array.isArray(marketData.patternStats) ? marketData.patternStats : [];
  const trendInfo = marketData.trendLine ?? {};
  const tradePlan = marketData.tradePlan ?? {};
  const assumptions = marketData.executionAssumptions ?? {};
  const planTradable = Boolean(tradePlan.tradable);
  const planStatusClass = planTradable ? 'long' : tradePlan.status === 'no_signal' ? 'none' : 'short';
  const roundTripCostPercent = Number(assumptions.roundTripCostPercent);
  const minPatternSample = Number(assumptions.patternMinSample);

  return (
    <div className="screenStack">
      <div className="tagRow">
        <span className={`signalTag ${signalClass}`}>{String(signal.signal ?? 'none').toUpperCase()}</span>
        <span className={`signalTag ${planStatusClass}`}>{planTradable ? 'TRADABLE' : 'NOT TRADABLE'}</span>
        <span className="signalTag">TF: {timeframe}</span>
        <span className="signalTag">Confidence: {String(signal.confidence ?? 'none')}</span>
        <span className="signalTag">Regime: {String(signal.market_regime ?? 'unknown')}</span>
        <span className="signalTag">Phase: {String(signal.phase ?? 'unknown')}</span>
        <span className="signalTag">Structure: {String(structure.structure ?? 'unknown')}</span>
        <span className="signalTag">Breakout: {structure.breakout ? 'yes' : 'no'}</span>
        <span className="signalTag">
          Trend: {String(trendInfo.direction ?? 'flat')} ({formatSignedPercent(Number(trendInfo.slopePercent))})
        </span>
        <span className="signalTag">Costs RT: {formatPercent(roundTripCostPercent)}</span>
      </div>

      {!planTradable && Array.isArray(tradePlan.issues) && tradePlan.issues.length > 0 ? (
        <section className="riskAlert">
          <h3>Trade Plan Invalid</h3>
          <p>Сигнал не проходит валидацию плана сделки и не должен использоваться для входа.</p>
          <div className="riskIssues">
            {tradePlan.issues.map((issue) => (
              <span key={String(issue)} className="riskIssueChip">
                {String(issue)}
              </span>
            ))}
          </div>
        </section>
      ) : null}

      <div className="layerControls">
        <button
          type="button"
          className={`layerChip ${layers.showTrend ? 'active' : ''}`}
          onClick={() => onToggleLayer('showTrend')}
        >
          Trend Layer
        </button>
        <button
          type="button"
          className={`layerChip ${layers.showStructure ? 'active' : ''}`}
          onClick={() => onToggleLayer('showStructure')}
        >
          Structure Layer
        </button>
        <button
          type="button"
          className={`layerChip ${layers.showPatterns ? 'active' : ''}`}
          onClick={() => onToggleLayer('showPatterns')}
        >
          Pattern Markers
        </button>
      </div>

      <div className="metricsGrid">
        <MetricCard label="Entry" value={formatPrice(Number(signal.entry))} tone={signalTone} />
        <MetricCard label="Stop" value={formatPrice(Number(signal.stop))} tone="negative" />
        <MetricCard label="Target" value={formatPrice(Number(signal.target))} tone="positive" />
        <MetricCard label="RR" value={formatNumber(Number(signal.rr), 2)} />
        <MetricCard label="Trend Strength" value={formatPercent(Number(structure.trend_strength))} />
        <MetricCard label="RSI" value={formatNumber(Number(indicatorSummary.rsi), 2)} />
        <MetricCard label="ATR" value={formatNumber(Number(indicatorSummary.atr), 4)} />
        <MetricCard
          label="Volume Ratio"
          value={`${formatNumber(Number(indicatorSummary.volume_ratio), 2)}x / ${formatNumber(
            Number(indicatorSummary.volume_threshold_ratio),
            2,
          )}x`}
        />
      </div>

      <DashboardMainChart
        candles={marketData.candles}
        volume={marketData.volume}
        ma50={marketData.ma50}
        ma200={marketData.ma200}
        signal={signal}
        structure={structure}
        trendLine={marketData.trendLine}
        regimeBands={marketData.regimeBands}
        patternEvents={marketData.patternEvents}
        layers={layers}
      />

      {patternStats.length > 0 ? (
        <section className="funnelPanel">
          <h3>Pattern Diagnostics</h3>
          <div className="patternStatsGrid">
            {patternStats.map((item) => (
              <article
                className={`patternStatCard ${
                  item.sampleQualified ? (item.edgeQualified ? 'tradable' : 'notTradable') : 'lowSample'
                }`}
                key={String(item.pattern)}
              >
                <h4>{String(item.label ?? item.pattern)}</h4>
                <p>
                  Status:{' '}
                  {item.sampleQualified ? (item.edgeQualified ? 'Tradable' : 'Not tradable') : 'Low sample'}
                </p>
                <p>Count: {String(item.count ?? 0)}</p>
                <p>Min sample: {String(Number.isFinite(minPatternSample) ? minPatternSample : item.minSample ?? DEFAULT_PATTERN_MIN_SAMPLE)}</p>
                <p>Winrate: {formatPercent(Number(item.winrate))}</p>
                <p>Avg Return (+8): {formatSignedPercent(Number(item.avgReturnPct))}</p>
                <p>Avg Net Return: {formatSignedPercent(Number(item.avgReturnNetPct))}</p>
                <p>Confidence: {formatNumber(Number(item.avgConfidence), 1)}</p>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      <div className="subChartGrid">
        <section className="subPanel">
          <h3>RSI</h3>
          <CompactLineChart
            data={marketData.rsi}
            title="RSI"
            color="#c26ce7"
            minValue={Number(indicatorSummary.rsi_oversold ?? 30)}
            maxValue={Number(indicatorSummary.rsi_overbought ?? 70)}
          />
        </section>
        <section className="subPanel">
          <h3>ATR</h3>
          <CompactLineChart data={marketData.atr} title="ATR" color="#3f7ad9" />
        </section>
        <section className="subPanel">
          <h3>Volume Ratio</h3>
          <CompactLineChart
            data={marketData.volumeRatio}
            title="Vol Ratio"
            color="#24a38c"
            minValue={Number(indicatorSummary.volume_threshold_ratio ?? 1.5)}
          />
        </section>
      </div>
    </div>
  );
}

function BacktestScreen({ backtestData, isLoading, error }) {
  if (isLoading) {
    return <div className="screenStatus">Запуск бэктеста...</div>;
  }
  if (error) {
    return <div className="screenStatus error">{error}</div>;
  }
  if (!backtestData) {
    return <div className="screenStatus">Нет данных бэктеста</div>;
  }

  const summary = backtestData.summary ?? {};
  const funnel = backtestData.filterFunnel ?? null;
  const trades = backtestData.trades ?? [];

  return (
    <div className="screenStack">
      <div className="metricsGrid">
        <MetricCard label="Trades" value={String(summary.total_trades ?? 0)} />
        <MetricCard label="Winrate" value={formatPercent(Number(summary.winrate))} />
        <MetricCard label="Profit Factor" value={formatNumber(Number(summary.profit_factor), 2)} />
        <MetricCard label="Expectancy" value={formatNumber(Number(summary.expectancy), 2)} />
        <MetricCard label="Sharpe" value={formatNumber(Number(summary.sharpe_ratio), 2)} />
        <MetricCard label="Return" value={formatPercent(Number(summary.return_pct))} tone="positive" />
        <MetricCard label="Max DD" value={formatPercent(Number(summary.max_drawdown_percent))} tone="negative" />
        <MetricCard label="Final Balance" value={formatPrice(Number(summary.final_balance))} />
      </div>

      <div className="subChartGrid">
        <section className="subPanel">
          <h3>Equity Curve</h3>
          <CompactLineChart data={backtestData.equityCurve} title="Equity" color="#2876d8" />
        </section>
        <section className="subPanel">
          <h3>Drawdown Curve</h3>
          <CompactLineChart data={backtestData.drawdownCurve} title="Drawdown %" color="#d14f65" minValue={0} />
        </section>
      </div>

      {funnel ? (
        <section className="funnelPanel">
          <h3>Filter Funnel</h3>
          <div className="funnelGrid">
            <MetricCard label="Potential setups" value={String(funnel.potential_setups ?? 0)} />
            <MetricCard label="Filtered trend" value={String(funnel.filtered_by_trend ?? 0)} />
            <MetricCard label="Filtered volume" value={String(funnel.filtered_by_volume ?? 0)} />
            <MetricCard label="Filtered RR" value={String(funnel.filtered_by_rr ?? 0)} />
            <MetricCard label="Filtered ATR" value={String(funnel.filtered_by_atr ?? 0)} />
            <MetricCard label="Final trades" value={String(funnel.final_trades ?? 0)} tone="positive" />
          </div>
        </section>
      ) : null}

      <section className="tradesPanel">
        <h3>Trades</h3>
        <div className="tradesTableWrap">
          <table className="tradesTable">
            <thead>
              <tr>
                <th>Entry</th>
                <th>Exit</th>
                <th>Dir</th>
                <th>PnL</th>
                <th>Exit Reason</th>
                <th>RR planned</th>
                <th>RR actual</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {trades.slice(0, 120).map((trade, index) => {
                const pnl = Number(trade.pnl);
                const tone = pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral';
                return (
                  <tr key={`${trade.entry_time}-${trade.exit_time}-${index}`}>
                    <td>{String(trade.entry_time ?? '')}</td>
                    <td>{String(trade.exit_time ?? '')}</td>
                    <td>{String(trade.direction ?? '').toUpperCase()}</td>
                    <td className={tone}>{formatPrice(pnl)}</td>
                    <td>{String(trade.exit_reason ?? '')}</td>
                    <td>{formatNumber(Number(trade.rr_planned), 2)}</td>
                    <td>{formatNumber(Number(trade.rr_actual), 2)}</td>
                    <td>{String(trade.duration_candles ?? 0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function RobustnessScreen({ robustData, isLoading, error }) {
  if (isLoading) {
    return <div className="screenStatus">Расчет walk-forward и risk...</div>;
  }
  if (error) {
    return <div className="screenStatus error">{error}</div>;
  }
  if (!robustData) {
    return <div className="screenStatus">Нет данных robustness</div>;
  }

  const train = robustData.train ?? {};
  const test = robustData.test ?? {};
  const robust = robustData.robustness ?? {};
  const risk = robustData.risk ?? {};
  const monteCarlo = robustData.monteCarlo ?? null;
  const regimePerformance = robustData.regimePerformance ?? {};

  return (
    <div className="screenStack">
      <div className="tagRow">
        <span className={`signalTag ${robustData.edgeFound ? 'long' : 'short'}`}>
          Edge: {robustData.edgeFound ? 'FOUND' : 'NOT FOUND'}
        </span>
        <span className="signalTag">Train PF: {formatNumber(Number(train.profit_factor), 2)}</span>
        <span className="signalTag">Test PF: {formatNumber(Number(test.profit_factor), 2)}</span>
        <span className="signalTag">Stability: {formatNumber(Number(robust.stability_ratio), 2)}</span>
        <span className="signalTag">Robustness: {formatNumber(Number(robust.robustness_score), 4)}</span>
      </div>

      <div className="metricsGrid">
        <MetricCard label="Train PF" value={formatNumber(Number(train.profit_factor), 2)} />
        <MetricCard label="Test PF" value={formatNumber(Number(test.profit_factor), 2)} />
        <MetricCard label="Train DD" value={formatPercent(Number(train.max_drawdown_percent))} tone="negative" />
        <MetricCard label="Test DD" value={formatPercent(Number(test.max_drawdown_percent))} tone="negative" />
        <MetricCard label="Stability Ratio" value={formatNumber(Number(robust.stability_ratio), 2)} />
        <MetricCard label="Risk of Ruin" value={formatPercent(Number(risk.risk_of_ruin))} tone="negative" />
        <MetricCard label="Kelly %" value={formatPercent(Number(risk.kelly_percent))} />
        <MetricCard label="Robustness Score" value={formatNumber(Number(robust.robustness_score), 4)} />
      </div>

      {monteCarlo ? (
        <section className="funnelPanel">
          <h3>Monte Carlo</h3>
          <div className="funnelGrid">
            <MetricCard
              label="Worst DD %"
              value={formatPercent(Number(monteCarlo.worst_drawdown_percent))}
              tone="negative"
            />
            <MetricCard label="Q5 Equity" value={formatPrice(Number(monteCarlo.quantile_5_equity))} />
            <MetricCard label="Median Equity" value={formatPrice(Number(monteCarlo.median_final_equity))} />
            <MetricCard label="Stability Score" value={formatNumber(Number(monteCarlo.stability_score), 4)} />
            <MetricCard
              label="Prob DD > 30%"
              value={formatPercent(Number(monteCarlo.probability_drawdown_over_30))}
              tone="negative"
            />
          </div>
        </section>
      ) : null}

      <section className="funnelPanel">
        <h3>Performance by Regime</h3>
        <div className="regimeGrid">
          {['trend', 'range', 'high_volatility'].map((regimeKey) => {
            const row = regimePerformance[regimeKey] ?? {};
            return (
              <article className="regimeCard" key={regimeKey}>
                <h4>{regimeKey}</h4>
                <p>Trades: {String(row.trades ?? 0)}</p>
                <p>PF: {formatNumber(Number(row.pf), 2)}</p>
                <p>Winrate: {formatPercent(Number(row.winrate))}</p>
                <p>MaxDD: {formatPercent(Number(row.maxdd))}</p>
                <p>Return: {formatPercent(Number(row.return_pct))}</p>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}

function SystemTestingScreen({
  selectedTab,
  onSelectTab,
  backtestData,
  isLoadingBacktest,
  backtestError,
  robustData,
  isLoadingRobust,
  robustError,
}) {
  return (
    <div className="screenStack">
      <nav className="screenTabs" aria-label="Testing tabs">
        {TESTING_TAB_OPTIONS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            className={`screenTab ${selectedTab === tab.key ? 'active' : ''}`}
            onClick={() => onSelectTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      {selectedTab === 'backtest' ? (
        <BacktestScreen backtestData={backtestData} isLoading={isLoadingBacktest} error={backtestError} />
      ) : null}
      {selectedTab === 'robustness' ? (
        <RobustnessScreen robustData={robustData} isLoading={isLoadingRobust} error={robustError} />
      ) : null}
    </div>
  );
}

function SystemSettingsScreen({
  config,
  systems,
  selectedSystemId,
  currentSystemId,
  isAuthenticated,
  onSelectSystem,
  onAddSystem,
  onSetCurrent,
  isSavingSystem,
  systemsError,
  onResetDefaults,
  onConfigChange,
  onConfigCommit,
}) {
  const sections = [
    {
      title: 'Market & Signals',
      fields: [
        { key: 'deposit', label: 'Deposit', step: 1000, min: 0 },
        { key: 'marketLimit', label: 'Market candles limit', step: 50, min: 100 },
        { key: 'commissionBps', label: 'Commission (bps/side)', step: 1, min: 0 },
        { key: 'slippageBps', label: 'Slippage (bps/side)', step: 1, min: 0 },
        { key: 'patternMinSample', label: 'Pattern min sample', step: 1, min: 1 },
      ],
    },
    {
      title: 'Signal Scanner',
      fields: [
        { key: 'candidateScanLimit', label: 'Scan top instruments', step: 1, min: 1 },
        { key: 'candidateScanConcurrency', label: 'Scan concurrency', step: 1, min: 1 },
      ],
    },
    {
      title: 'Backtest',
      fields: [
        { key: 'backtestLimit', label: 'Backtest candles limit', step: 50, min: 200 },
        { key: 'backtestLookbackWindow', label: 'Lookback window', step: 10, min: 20 },
        { key: 'backtestMaxHoldingCandles', label: 'Max holding candles', step: 1, min: 1 },
      ],
    },
    {
      title: 'Robustness',
      fields: [
        { key: 'robustnessLimit', label: 'Robustness candles limit', step: 50, min: 200 },
        { key: 'monteCarloSimulations', label: 'Monte Carlo simulations', step: 50, min: 50 },
        { key: 'trainRatio', label: 'Train ratio', step: 0.05, min: 0.1, max: 0.95 },
      ],
    },
  ];

  return (
    <div className="screenStack">
      <section className="funnelPanel">
        <div className="settingsHeader">
          <div>
            <h3>System Parameters</h3>
            <p>Параметры применяются сразу для market, сканера, backtest и robustness.</p>
          </div>
          <div className="settingsActions">
            <div className="systemPicker">
              <select
                value={selectedSystemId}
                onChange={(event) => onSelectSystem(event.target.value)}
                disabled={!isAuthenticated || systems.length === 0}
              >
                {systems.map((system) => (
                  <option key={system.id} value={system.id}>
                    {system.name}{system.id === currentSystemId ? " (current)" : ""}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="scanButton systemAddButton"
                onClick={onAddSystem}
                title="Add new system"
                disabled={!isAuthenticated || isSavingSystem}
              >
                +
              </button>
              <button
                type="button"
                className="scanButton"
                onClick={onSetCurrent}
                disabled={!isAuthenticated || isSavingSystem || selectedSystemId === currentSystemId}
                title="Сделать выбранную систему текущей"
              >
                {isSavingSystem ? "Saving..." : "Set current"}
              </button>
            </div>
            <button type="button" className="scanButton" onClick={onResetDefaults} disabled={isSavingSystem}>
              Reset defaults
            </button>
          </div>
        </div>
      </section>

      {systemsError ? <div className="screenStatus error">{systemsError}</div> : null}

      {sections.map((section) => (
        <section className="funnelPanel" key={section.title}>
          <h3>{section.title}</h3>
          <div className="settingsGrid">
            {section.fields.map((field) => (
              <label className="settingField" key={field.key}>
                <span>{field.label}</span>
                <input
                  type="number"
                  step={field.step}
                  min={field.min}
                  max={field.max}
                  value={String(config[field.key])}
                  onChange={(event) => onConfigChange(field.key, event.target.value)}
                  onBlur={() => onConfigCommit?.(field.key)}
                />
              </label>
            ))}
          </div>
        </section>
      ))}

      <section className="funnelPanel">
        <h3>Runtime</h3>
        <div className="tagRow">
          <span className="signalTag">API: {API_BASE_URL || 'same-origin'}</span>
          <span className="signalTag">Model defaults managed in UI</span>
        </div>
      </section>
    </div>
  );
}

function GuestMarketingScreen() {
  const features = [
    {
      title: 'Portfolio & Signals',
      description: 'Сканер рынка ищет кандидатов LONG/SHORT и строит готовый trade plan с RR, stop и target.',
    },
    {
      title: 'Technical Analysis',
      description: 'Свечной график с MA50/MA200, структурой рынка, фазами, паттернами и диагностикой сигналов.',
    },
    {
      title: 'System Testing',
      description: 'Backtest, robustness, Monte Carlo и риск-метрики для оценки устойчивости стратегии.',
    },
    {
      title: 'System Profiles',
      description: 'Именованные системы с версионированием параметров, привязкой к пользователю и истории запусков.',
    },
  ];

  return (
    <div className="guestScreen">
      <section className="guestHero">
        <span className="guestEyebrow">Trading System Platform</span>
        <h2>Одна платформа для сигналов, анализа и тестирования торговой системы</h2>
        <p>
          Войдите по email и паролю, чтобы сохранять системы, портфель, результаты backtest и запускать
          персональный сканер рынка.
        </p>
        <div className="guestChips">
          <span className="signalTag">Live market scan</span>
          <span className="signalTag">Risk-first trade plans</span>
          <span className="signalTag">Backtest + robustness</span>
          <span className="signalTag">User-bound data storage</span>
        </div>
      </section>

      <section className="guestFeatureGrid">
        {features.map((item) => (
          <article key={item.title} className="guestFeatureCard">
            <h3>{item.title}</h3>
            <p>{item.description}</p>
          </article>
        ))}
      </section>

      <section className="guestFlow">
        <h3>Как начать</h3>
        <div className="guestFlowGrid">
          <div className="guestStep">
            <strong>1</strong>
            <p>Нажмите кнопку "Вход", затем зарегистрируйтесь или войдите в модальном окне.</p>
          </div>
          <div className="guestStep">
            <strong>2</strong>
            <p>Создайте свою систему и сделайте ее текущей.</p>
          </div>
          <div className="guestStep">
            <strong>3</strong>
            <p>Укажите размер портфеля и запустите сканирование.</p>
          </div>
          <div className="guestStep">
            <strong>4</strong>
            <p>Проверьте стратегию в Backtest и Robustness.</p>
          </div>
        </div>
      </section>
    </div>
  );
}

function AuthModal({
  isOpen,
  email,
  password,
  onEmailChange,
  onPasswordChange,
  onLogin,
  onRegister,
  onClose,
  isAuthPending,
  authError,
}) {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="authModalOverlay" role="presentation" onClick={onClose}>
      <section
        className="authModal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-modal-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="authModalHeader">
          <h3 id="auth-modal-title">Регистрация и вход</h3>
          <button type="button" className="authModalClose" onClick={onClose} aria-label="Закрыть окно авторизации">
            ×
          </button>
        </header>
        <p>Создайте аккаунт или войдите, чтобы сохранять системы и результаты тестов в вашем профиле.</p>
        <p>Демо-пользователь: <strong>admin / admin</strong>.</p>
        <div className="guestAuthForm">
          <label className="settingField">
            <span>Email</span>
            <input
              type="email"
              value={email}
              onChange={(event) => onEmailChange(event.target.value)}
              placeholder="user@company.com"
            />
          </label>
          <label className="settingField">
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(event) => onPasswordChange(event.target.value)}
              placeholder="Минимум 8 символов"
            />
          </label>
          <div className="guestAuthActions">
            <button type="button" className="scanButton" onClick={onRegister} disabled={isAuthPending}>
              {isAuthPending ? 'Please wait...' : 'Register'}
            </button>
            <button type="button" className="scanButton" onClick={onLogin} disabled={isAuthPending}>
              {isAuthPending ? 'Please wait...' : 'Login'}
            </button>
          </div>
        </div>
        {authError ? <div className="screenStatus error">{authError}</div> : null}
      </section>
    </div>
  );
}

export default function App() {
  const [authToken, setAuthToken] = useState(() => {
    if (typeof window === 'undefined') {
      return '';
    }
    return String(window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) ?? '').trim();
  });
  const [authPassword, setAuthPassword] = useState('');
  const [isAuthPending, setIsAuthPending] = useState(false);
  const [authError, setAuthError] = useState('');
  const [knownUserEmails, setKnownUserEmails] = useState(() => readStoredUserEmails());
  const [activeUserEmail, setActiveUserEmail] = useState(() => {
    if (typeof window === 'undefined') {
      return '';
    }
    const stored = window.localStorage.getItem(USER_EMAIL_STORAGE_KEY);
    return normalizeUserEmail(stored);
  });
  const [userEmailDraft, setUserEmailDraft] = useState(() => {
    if (typeof window === 'undefined') {
      return '';
    }
    const stored = window.localStorage.getItem(USER_EMAIL_STORAGE_KEY);
    return normalizeUserEmail(stored);
  });
  const [selectedMarketKey, setSelectedMarketKey] = useState(MARKET_OPTIONS[0].key);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [selectedScreen, setSelectedScreen] = useState('portfolio');
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [selectedTestingTab, setSelectedTestingTab] = useState('backtest');
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
  const [savedSystems, setSavedSystems] = useState([]);
  const [selectedSystemId, setSelectedSystemId] = useState("");
  const [currentSystemId, setCurrentSystemId] = useState("");
  const [systemsError, setSystemsError] = useState("");
  const [isSavingSystem, setIsSavingSystem] = useState(false);
  const [systemConfig, setSystemConfig] = useState({ ...SYSTEM_DEFAULTS });
  const [portfolioState, setPortfolioState] = useState({
    balance: SYSTEM_DEFAULTS.deposit,
    currency: 'RUB',
  });
  const [isLoadingPortfolio, setIsLoadingPortfolio] = useState(false);
  const [isSavingPortfolio, setIsSavingPortfolio] = useState(false);
  const [portfolioError, setPortfolioError] = useState('');
  const [marketLayers, setMarketLayers] = useState({
    showTrend: true,
    showStructure: true,
    showPatterns: true,
  });

  const [instruments, setInstruments] = useState([]);
  const [selectedInstrument, setSelectedInstrument] = useState(null);
  const [isLoadingInstruments, setIsLoadingInstruments] = useState(true);
  const [instrumentsError, setInstrumentsError] = useState('');

  const [marketData, setMarketData] = useState(null);
  const [isLoadingMarket, setIsLoadingMarket] = useState(false);
  const [marketError, setMarketError] = useState('');

  const [backtestData, setBacktestData] = useState(null);
  const [isLoadingBacktest, setIsLoadingBacktest] = useState(false);
  const [backtestError, setBacktestError] = useState('');
  const [backtestLoadedKey, setBacktestLoadedKey] = useState('');

  const [robustData, setRobustData] = useState(null);
  const [isLoadingRobust, setIsLoadingRobust] = useState(false);
  const [robustError, setRobustError] = useState('');
  const [robustLoadedKey, setRobustLoadedKey] = useState('');
  const [candidateScan, setCandidateScan] = useState({
    isScanning: false,
    error: '',
    scanned: 0,
    total: 0,
    long: [],
    short: [],
    lastUpdated: '',
    key: '',
  });
  const [scanHistory, setScanHistory] = useState({ scans: [], sessions: [] });
  const [isLoadingScanHistory, setIsLoadingScanHistory] = useState(false);
  const [scanHistoryError, setScanHistoryError] = useState('');
  const scanRequestRef = useRef(0);
  const isAuthenticated = Boolean(activeUserEmail);

  const selectedMarket = MARKET_OPTIONS.find((item) => item.key === selectedMarketKey) ?? MARKET_OPTIONS[0];
  const currentSystem = useMemo(
    () => savedSystems.find((system) => system.id === currentSystemId) ?? null,
    [savedSystems, currentSystemId],
  );
  const currentSystemName = String(currentSystem?.name ?? '').trim();
  const scannerConfig = useMemo(
    () => normalizeSystemConfig(currentSystem?.config),
    [currentSystem?.config],
  );
  const systemConfigKey = useMemo(
    () =>
      [
        portfolioState.balance,
        systemConfig.marketLimit,
        systemConfig.commissionBps,
        systemConfig.slippageBps,
        systemConfig.patternMinSample,
        systemConfig.candidateScanLimit,
        systemConfig.candidateScanConcurrency,
        systemConfig.backtestLimit,
        systemConfig.backtestLookbackWindow,
        systemConfig.backtestMaxHoldingCandles,
        systemConfig.robustnessLimit,
        systemConfig.monteCarloSimulations,
        systemConfig.trainRatio,
      ].join('|'),
    [systemConfig, portfolioState.balance],
  );
  const candidateScopeKey = useMemo(
    () =>
      [
        selectedMarket.key,
        currentSystemId || 'none',
        portfolioState.balance,
        scannerConfig.candidateScanLimit,
        scannerConfig.candidateScanConcurrency,
        scannerConfig.marketLimit,
        scannerConfig.patternMinSample,
        scannerConfig.commissionBps,
        scannerConfig.slippageBps,
      ].join('|'),
    [selectedMarket.key, currentSystemId, scannerConfig, portfolioState.balance],
  );
  const dataKey = useMemo(() => {
    const symbol = selectedInstrument?.symbol ?? '';
    return `${symbol}|${selectedMarket.key}|${selectedTimeframe}|${selectedModel}|${systemConfigKey}`;
  }, [selectedInstrument, selectedMarket, selectedTimeframe, selectedModel, systemConfigKey]);

  function applySystemsPayload(payload, preferredSystemId = "") {
    const systems = Array.isArray(payload?.systems)
      ? payload.systems
          .map((item) => {
            const id = String(item?.id ?? "").trim();
            const name = String(item?.name ?? "").trim();
            if (!id || !name) {
              return null;
            }
            return {
              id,
              name,
              config: normalizeSystemConfig(item?.config),
              model: String(item?.model ?? DEFAULT_MODEL),
              timeframe: String(item?.timeframe ?? "1h"),
              exchange: String(item?.exchange ?? "moex"),
              engine: String(item?.engine ?? "stock"),
              market: String(item?.market ?? "shares"),
              board: String(item?.board ?? ""),
              isCurrent: Boolean(item?.is_current),
            };
          })
          .filter(Boolean)
      : [];
    setSavedSystems(systems);
    if (systems.length === 0) {
      setSelectedSystemId("");
      setCurrentSystemId("");
      setSystemConfig({ ...SYSTEM_DEFAULTS });
      return;
    }

    const payloadCurrentId = String(payload?.current_system_id ?? "").trim();
    const targetId = String(preferredSystemId || payloadCurrentId).trim();
    const current =
      systems.find((system) => system.id === payloadCurrentId) ||
      systems.find((system) => system.isCurrent) ||
      systems[0];
    const selected =
      systems.find((system) => system.id === targetId) ||
      systems.find((system) => system.id === current.id) ||
      systems[0];

    setCurrentSystemId(current.id);
    setSelectedSystemId(selected.id);
    setSystemConfig({ ...selected.config });
    if (selected.model) {
      setSelectedModel(selected.model);
    }
    if (selected.timeframe) {
      setSelectedTimeframe(selected.timeframe);
    }
  }

  async function parseResponseError(response) {
    const text = (await response.text()).trim();
    if (!text) {
      return `HTTP ${response.status}`;
    }
    try {
      const payload = JSON.parse(text);
      const details = String(payload?.error ?? payload?.message ?? '').trim();
      if (details) {
        return `HTTP ${response.status}: ${details}`;
      }
    } catch (_error) {
      // Keep original text for non-JSON responses.
    }
    return `HTTP ${response.status}: ${text}`;
  }

  async function fetchAuthMe(token, signal) {
    const response = await fetch(buildAuthMeUrl(), {
      signal,
      cache: 'no-store',
      headers: buildAuthHeaders(token),
    });
    if (!response.ok) {
      throw new Error(await parseResponseError(response));
    }
    const payload = await response.json();
    const email = normalizeUserEmail(payload?.user?.email);
    if (!email) {
      throw new Error('Invalid auth payload');
    }
    return email;
  }

  async function submitAuth(mode) {
    const normalizedEmail = normalizeUserEmail(userEmailDraft);
    const password = String(authPassword ?? '');
    if (!normalizedEmail) {
      setAuthError('Укажите email.');
      return;
    }
    if (!password) {
      setAuthError('Укажите пароль.');
      return;
    }
    if (mode === 'register' && password.length < 8) {
      setAuthError('Пароль должен быть не короче 8 символов.');
      return;
    }

    setIsAuthPending(true);
    setAuthError('');
    try {
      const endpoint = mode === 'register' ? buildAuthRegisterUrl() : buildAuthLoginUrl();
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: normalizedEmail,
          password,
        }),
      });
      if (!response.ok) {
        throw new Error(await parseResponseError(response));
      }
      const payload = await response.json();
      const token = String(payload?.access_token ?? '').trim();
      const email = normalizeUserEmail(payload?.user?.email ?? normalizedEmail);
      if (!token || !email) {
        throw new Error('Invalid auth response');
      }
      setAuthToken(token);
      setActiveUserEmail(email);
      setUserEmailDraft(email);
      setAuthPassword('');
      setSelectedScreen('portfolio');
      setKnownUserEmails((previous) => Array.from(new Set([...previous, email])));
      setIsAuthModalOpen(false);
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'unknown error';
      setAuthError(`Ошибка авторизации: ${reason}`);
    } finally {
      setIsAuthPending(false);
    }
  }

  async function fetchSystems(preferredSystemId = "", signal) {
    const response = await fetch(buildSystemsUrl(activeUserEmail), {
      signal,
      cache: 'no-store',
      headers: buildAuthHeaders(authToken),
    });
    if (!response.ok) {
      throw new Error(await parseResponseError(response));
    }
    const payload = await response.json();
    applySystemsPayload(payload, preferredSystemId);
    return payload;
  }

  function applyPortfolioPayload(payload) {
    const rawPortfolio = payload?.portfolio ?? {};
    const parsedBalance = Number(rawPortfolio.balance);
    const nextBalance = Number.isFinite(parsedBalance) ? Math.max(0, parsedBalance) : SYSTEM_DEFAULTS.deposit;
    const nextCurrency = String(rawPortfolio.currency ?? 'RUB').trim().toUpperCase() || 'RUB';
    setPortfolioState({
      balance: nextBalance,
      currency: nextCurrency,
    });
    setSystemConfig((previous) => ({
      ...previous,
      deposit: nextBalance,
    }));
  }

  async function fetchPortfolio(signal) {
    const response = await fetch(buildPortfolioUrl(activeUserEmail), {
      signal,
      cache: 'no-store',
      headers: buildAuthHeaders(authToken),
    });
    if (!response.ok) {
      throw new Error(await parseResponseError(response));
    }
    const payload = await response.json();
    applyPortfolioPayload(payload);
    return payload;
  }

  async function fetchScans(signal, options = {}) {
    if (!isAuthenticated) {
      setScanHistory({ scans: [], sessions: [] });
      setScanHistoryError('');
      return { scans: [], sessions: [] };
    }
    const targetSystemId = Number(options.systemId ?? currentSystemId);
    setIsLoadingScanHistory(true);
    setScanHistoryError('');
    try {
      const response = await fetch(
        buildScansUrl(activeUserEmail, {
          systemId: Number.isInteger(targetSystemId) && targetSystemId > 0 ? targetSystemId : undefined,
          limit: 800,
        }),
        {
          signal,
          cache: 'no-store',
          headers: buildAuthHeaders(authToken),
        },
      );
      if (!response.ok) {
        throw new Error(await parseResponseError(response));
      }
      const payload = await response.json();
      setScanHistory({
        scans: Array.isArray(payload?.scans) ? payload.scans : [],
        sessions: Array.isArray(payload?.sessions) ? payload.sessions : [],
      });
      return payload;
    } catch (error) {
      if (signal?.aborted) {
        return { scans: [], sessions: [] };
      }
      const reason = error instanceof Error ? error.message : 'unknown error';
      setScanHistory({ scans: [], sessions: [] });
      setScanHistoryError(`Не удалось загрузить сканы: ${reason}`);
      return { scans: [], sessions: [] };
    } finally {
      if (!signal?.aborted) {
        setIsLoadingScanHistory(false);
      }
    }
  }

  async function persistScans({ scanKey, systemId, items }) {
    const resolvedSystemId = Number(systemId);
    if (!isAuthenticated || !Number.isInteger(resolvedSystemId) || resolvedSystemId <= 0) {
      return false;
    }
    const scans = Array.isArray(items) ? items : [];
    if (scans.length === 0) {
      return true;
    }
    try {
      const response = await fetch(buildCreateScansUrl(), {
        method: 'POST',
        headers: buildAuthHeaders(authToken, { 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          user_email: activeUserEmail,
          system_id: resolvedSystemId,
          scan_key: scanKey,
          scans: scans.map((item) => ({
            exchange: 'moex',
            engine: selectedMarket.engine,
            market: selectedMarket.market,
            board: selectedMarket.board,
            symbol: String(item.symbol ?? ''),
            timeframe: String(item.timeframe ?? ''),
            model_name: String(item.model ?? ''),
            signal: String(item.signal ?? 'none'),
            confidence: String(item.confidence ?? 'none'),
            tradable: true,
            entry: Number(item.entry),
            rr: Number(item.rr),
            market_regime: String(item.regime ?? 'unknown'),
            phase: String(item.phase ?? 'unknown'),
            issues: [],
          })),
        }),
      });
      return response.ok;
    } catch (_error) {
      return false;
    }
  }

  async function persistPortfolioBalance(nextBalance) {
    if (!isAuthenticated) {
      setPortfolioError('Сохранение недоступно: сначала войдите в систему.');
      return false;
    }
    const normalizedBalance = Math.max(0, Number(nextBalance) || 0);
    setIsSavingPortfolio(true);
    setPortfolioError('');
    try {
      const response = await fetch(buildPortfolioUrl(), {
        method: 'PUT',
        headers: buildAuthHeaders(authToken, { 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          user_email: activeUserEmail,
          balance: normalizedBalance,
          currency: portfolioState.currency,
        }),
      });
      if (!response.ok) {
        throw new Error(await parseResponseError(response));
      }
      const payload = await response.json();
      applyPortfolioPayload(payload);
      return true;
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'unknown error';
      setPortfolioError(`Не удалось сохранить портфель: ${reason}`);
      return false;
    } finally {
      setIsSavingPortfolio(false);
    }
  }

  useEffect(() => {
    if (!activeUserEmail) {
      if (!authToken) {
        setUserEmailDraft('');
      }
      return;
    }
    setKnownUserEmails((previous) => Array.from(new Set([...previous, normalizeUserEmail(activeUserEmail)])));
    setUserEmailDraft(normalizeUserEmail(activeUserEmail));
  }, [activeUserEmail, authToken]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const token = String(authToken ?? '').trim();
    if (!token) {
      window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
  }, [authToken]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const normalized = normalizeUserEmail(activeUserEmail);
    if (!normalized) {
      window.localStorage.removeItem(USER_EMAIL_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(USER_EMAIL_STORAGE_KEY, normalized);
  }, [activeUserEmail]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const normalizedList = Array.from(
      new Set(knownUserEmails.map((value) => normalizeUserEmail(value)).filter(Boolean)),
    );
    window.localStorage.setItem(USER_EMAILS_STORAGE_KEY, JSON.stringify(normalizedList));
  }, [knownUserEmails]);

  useEffect(() => {
    const abortController = new AbortController();
    async function hydrateAuth() {
      if (!authToken) {
        return;
      }
      try {
        const email = await fetchAuthMe(authToken, abortController.signal);
        if (abortController.signal.aborted) {
          return;
        }
        setActiveUserEmail(email);
        setKnownUserEmails((previous) => Array.from(new Set([...previous, email])));
        setAuthError('');
      } catch (error) {
        if (abortController.signal.aborted) {
          return;
        }
        const reason = error instanceof Error ? error.message : 'unknown error';
        setAuthToken('');
        setActiveUserEmail('');
        setAuthError(`Сессия недействительна: ${reason}`);
      }
    }
    hydrateAuth();
    return () => abortController.abort();
  }, [authToken]);

  useEffect(() => {
    const abortController = new AbortController();
    async function loadInstruments() {
      setIsLoadingInstruments(true);
      setInstrumentsError('');
      try {
        const response = await fetch(buildMoexInstrumentsUrl(selectedMarket), { signal: abortController.signal });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        const nextInstruments = parseInstruments(payload);
        setInstruments(nextInstruments);
        setSelectedInstrument(nextInstruments[0] ?? null);
      } catch (error) {
        if (abortController.signal.aborted) {
          return;
        }
        setInstruments([]);
        setSelectedInstrument(null);
        setInstrumentsError(`Не удалось загрузить инструменты для ${selectedMarket.label}`);
      } finally {
        if (!abortController.signal.aborted) {
          setIsLoadingInstruments(false);
        }
      }
    }
    loadInstruments();
    return () => abortController.abort();
  }, [selectedMarket]);

  useEffect(() => {
    setBacktestLoadedKey('');
    setRobustLoadedKey('');
    setBacktestData(null);
    setRobustData(null);
  }, [dataKey]);

  useEffect(() => {
    setCandidateScan((previous) => ({
      ...previous,
      long: [],
      short: [],
      error: '',
      scanned: 0,
      total: 0,
      key: '',
    }));
  }, [candidateScopeKey, instruments]);

  useEffect(() => {
    const abortController = new AbortController();
    async function loadSystems() {
      if (!isAuthenticated) {
        setSavedSystems([]);
        setSelectedSystemId('');
        setCurrentSystemId('');
        setIsLoadingPortfolio(false);
        setSystemsError('');
        setPortfolioError('');
        setScanHistory({ scans: [], sessions: [] });
        setScanHistoryError('');
        return;
      }
      setSystemsError("");
      setIsLoadingPortfolio(true);
      setPortfolioError('');
      let systemsLoaded = false;
      try {
        const systemsPayload = await fetchSystems("", abortController.signal);
        systemsLoaded = true;
        await fetchPortfolio(abortController.signal);
        const payloadSystemId = Number(systemsPayload?.current_system_id);
        await fetchScans(abortController.signal, {
          systemId: Number.isInteger(payloadSystemId) && payloadSystemId > 0 ? payloadSystemId : undefined,
        });
      } catch (error) {
        if (abortController.signal.aborted) {
          return;
        }
        const reason = error instanceof Error ? error.message : "unknown error";
        if (systemsLoaded) {
          setPortfolioError(`Не удалось загрузить портфель: ${reason}`);
        } else {
          setSystemsError(`Не удалось загрузить системы: ${reason}`);
        }
      } finally {
        if (!abortController.signal.aborted) {
          setIsLoadingPortfolio(false);
        }
      }
    }
    loadSystems();
    return () => abortController.abort();
  }, [activeUserEmail, isAuthenticated, authToken]);

  useEffect(() => {
    if (!selectedSystemId) {
      return;
    }
    setSavedSystems((previous) =>
      previous.map((system) =>
        system.id === selectedSystemId
          ? {
              ...system,
              config: { ...systemConfig },
              model: selectedModel,
              timeframe: selectedTimeframe,
            }
          : system,
      ),
    );
  }, [selectedSystemId, systemConfig, selectedModel, selectedTimeframe]);

  useEffect(() => {
    if (!selectedInstrument?.symbol) {
      setMarketData(null);
      setMarketError('');
      setIsLoadingMarket(false);
      return undefined;
    }
    const abortController = new AbortController();
    async function loadMarket() {
      setIsLoadingMarket(true);
      setMarketError('');
      const url = buildDashboardMarketUrl({
        ticker: selectedInstrument.symbol,
        exchange: 'moex',
        timeframe: selectedTimeframe,
        engine: selectedMarket.engine,
        market: selectedMarket.market,
        board: selectedMarket.board,
        model: selectedModel,
        deposit: portfolioState.balance,
        limit: systemConfig.marketLimit,
        commissionBps: systemConfig.commissionBps,
        slippageBps: systemConfig.slippageBps,
        patternMinSample: systemConfig.patternMinSample,
      });
      try {
        const response = await fetch(url, { signal: abortController.signal, cache: 'no-store' });
        if (!response.ok) {
          let details = '';
          try {
            const payload = await response.json();
            details = String(payload?.error ?? payload?.message ?? '');
          } catch (_error) {
            details = '';
          }
          throw new Error(`HTTP ${response.status}${details ? `: ${details}` : ''}`);
        }
        const payload = await response.json();
        const parsed = parseMarketResponse(payload);
        if (!parsed) {
          throw new Error('empty market payload');
        }
        setMarketData(parsed);
      } catch (error) {
        if (abortController.signal.aborted) {
          return;
        }
        const reason = error instanceof Error ? error.message : 'unknown error';
        setMarketData(null);
        setMarketError(`Market dashboard недоступен: ${reason}`);
      } finally {
        if (!abortController.signal.aborted) {
          setIsLoadingMarket(false);
        }
      }
    }
    loadMarket();
    return () => abortController.abort();
  }, [selectedInstrument, selectedMarket, selectedTimeframe, selectedModel, systemConfig]);

  useEffect(() => {
    if (selectedScreen !== 'testing' || selectedTestingTab !== 'backtest' || !selectedInstrument?.symbol) {
      return undefined;
    }
    if (backtestLoadedKey === dataKey) {
      return undefined;
    }
    const abortController = new AbortController();
    async function loadBacktest() {
      setIsLoadingBacktest(true);
      setBacktestError('');
      const payload = {
        ...(isAuthenticated ? { user_email: activeUserEmail } : {}),
        ...(isAuthenticated && Number(currentSystemId) > 0 ? { system_id: Number(currentSystemId) } : {}),
        ticker: selectedInstrument.symbol,
        deposit: portfolioState.balance,
        timeframe: selectedTimeframe,
        model: selectedModel,
        exchange: 'moex',
        engine: selectedMarket.engine,
        market: selectedMarket.market,
        board: selectedMarket.board,
        limit: systemConfig.backtestLimit,
        lookback_window: systemConfig.backtestLookbackWindow,
        max_holding_candles: systemConfig.backtestMaxHoldingCandles,
        debug_filters: true,
      };
      try {
        const response = await fetch(buildDashboardBacktestUrl(), {
          method: 'POST',
          headers: buildAuthHeaders(authToken, { 'Content-Type': 'application/json' }),
          signal: abortController.signal,
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          const details = await response.text();
          throw new Error(`HTTP ${response.status}: ${details}`);
        }
        const result = await response.json();
        const parsed = parseBacktestResponse(result);
        if (!parsed) {
          throw new Error('empty backtest payload');
        }
        setBacktestData(parsed);
        setBacktestLoadedKey(dataKey);
      } catch (error) {
        if (abortController.signal.aborted) {
          return;
        }
        const reason = error instanceof Error ? error.message : 'unknown error';
        setBacktestData(null);
        setBacktestError(`Backtest dashboard недоступен: ${reason}`);
      } finally {
        if (!abortController.signal.aborted) {
          setIsLoadingBacktest(false);
        }
      }
    }
    loadBacktest();
    return () => abortController.abort();
  }, [
    selectedScreen,
    selectedTestingTab,
    selectedInstrument,
    selectedMarket,
    selectedTimeframe,
    selectedModel,
    systemConfig,
    isAuthenticated,
    authToken,
    activeUserEmail,
    currentSystemId,
    backtestLoadedKey,
    dataKey,
  ]);

  useEffect(() => {
    if (selectedScreen !== 'testing' || selectedTestingTab !== 'robustness' || !selectedInstrument?.symbol) {
      return undefined;
    }
    if (robustLoadedKey === dataKey) {
      return undefined;
    }
    const abortController = new AbortController();
    async function loadRobustness() {
      setIsLoadingRobust(true);
      setRobustError('');
      const payload = {
        ...(isAuthenticated ? { user_email: activeUserEmail } : {}),
        ...(isAuthenticated && Number(currentSystemId) > 0 ? { system_id: Number(currentSystemId) } : {}),
        ticker: selectedInstrument.symbol,
        deposit: portfolioState.balance,
        timeframe: selectedTimeframe,
        model: selectedModel,
        exchange: 'moex',
        engine: selectedMarket.engine,
        market: selectedMarket.market,
        board: selectedMarket.board,
        limit: systemConfig.robustnessLimit,
        monte_carlo_simulations: systemConfig.monteCarloSimulations,
        train_ratio: systemConfig.trainRatio,
      };
      try {
        const response = await fetch(buildDashboardRobustnessUrl(), {
          method: 'POST',
          headers: buildAuthHeaders(authToken, { 'Content-Type': 'application/json' }),
          signal: abortController.signal,
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          const details = await response.text();
          throw new Error(`HTTP ${response.status}: ${details}`);
        }
        const result = await response.json();
        const parsed = parseRobustnessResponse(result);
        if (!parsed) {
          throw new Error('empty robustness payload');
        }
        setRobustData(parsed);
        setRobustLoadedKey(dataKey);
      } catch (error) {
        if (abortController.signal.aborted) {
          return;
        }
        const reason = error instanceof Error ? error.message : 'unknown error';
        setRobustData(null);
        setRobustError(`Robustness dashboard недоступен: ${reason}`);
      } finally {
        if (!abortController.signal.aborted) {
          setIsLoadingRobust(false);
        }
      }
    }
    loadRobustness();
    return () => abortController.abort();
  }, [
    selectedScreen,
    selectedTestingTab,
    selectedInstrument,
    selectedMarket,
    selectedTimeframe,
    selectedModel,
    systemConfig,
    isAuthenticated,
    authToken,
    activeUserEmail,
    currentSystemId,
    robustLoadedKey,
    dataKey,
  ]);

  function handleToggleMarketLayer(layerKey) {
    setMarketLayers((previous) => {
      if (!Object.prototype.hasOwnProperty.call(previous, layerKey)) {
        return previous;
      }
      return {
        ...previous,
        [layerKey]: !previous[layerKey],
      };
    });
  }

  async function handleScanCandidates() {
    if (isLoadingInstruments || instruments.length === 0) {
      setCandidateScan((previous) => ({
        ...previous,
        error: 'Сначала дождитесь загрузки инструментов',
      }));
      return;
    }
    if (!currentSystemId) {
      setCandidateScan((previous) => ({
        ...previous,
        error: 'Сначала выберите текущую систему на вкладке System',
      }));
      return;
    }

    const requestId = scanRequestRef.current + 1;
    scanRequestRef.current = requestId;
    const scanBatchKey = `scan-${Date.now()}-${currentSystemId}`;
    const pool = instruments.slice(0, Math.max(1, Math.floor(scannerConfig.candidateScanLimit)));
    const scanTargets = pool.flatMap((instrument) =>
      SCANNER_TIMEFRAMES.flatMap((targetTimeframe) =>
        SCANNER_MODELS.map((targetModel) => ({
          instrument,
          timeframe: targetTimeframe,
          model: targetModel,
        })),
      ),
    );

    setCandidateScan({
      isScanning: true,
      error: '',
      scanned: 0,
      total: scanTargets.length,
      long: [],
      short: [],
      lastUpdated: '',
      key: candidateScopeKey,
    });

    const rows = await mapWithConcurrency(
      scanTargets,
      Math.max(1, Math.floor(scannerConfig.candidateScanConcurrency)),
      async (target) => {
        const instrument = target.instrument;
        const response = await fetch(
          buildDashboardMarketUrl({
            ticker: instrument.symbol,
            exchange: 'moex',
            timeframe: target.timeframe,
            engine: selectedMarket.engine,
            market: selectedMarket.market,
            board: selectedMarket.board,
            model: target.model,
            deposit: portfolioState.balance,
            limit: scannerConfig.marketLimit,
            commissionBps: scannerConfig.commissionBps,
            slippageBps: scannerConfig.slippageBps,
            patternMinSample: scannerConfig.patternMinSample,
          }),
          { cache: 'no-store' },
        );
        if (!response.ok) {
          return null;
        }
        const payload = await response.json();
        const signal = String(payload?.signal?.signal ?? 'none').toLowerCase();
        if (signal !== 'long' && signal !== 'short') {
          return null;
        }
        const tradePlan = payload?.trade_plan;
        if (!tradePlan || tradePlan.tradable !== true) {
          return null;
        }
        const confidence = String(payload?.signal?.confidence ?? 'none').toLowerCase();
        return {
          symbol: instrument.symbol,
          name: instrument.name,
          timeframe: target.timeframe,
          model: target.model,
          signal,
          confidence,
          confidenceRank: CONFIDENCE_RANK[confidence] ?? 0,
          rr: Number(payload?.signal?.rr),
          entry: Number(payload?.signal?.entry),
          regime: String(payload?.signal?.market_regime ?? 'unknown'),
          phase: String(payload?.signal?.phase ?? 'unknown'),
          scanKey: `${instrument.symbol}:${target.timeframe}:${target.model}:${signal}`,
        };
      },
      (completed, total) => {
        if (scanRequestRef.current !== requestId) {
          return;
        }
        setCandidateScan((previous) => ({
          ...previous,
          scanned: completed,
          total,
        }));
      },
    );

    if (scanRequestRef.current !== requestId) {
      return;
    }

    const candidates = rows.filter(Boolean);
    candidates.sort((left, right) => {
      if (right.confidenceRank !== left.confidenceRank) {
        return right.confidenceRank - left.confidenceRank;
      }
      const leftRr = Number.isFinite(left.rr) ? left.rr : -1;
      const rightRr = Number.isFinite(right.rr) ? right.rr : -1;
      return rightRr - leftRr;
    });

    setCandidateScan({
      isScanning: false,
      error: '',
      scanned: scanTargets.length,
      total: scanTargets.length,
      long: candidates.filter((item) => item.signal === 'long'),
      short: candidates.filter((item) => item.signal === 'short'),
      lastUpdated: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
      key: candidateScopeKey,
    });

    const persisted = await persistScans({
      scanKey: scanBatchKey,
      systemId: Number(currentSystemId),
      items: candidates,
    });
    if (!persisted) {
      setCandidateScan((previous) => ({
        ...previous,
        error: 'Скан завершен, но не удалось сохранить результаты в историю',
      }));
      return;
    }
    await fetchScans(undefined, { systemId: Number(currentSystemId) });
  }

  function handleSystemConfigChange(key, rawValue) {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) {
      return;
    }
    setSystemConfig((previous) => {
      let nextValue = parsed;
      switch (key) {
        case 'deposit':
          nextValue = Math.max(0, parsed);
          break;
        case 'commissionBps':
        case 'slippageBps':
          nextValue = Math.max(0, parsed);
          break;
        case 'patternMinSample':
        case 'marketLimit':
        case 'candidateScanLimit':
        case 'candidateScanConcurrency':
        case 'backtestLimit':
        case 'backtestLookbackWindow':
        case 'backtestMaxHoldingCandles':
        case 'robustnessLimit':
        case 'monteCarloSimulations':
          nextValue = Math.max(1, Math.floor(parsed));
          break;
        case 'trainRatio':
          nextValue = clamp(parsed, 0.1, 0.95);
          break;
        default:
          break;
      }
      return {
        ...previous,
        [key]: nextValue,
      };
    });
    if (key === 'deposit') {
      const nextBalance = Math.max(0, parsed);
      setPortfolioState((previous) => ({
        ...previous,
        balance: nextBalance,
      }));
      setPortfolioError('');
    }
  }

  function handleResetSystemDefaults() {
    setSystemConfig({ ...SYSTEM_DEFAULTS });
  }

  function handleSelectSystem(systemId) {
    const targetId = String(systemId ?? "").trim();
    if (!targetId) {
      return;
    }
    const system = savedSystems.find((item) => item.id === targetId);
    if (!system) {
      return;
    }
    setSelectedSystemId(system.id);
    setSystemConfig({ ...normalizeSystemConfig(system.config) });
    setSelectedModel(String(system.model ?? DEFAULT_MODEL));
    setSelectedTimeframe(String(system.timeframe ?? "1h"));
  }

  async function handleAddSystem() {
    if (!isAuthenticated) {
      setSystemsError('Войдите в систему, чтобы сохранить новую систему.');
      return;
    }
    const proposedName = window.prompt("Название новой системы", `System ${savedSystems.length + 1}`);
    const name = String(proposedName ?? "").trim();
    if (!name) {
      return;
    }
    const exists = savedSystems.some((system) => system.name.toLowerCase() === name.toLowerCase());
    if (exists) {
      window.alert("Система с таким именем уже существует.");
      return;
    }
    setIsSavingSystem(true);
    setSystemsError('');
    try {
      const response = await fetch(buildSystemsUrl(), {
        method: 'POST',
        headers: buildAuthHeaders(authToken, { 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          user_email: activeUserEmail,
          name,
          model: selectedModel,
          timeframe: selectedTimeframe,
          exchange: 'moex',
          engine: selectedMarket.engine,
          market: selectedMarket.market,
          board: selectedMarket.board,
          config: systemConfig,
          make_current: false,
        }),
      });
      if (!response.ok) {
        throw new Error(await parseResponseError(response));
      }
      const payload = await response.json();
      const createdId = String(payload?.system?.id ?? '').trim();
      await fetchSystems(createdId);
      await fetchScans(undefined, { systemId: Number(createdId) });
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'unknown error';
      setSystemsError(`Не удалось создать систему: ${reason}`);
    } finally {
      setIsSavingSystem(false);
    }
  }

  async function handleSetCurrentSystem() {
    if (!isAuthenticated) {
      setSystemsError('Войдите в систему, чтобы сохранить настройки системы.');
      return;
    }
    const systemId = Number(selectedSystemId);
    if (!Number.isInteger(systemId) || systemId <= 0) {
      return;
    }

    setIsSavingSystem(true);
    setSystemsError('');
    try {
      const updateResponse = await fetch(buildUpdateSystemConfigUrl(systemId), {
        method: 'PUT',
        headers: buildAuthHeaders(authToken, { 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          user_email: activeUserEmail,
          model: selectedModel,
          timeframe: selectedTimeframe,
          config: systemConfig,
        }),
      });
      if (!updateResponse.ok) {
        throw new Error(await parseResponseError(updateResponse));
      }

      const setCurrentResponse = await fetch(buildSetCurrentSystemUrl(), {
        method: 'POST',
        headers: buildAuthHeaders(authToken, { 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          user_email: activeUserEmail,
          system_id: systemId,
        }),
      });
      if (!setCurrentResponse.ok) {
        throw new Error(await parseResponseError(setCurrentResponse));
      }
      await fetchSystems(String(systemId));
      await fetchScans(undefined, { systemId });
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'unknown error';
      setSystemsError(`Не удалось сделать систему текущей: ${reason}`);
    } finally {
      setIsSavingSystem(false);
    }
  }

  async function handleRegister() {
    await submitAuth('register');
  }

  async function handleLogin() {
    await submitAuth('login');
  }

  function handleLogout() {
    setAuthToken('');
    setActiveUserEmail('');
    setUserEmailDraft('');
    setAuthPassword('');
    setAuthError('');
    setSystemsError('');
    setPortfolioError('');
    setSelectedScreen('portfolio');
  }

  function openAuthModal() {
    setAuthError('');
    setIsAuthModalOpen(true);
  }

  function closeAuthModal() {
    if (!isAuthPending) {
      setIsAuthModalOpen(false);
    }
  }

  function handlePortfolioAmountChange(rawValue) {
    handleSystemConfigChange("deposit", rawValue);
  }

  async function handlePortfolioAmountCommit() {
    if (!isAuthenticated) {
      setPortfolioError('Сохранение недоступно: сначала войдите в систему.');
      return;
    }
    await persistPortfolioBalance(portfolioState.balance);
  }

  async function handleSystemConfigCommit(key) {
    if (key !== 'deposit') {
      return;
    }
    if (!isAuthenticated) {
      setPortfolioError('Сохранение недоступно: сначала войдите в систему.');
      return;
    }
    await persistPortfolioBalance(portfolioState.balance);
  }

  const isSystemScreen = isAuthenticated && selectedScreen === 'system';
  const isPortfolioScreen = isAuthenticated && selectedScreen === 'portfolio';
  const showInstrumentsPanel = !isSystemScreen && !isPortfolioScreen;
  const layoutClassName = [
    'layout',
    isSystemScreen ? 'layout-system' : '',
    !showInstrumentsPanel ? 'layout-full' : '',
  ]
    .filter(Boolean)
    .join(' ');

  useEffect(() => {
    if (!isAuthModalOpen) {
      return undefined;
    }
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        closeAuthModal();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isAuthModalOpen, isAuthPending]);

  return (
    <main className="page">
      <nav className="topMenu" aria-label="Main navigation">
        <div className="topMenuTabs">
          {isAuthenticated ? (
            SCREEN_OPTIONS.map((screen) => (
              <button
                key={screen.key}
                type="button"
                className={`topMenuTab ${selectedScreen === screen.key ? 'active' : ''}`}
                onClick={() => setSelectedScreen(screen.key)}
              >
                {screen.label}
              </button>
            ))
          ) : (
            <span className="guestTopBrand">Trading System</span>
          )}
        </div>
        <div className="topMenuUserControls">
          {isAuthenticated ? (
            <>
              <span className="topMenuUserLabel">User</span>
              <span className="topMenuUserCurrent">{activeUserEmail}</span>
              <button type="button" className="scanButton" onClick={handleLogout} title="Выйти из системы">
                Logout
              </button>
            </>
          ) : (
            <>
              <span className="topMenuUserLabel">Guest</span>
              <button type="button" className="scanButton" onClick={openAuthModal}>
                Вход
              </button>
            </>
          )}
        </div>
      </nav>
      {!isAuthenticated ? (
        <section className="panel guestPanel">
          <GuestMarketingScreen />
        </section>
      ) : (
        <section className={layoutClassName}>
        {showInstrumentsPanel ? (
          <aside className="instruments">
            <h2>Instruments</h2>
            <div className="instrumentToolbar">
              <label htmlFor="market-select">Рынок</label>
              <select id="market-select" value={selectedMarketKey} onChange={(event) => setSelectedMarketKey(event.target.value)}>
                {MARKET_OPTIONS.map((market) => (
                  <option key={market.key} value={market.key}>
                    {market.label}
                  </option>
                ))}
              </select>
            </div>
            <ul className="instrumentList">
              {isLoadingInstruments ? <li className="instrumentStatus">Загрузка инструментов...</li> : null}
              {!isLoadingInstruments && instrumentsError ? <li className="instrumentStatus error">{instrumentsError}</li> : null}
              {!isLoadingInstruments && !instrumentsError && instruments.length === 0 ? (
                <li className="instrumentStatus">Список инструментов пуст</li>
              ) : null}
              {!isLoadingInstruments &&
                !instrumentsError &&
                instruments.map((instrument) => {
                  const isActive = selectedInstrument ? instrument.symbol === selectedInstrument.symbol : false;
                  const changeClass = instrument.change.startsWith('+')
                    ? 'instrumentChange positive'
                    : instrument.change.startsWith('-')
                      ? 'instrumentChange negative'
                      : 'instrumentChange neutral';
                  return (
                    <li key={instrument.symbol}>
                      <button
                        className={`instrumentButton ${isActive ? 'active' : ''}`}
                        onClick={() => setSelectedInstrument(instrument)}
                        type="button"
                      >
                        <span className="instrumentSymbol">{instrument.symbol}</span>
                        <span className="instrumentName">{instrument.name}</span>
                        <span className="instrumentMeta">
                          <span>{instrument.price}</span>
                          <span className={changeClass}>{instrument.change}</span>
                        </span>
                      </button>
                    </li>
                  );
                })}
            </ul>
          </aside>
        ) : null}

        <section className="panel">
          <div className="panelHeader">
            {isSystemScreen ? (
              <>
                <h2>System Settings</h2>
                <p>Конфигурация движка сигналов и тестирования</p>
              </>
            ) : isPortfolioScreen ? (
              <>
                <h2>Portfolio & Scan</h2>
                <p>Сканирование для пользователя {activeUserEmail} / система {currentSystemName || '—'}</p>
              </>
            ) : (
              <div className="panelHeaderRow">
                <div className="panelIdentity">
                  <span className="panelShortName">{selectedInstrument?.symbol ?? '—'}</span>
                  <span className="panelFullName">{selectedInstrument?.name ?? 'Инструмент не выбран'}</span>
                </div>
                <div className="panelControls">
                  <div className="panelControlField">
                    <label htmlFor="timeframe-select">Timeframe</label>
                    <select id="timeframe-select" value={selectedTimeframe} onChange={(event) => setSelectedTimeframe(event.target.value)}>
                      {TIMEFRAME_OPTIONS.map((timeframe) => (
                        <option key={timeframe} value={timeframe}>
                          {timeframe}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="panelControlField">
                    <label htmlFor="model-select">Model</label>
                    <select id="model-select" value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                      {MODEL_OPTIONS.map((modelName) => (
                        <option key={modelName} value={modelName}>
                          {modelName}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}
          </div>

          {selectedScreen === 'portfolio' ? (
            <PortfolioSignalsScreen
              currentSystemName={currentSystemName}
              scanTimeframes={SCANNER_TIMEFRAMES}
              scanModels={SCANNER_MODELS}
              candidateLimit={scannerConfig.candidateScanLimit}
              portfolioAmount={portfolioState.balance}
              onPortfolioAmountChange={handlePortfolioAmountChange}
              onPortfolioAmountCommit={handlePortfolioAmountCommit}
              isSavingPortfolio={isSavingPortfolio || isLoadingPortfolio}
              portfolioError={portfolioError}
              candidateScan={candidateScan}
              onScanCandidates={handleScanCandidates}
              scanHistory={scanHistory}
              isLoadingScanHistory={isLoadingScanHistory}
              scanHistoryError={scanHistoryError}
            />
          ) : null}
          {selectedScreen === 'analysis' ? (
            <TechnicalAnalysisScreen
              marketData={marketData}
              isLoading={isLoadingMarket}
              error={marketError}
              timeframe={selectedTimeframe}
              layers={marketLayers}
              onToggleLayer={handleToggleMarketLayer}
            />
          ) : null}
          {selectedScreen === 'testing' ? (
            <SystemTestingScreen
              selectedTab={selectedTestingTab}
              onSelectTab={setSelectedTestingTab}
              backtestData={backtestData}
              isLoadingBacktest={isLoadingBacktest}
              backtestError={backtestError}
              robustData={robustData}
              isLoadingRobust={isLoadingRobust}
              robustError={robustError}
            />
          ) : null}
          {selectedScreen === 'system' ? (
            <SystemSettingsScreen
              config={systemConfig}
              systems={savedSystems}
              selectedSystemId={selectedSystemId}
              currentSystemId={currentSystemId}
              isAuthenticated={isAuthenticated}
              onSelectSystem={handleSelectSystem}
              onAddSystem={handleAddSystem}
              onSetCurrent={handleSetCurrentSystem}
              isSavingSystem={isSavingSystem}
              systemsError={systemsError}
              onConfigChange={handleSystemConfigChange}
              onConfigCommit={handleSystemConfigCommit}
              onResetDefaults={handleResetSystemDefaults}
            />
          ) : null}
        </section>
      </section>
      )}
      {!isAuthenticated ? (
        <AuthModal
          isOpen={isAuthModalOpen}
          email={userEmailDraft}
          password={authPassword}
          onEmailChange={setUserEmailDraft}
          onPasswordChange={setAuthPassword}
          onLogin={handleLogin}
          onRegister={handleRegister}
          onClose={closeAuthModal}
          isAuthPending={isAuthPending}
          authError={authError}
        />
      ) : null}
    </main>
  );
}
