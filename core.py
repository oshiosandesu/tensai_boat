# core.py
# 天才ボートくん：オッズ解析コア
# - 3連複/3連単オッズ取得（pyjpboatraceラッパー＋フェイルセーフ）
# - 3複→3単分解（順序バイアス・レーン別重み）
# - 市場確率 q とモデル確率 p、ブレンド確率 p*（遊びモード）
# - EVモード（勝ち向け）の候補抽出・配分
# - ヒット率モード（遊び向け）のK点抽出／目標的中率K算定

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import itertools
import math
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime

# ---- pyjpboatrace の有無を確認（無い時はデモフェイルセーフに切替） ----
try:
    import pyjpboatrace as pjb  # 実環境で利用
except Exception:
    pjb = None


# ===== ボートレース場マスタ =====
VENUE_ID2NAME: Dict[int, str] = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}
VENUE_NAME2ID: Dict[str, int] = {v: k for k, v in VENUE_ID2NAME.items()}


# ===== データ構造 =====
@dataclass
class Runner:
    lane: int
    name: str = ""
    reg_id: Optional[int] = None
    klass: str = ""     # A1/A2/B1/B2
    branch: str = ""
    age: Optional[int] = None
    weight: Optional[float] = None
    avg_st_all: Optional[float] = None
    avg_st_meet: Optional[float] = None
    winrate_all: Optional[float] = None
    winrate_local: Optional[float] = None
    winrate_recent: Optional[float] = None
    motor_no: Optional[int] = None
    motor_2r: Optional[float] = None
    boat_no: Optional[int] = None
    boat_2r: Optional[float] = None
    recent_results: List[Dict] = field(default_factory=list)
    exhibit_time: Optional[float] = None
    tilt: Optional[float] = None
    parts_change: str = ""


@dataclass
class TrioOdds:
    comb: Tuple[int, int, int]  # 昇順(例: (1,2,3))
    odds: float


@dataclass
class TrifectaOdds:
    comb: Tuple[int, int, int]  # 順序あり(例: (1,2,3))
    odds: float


@dataclass
class Snapshot:
    date: str            # YYYYMMDD
    venue_id: int
    race_no: int
    taken_at: datetime
    weather: Dict
    entries: Dict[int, Runner]
    odds_trio: List[TrioOdds]
    odds_trifecta: List[TrifectaOdds]
    meta: Dict


@dataclass
class ModelParams:
    # 3複→確率化
    alpha_3f: float = 1.0
    # 順序バイアス（レーン別重み）：内有利を軽く表現（デフォルトは控えめ）
    head_lane_weights: np.ndarray = field(default_factory=lambda: np.array([1.00, 0.96, 0.92, 0.88, 0.84, 0.80]))
    mid_lane_weights:  np.ndarray = field(default_factory=lambda: np.array([1.00, 0.98, 0.96, 0.94, 0.92, 0.90]))
    tail_lane_weights: np.ndarray = field(default_factory=lambda: np.array([1.00, 0.99, 0.98, 0.97, 0.96, 0.95]))
    # 遊びモードのブレンド（p* = w*p + (1-w)*q）
    blend_w: float = 0.6
    # EV縮小（スリッページ・不確実性）
    slippage_pct: float = 0.01
    lambda_uncertainty: float = 0.03
    # フィルタ
    ev_threshold: float = 0.03
    gap_threshold: float = 0.005
    max_points: int = 10
    max_pair_per_head: int = 3


@dataclass
class Candidate:
    comb: Tuple[int, int, int]
    odds: float
    p: float
    q: float
    pstar: float
    edge: float
    ev: float
    ev_shrunk: float
    reason_tags: List[str] = field(default_factory=list)


@dataclass
class Allocation:
    comb: Tuple[int, int, int]
    odds: float
    stake: int            # 最小単位ベット数
    stake_cash: float     # 金額（円想定）
    kelly_fraction: float # 理論ケリー(縮小後)の素値


# ===== ユーティリティ =====
def _all_trifecta_perms() -> List[Tuple[int, int, int]]:
    return list(itertools.permutations([1, 2, 3, 4, 5, 6], 3))


def _all_trio_combs() -> List[Tuple[int, int, int]]:
    return list(itertools.combinations([1, 2, 3, 4, 5, 6], 3))


PERMS_3T: List[Tuple[int, int, int]] = _all_trifecta_perms()
COMBS_3F: List[Tuple[int, int, int]] = _all_trio_combs()


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _normalize_weights(vals: np.ndarray) -> np.ndarray:
    s = np.nansum(vals)
    if not np.isfinite(s) or s <= 0:
        return np.full_like(vals, 1.0 / len(vals))
    return vals / s


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log(p)))


# ===== pyjpboatrace ラッパ（あいまいAPIに対応／失敗時はデモ用乱数） =====
def _simulate_odds_trifecta() -> Dict[Tuple[int, int, int], float]:
    # ランダムディリクレ分布→確率→オッズ生成（デモ用）
    probs = np.random.dirichlet(alpha=np.ones(len(PERMS_3T)) * 1.2)
    # オッズは 1/q に係数（還元率80%想定で係数1.2〜1.3付近）
    coef = 1.20 + random.random() * 0.15
    return {perm: max(1.01, coef / float(probs[i])) for i, perm in enumerate(PERMS_3T)}


def _simulate_odds_trio() -> Dict[Tuple[int, int, int], float]:
    probs = np.random.dirichlet(alpha=np.ones(len(COMBS_3F)) * 1.3)
    coef = 1.15 + random.random() * 0.15
    return {comb: max(1.01, coef / float(probs[i])) for i, comb in enumerate(COMBS_3F)}


def get_trifecta_odds(date: str, venue_id: int, race_no: int, allow_sim_fallback: bool = True) -> Dict[Tuple[int, int, int], float]:
    """
    3連単オッズ取得。辞書 {(a,b,c): odds} を返す。
    """
    # 実API（pyjpboatrace）の試行
    if pjb is not None:
        try:
            # 代表的な呼び出しパターンを試行（パッケージ版差異を吸収）
            # 例：pyjpboatrace.get_trifecta_odds(date, jcd, race_no) のようなもの
            # 下記は仮API。実環境の関数名に合わせて使われます（try/except で吸収）
            if hasattr(pjb, "get_trifecta_odds"):
                data = pjb.get_trifecta_odds(date=date, jcd=venue_id, rno=race_no)
            elif hasattr(pjb, "OfficialAPI"):
                api = pjb.OfficialAPI()
                data = api.odds_trifecta(date=date, jcd=venue_id, rno=race_no)
            else:
                data = None

            odds_map = {}
            if data:
                # data の形に応じて解釈（例：list[dict] or dict）
                if isinstance(data, dict):
                    iterator = data.items()
                else:
                    iterator = data

                for row in iterator:
                    # 代表的フィールド名を総当たり
                    if isinstance(row, tuple) and len(row) == 2 and isinstance(row[0], tuple):
                        comb, odds = row
                        comb = tuple(int(x) for x in comb)
                        odds_map[comb] = _safe_float(odds, None)
                    elif isinstance(row, dict):
                        a = int(row.get("first") or row.get("a") or row.get("i") or 0)
                        b = int(row.get("second") or row.get("b") or row.get("j") or 0)
                        c = int(row.get("third") or row.get("c") or row.get("k") or 0)
                        o = _safe_float(row.get("odds") or row.get("value") or row.get("o"), None)
                        if a and b and c and o:
                            odds_map[(a, b, c)] = o
                if odds_map:
                    return odds_map
        except Exception:
            pass

    # フォールバック（デモ）
    if allow_sim_fallback:
        return _simulate_odds_trifecta()
    return {}


def get_trio_odds(date: str, venue_id: int, race_no: int, allow_sim_fallback: bool = True) -> Dict[Tuple[int, int, int], float]:
    """
    3連複オッズ取得。辞書 {(i,j,k): odds}（i<j<k）を返す。
    """
    if pjb is not None:
        try:
            if hasattr(pjb, "get_trio_odds"):
                data = pjb.get_trio_odds(date=date, jcd=venue_id, rno=race_no)
            elif hasattr(pjb, "OfficialAPI"):
                api = pjb.OfficialAPI()
                data = api.odds_trio(date=date, jcd=venue_id, rno=race_no)
            else:
                data = None

            odds_map = {}
            if data:
                if isinstance(data, dict):
                    iterator = data.items()
                else:
                    iterator = data
                for row in iterator:
                    if isinstance(row, tuple) and len(row) == 2 and isinstance(row[0], tuple):
                        comb, odds = row
                        comb = tuple(sorted(int(x) for x in comb))
                        odds_map[comb] = _safe_float(odds, None)
                    elif isinstance(row, dict):
                        a = int(row.get("first") or row.get("a") or row.get("i") or 0)
                        b = int(row.get("second") or row.get("b") or row.get("j") or 0)
                        c = int(row.get("third") or row.get("c") or row.get("k") or 0)
                        tup = tuple(sorted([a, b, c]))
                        o = _safe_float(row.get("odds") or row.get("value") or row.get("o"), None)
                        if all(tup) and o:
                            odds_map[tup] = o
                if odds_map:
                    return odds_map
        except Exception:
            pass

    if allow_sim_fallback:
        return _simulate_odds_trio()
    return {}


def get_race_context(date: str, venue_id: int, race_no: int) -> Dict:
    """
    天候・風・波・水質・発走予定など。取得できない場合は空で返す。
    """
    ctx = {}
    if pjb is not None:
        try:
            if hasattr(pjb, "get_race_info"):
                data = pjb.get_race_info(date=date, jcd=venue_id, rno=race_no)
            elif hasattr(pjb, "OfficialAPI"):
                api = pjb.OfficialAPI()
                data = api.race_info(date=date, jcd=venue_id, rno=race_no)
            else:
                data = None
            if isinstance(data, dict):
                ctx.update(data)
        except Exception:
            pass
    return ctx


def get_entries(date: str, venue_id: int, race_no: int) -> Dict[int, Runner]:
    """
    出走表（選手/モーター/展示など）。取れないフィールドは空で埋める。
    """
    runners: Dict[int, Runner] = {i: Runner(lane=i) for i in range(1, 7)}
    if pjb is not None:
        try:
            # 仮API（実環境依存）
            data = None
            if hasattr(pjb, "get_entries"):
                data = pjb.get_entries(date=date, jcd=venue_id, rno=race_no)
            elif hasattr(pjb, "OfficialAPI"):
                api = pjb.OfficialAPI()
                data = api.entries(date=date, jcd=venue_id, rno=race_no)

            if data:
                # dataが list[dict] を想定。無ければ安全にスキップ
                for row in data:
                    lane = int(row.get("lane") or row.get("w") or 0)
                    if lane in runners:
                        r = runners[lane]
                        r.name = str(row.get("name") or "")
                        r.reg_id = _safe_float(row.get("reg_id") or row.get("id"), None)
                        r.klass = str(row.get("class") or row.get("grade") or "")
                        r.branch = str(row.get("branch") or "")
                        r.age = _safe_float(row.get("age"), None)
                        r.weight = _safe_float(row.get("weight"), None)
                        r.avg_st_all = _safe_float(row.get("avg_st_all") or row.get("st_avg"), None)
                        r.avg_st_meet = _safe_float(row.get("avg_st_meet") or row.get("st_meet"), None)
                        r.winrate_all = _safe_float(row.get("winrate_all") or row.get("win_all"), None)
                        r.winrate_local = _safe_float(row.get("winrate_local") or row.get("win_local"), None)
                        r.winrate_recent = _safe_float(row.get("winrate_recent") or row.get("win_recent"), None)
                        r.motor_no = int(row.get("motor_no") or row.get("engine") or 0) or None
                        r.motor_2r = _safe_float(row.get("motor_2r") or row.get("engine_2r"), None)
                        r.boat_no = int(row.get("boat_no") or 0) or None
                        r.boat_2r = _safe_float(row.get("boat_2r"), None)
                        r.exhibit_time = _safe_float(row.get("exhibit_time") or row.get("ex"), None)
                        r.tilt = _safe_float(row.get("tilt"), None)
                        r.parts_change = str(row.get("parts_change") or "")
                        # 直近走
                        rr = row.get("recent_results")
                        if isinstance(rr, list):
                            r.recent_results = rr
        except Exception:
            pass
    return runners


# ===== スナップショット作成 =====
def fetch_snapshot(date: str, venue_id: int, race_no: int) -> Snapshot:
    odds3t = get_trifecta_odds(date, venue_id, race_no, allow_sim_fallback=True)
    odds3f = get_trio_odds(date, venue_id, race_no, allow_sim_fallback=True)
    info = get_race_context(date, venue_id, race_no)
    entries = get_entries(date, venue_id, race_no)

    # 市場確率 q（3単）からエントロピー、ブック合算などを計算
    q_vec = []
    for perm in PERMS_3T:
        o = odds3t.get(perm)
        if o and o > 0:
            q_vec.append(1.0 / o)
        else:
            q_vec.append(0.0)
    q_vec = np.array(q_vec)
    q_norm = _normalize_weights(q_vec)
    entropy_3t = _entropy(q_norm)
    book_sum = float(np.sum(q_norm))  # 正規化後は1.0になるが、未正規化合算も別途出せる

    # 3複Top5カバレッジ（alpha=1で一旦計算、UIで表示）
    if odds3f:
        inv = np.array([1.0 / max(1.0e-9, odds3f[c]) for c in COMBS_3F])
        p3f = _normalize_weights(inv)
        top5 = float(np.sum(np.sort(p3f)[::-1][:5]))
    else:
        top5 = 0.0

    snap = Snapshot(
        date=date,
        venue_id=venue_id,
        race_no=race_no,
        taken_at=datetime.now(),
        weather={
            "raw": info or {},
        },
        entries=entries,
        odds_trio=[TrioOdds(c, float(odds3f[c])) for c in odds3f.keys()],
        odds_trifecta=[TrifectaOdds(c, float(odds3t[c])) for c in odds3t.keys()],
        meta={
            "entropy_3t": entropy_3t,
            "book_sum": float(np.sum(1.0 / np.maximum(1.0e-9, q_vec))) if len(q_vec) else None,  # 未正規合算
            "coverage_top5_3f": top5,
        },
    )
    return snap


# ===== 確率化・分解 =====
def normalize_trio(odds_trio: List[TrioOdds], alpha: float = 1.0) -> Dict[Tuple[int, int, int], float]:
    """
    3連複オッズを確率化: p ∝ (1/odds)^alpha
    """
    if not odds_trio:
        return {c: 1.0 / len(COMBS_3F) for c in COMBS_3F}
    vals = np.array([(1.0 / max(1.0e-9, x.odds)) ** alpha for x in odds_trio])
    vals = _normalize_weights(vals)
    return {odds_trio[i].comb: float(vals[i]) for i in range(len(odds_trio))}


def normalize_trifecta(odds_trifecta: List[TrifectaOdds]) -> Dict[Tuple[int, int, int], float]:
    """
    3連単オッズから市場確率 q を作成: q ∝ 1/odds
    """
    if not odds_trifecta:
        return {p: 1.0 / len(PERMS_3T) for p in PERMS_3T}
    vals = np.array([1.0 / max(1.0e-9, x.odds) for x in odds_trifecta])
    vals = _normalize_weights(vals)
    return {odds_trifecta[i].comb: float(vals[i]) for i in range(len(odds_trifecta))}


def decompose_trio_to_trifecta(p3f: Dict[Tuple[int, int, int], float],
                               head_w: np.ndarray,
                               mid_w: np.ndarray,
                               tail_w: np.ndarray) -> Dict[Tuple[int, int, int], float]:
    """
    3複確率 p3f を順序バイアスで 3単 p3t に分配。
    - head_w, mid_w, tail_w は 6要素（レーン1..6）の重み
    """
    p3t = {perm: 0.0 for perm in PERMS_3T}
    for comb, prob in p3f.items():
        a, b, c = comb
        perms = list(itertools.permutations(comb, 3))
        w = []
        for (h, m, t) in perms:
            w.append(head_w[h - 1] * mid_w[m - 1] * tail_w[t - 1])
        w = np.array(w, dtype=float)
        w = _normalize_weights(w)  # 6個の重みを正規化
        for idx, (h, m, t) in enumerate(perms):
            p3t[(h, m, t)] += float(prob * w[idx])
    # 正規化（念のため）
    s = sum(p3t.values())
    if s > 0:
        for k in p3t:
            p3t[k] /= s
    else:
        p3t = {perm: 1.0 / len(PERMS_3T) for perm in PERMS_3T}
    return p3t


def build_probabilities(snapshot: Snapshot, params: ModelParams) -> Dict[str, Dict[Tuple[int, int, int], float]]:
    """
    p（モデル：3複→3単分解）、q（市場：3単オッズ）、p*（遊び向けブレンド）を作る。
    """
    p3f = normalize_trio(snapshot.odds_trio, alpha=params.alpha_3f)
    p3t = decompose_trio_to_trifecta(
        p3f,
        head_w=params.head_lane_weights,
        mid_w=params.mid_lane_weights,
        tail_w=params.tail_lane_weights,
    )
    q3t = normalize_trifecta(snapshot.odds_trifecta)
    # 遊び向けブレンド
    pstar = {}
    w = float(params.blend_w)
    for perm in PERMS_3T:
        p = p3t.get(perm, 0.0)
        q = q3t.get(perm, 0.0)
        pstar[perm] = w * p + (1.0 - w) * q
    # 3複側のブレンドも用意しておく（遊びモード3複用）
    q3f = aggregate_to_trio(q3t)
    pstar3f = {}
    p3f_norm = _normalize_weights(np.array([p3f.get(c, 0.0) for c in COMBS_3F]))
    for i, comb in enumerate(COMBS_3F):
        p = float(p3f_norm[i])
        q = float(q3f.get(comb, 0.0))
        pstar3f[comb] = w * p + (1.0 - w) * q

    return {"p3t": p3t, "q3t": q3t, "pstar3t": pstar, "p3f": p3f, "q3f": q3f, "pstar3f": pstar3f}


def aggregate_to_trio(prob3t: Dict[Tuple[int, int, int], float]) -> Dict[Tuple[int, int, int], float]:
    """
    3単確率から 3複（順序無視）へ集約。
    """
    out = {c: 0.0 for c in COMBS_3F}
    for (a, b, c), val in prob3t.items():
        comb = tuple(sorted((a, b, c)))
        out[comb] += float(val)
    # 正規化
    s = sum(out.values())
    if s > 0:
        for k in out:
            out[k] /= s
    return out


# ===== EVモード（勝ち向け） =====
def _shrink_for_risk(p: float, odds: float, lam: float, slip: float) -> Tuple[float, float]:
    """
    EV縮小：p と odds を少し保守的に縮める
    """
    p2 = max(0.0, min(1.0, p * (1.0 - lam)))
    odds2 = max(1.0, odds * (1.0 - slip))
    return p2, odds2


def _kelly_fraction(p: float, odds: float) -> float:
    """
    ケリー素値 f* = (b*p - (1-p)) / b = (odds*p - 1) / (odds - 1)
    （odds はデシマル、b = odds - 1）
    """
    b = max(1e-9, odds - 1.0)
    f = (odds * p - 1.0) / b
    return max(0.0, float(f))


def build_ev_candidates(snapshot: Snapshot,
                        probs: Dict[str, Dict[Tuple[int, int, int], float]],
                        params: ModelParams) -> List[Candidate]:
    odds_map = {x.comb: x.odds for x in snapshot.odds_trifecta}
    p3t, q3t = probs["p3t"], probs["q3t"]
    out: List[Candidate] = []
    for perm in PERMS_3T:
        odds = odds_map.get(perm)
        if not odds or odds <= 1.0:
            continue
        p = p3t.get(perm, 0.0)
        q = q3t.get(perm, 0.0)
        p_s, odds_s = _shrink_for_risk(p, odds, params.lambda_uncertainty, params.slippage_pct)
        ev = odds * p - 1.0
        ev_s = odds_s * p_s - 1.0
        edge = p - q
        if ev_s > params.ev_threshold and edge > params.gap_threshold:
            tags = []
            if edge > 0.02:
                tags.append("過小評価（p>q）")
            if ev_s > 0.1:
                tags.append("高EV")
            out.append(Candidate(perm, float(odds), float(p), float(q), float(probs["pstar3t"].get(perm, 0.0)),
                                 float(edge), float(ev), float(ev_s), reason_tags=tags))
    # トリミング：同一(頭-2着)上限／点数上限
    out.sort(key=lambda c: (c.ev_shrunk, c.ev), reverse=True)
    trimmed: List[Candidate] = []
    head2_count: Dict[Tuple[int, int], int] = {}
    for c in out:
        key = (c.comb[0], c.comb[1])
        if head2_count.get(key, 0) >= params.max_pair_per_head:
            continue
        trimmed.append(c)
        head2_count[key] = head2_count.get(key, 0) + 1
        if len(trimmed) >= params.max_points:
            break
    return trimmed


# ===== ヒット率モード（遊び向け） =====
def pick_topK_by_prob(prob_map: Dict[Tuple[int, int, int], float], K: int) -> Tuple[List[Tuple[int, int, int]], float]:
    items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    chosen = items[:max(0, K)]
    hitrate = float(sum(v for _, v in chosen))
    return [k for k, _ in chosen], hitrate


def pick_by_target_hit(prob_map: Dict[Tuple[int, int, int], float], target: float) -> Tuple[List[Tuple[int, int, int]], float]:
    items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    s = 0.0
    chosen = []
    for k, v in items:
        chosen.append(k)
        s += v
        if s >= target:
            break
    return chosen, float(s)


# ===== 配分 =====
def allocate_budget(candidates: List[Candidate],
                    race_budget: int,
                    min_unit: int = 100,
                    kelly_fraction_scale: float = 0.5) -> List[Allocation]:
    """
    半ケリー（デフォルト0.5）で配分→上限/最小単位で丸め。
    """
    if race_budget <= 0 or not candidates:
        return []
    # ケリー素値（縮小後p,oddsで計算）
    kelly_vals = []
    for c in candidates:
        p_s, odds_s = _shrink_for_risk(c.p, c.odds, 0.0, 0.0)  # ここでは縮小せず素のp/oddsでも可
        f = _kelly_fraction(p_s, odds_s)
        kelly_vals.append(max(0.0, f))
    arr = np.array(kelly_vals, dtype=float)
    if np.all(arr == 0):
        # 全て0なら等分（最低限）
        base = np.ones(len(candidates))
    else:
        base = arr
    weights = base / np.sum(base)
    raw = weights * float(race_budget)
    # 最小単位で丸め
    stakes = np.floor(raw / float(min_unit)) * float(min_unit)
    # 端数を配る
    remain = race_budget - int(np.sum(stakes))
    i = 0
    while remain >= min_unit and i < len(stakes):
        stakes[i] += float(min_unit)
        remain -= min_unit
        i = (i + 1) % len(stakes)
    out: List[Allocation] = []
    for i, c in enumerate(candidates):
        out.append(Allocation(
            comb=c.comb,
            odds=c.odds,
            stake=int(stakes[i]),
            stake_cash=float(stakes[i]),
            kelly_fraction=float(arr[i] * kelly_fraction_scale),
        ))
    return out


# ===== 周辺確率（頭・連対・含有、モデル/市場の比較用） =====
def marginal_probs_head(prob3t: Dict[Tuple[int, int, int], float]) -> Dict[int, float]:
    head = {i: 0.0 for i in range(1, 7)}
    for (a, b, c), v in prob3t.items():
        head[a] += float(v)
    return head


def marginal_probs_contend(prob3t: Dict[Tuple[int, int, int], float]) -> Dict[int, float]:
    cont = {i: 0.0 for i in range(1, 7)}
    for (a, b, c), v in prob3t.items():
        cont[a] += float(v)
        cont[b] += float(v)
    # 正規化（合計2になるので1に合わせるなら/2）
    s = sum(cont.values())
    if s > 0:
        for k in cont:
            cont[k] /= 2.0
    return cont


def marginal_probs_include(prob3t: Dict[Tuple[int, int, int], float]) -> Dict[int, float]:
    inc = {i: 0.0 for i in range(1, 7)}
    for (a, b, c), v in prob3t.items():
        inc[a] += float(v)
        inc[b] += float(v)
        inc[c] += float(v)
    # 合計3になるので/3で1に合わせる
    s = sum(inc.values())
    if s > 0:
        for k in inc:
            inc[k] /= 3.0
    return inc


# ===== テーブル作成ヘルパ =====
def candidates_to_frame(cands: List[Candidate]) -> pd.DataFrame:
    rows = []
    for c in cands:
        rows.append({
            "組番(3単)": f"{c.comb[0]}-{c.comb[1]}-{c.comb[2]}",
            "odds": round(c.odds, 2),
            "p(モデル)": round(c.p, 4),
            "q(市場)": round(c.q, 4),
            "Edge(p-q)": round(c.edge, 4),
            "EV": round(c.ev, 4),
            "EV'": round(c.ev_shrunk, 4),
            "理由": ", ".join(c.reason_tags) if c.reason_tags else "",
        })
    return pd.DataFrame(rows)


def allocations_to_frame(allocs: List[Allocation]) -> pd.DataFrame:
    rows = []
    for a in allocs:
        rows.append({
            "組番(3単)": f"{a.comb[0]}-{a.comb[1]}-{a.comb[2]}",
            "odds": round(a.odds, 2),
            "ベット額": int(a.stake),
            "Kelly素値(縮小前)": round(a.kelly_fraction, 4),
        })
    return pd.DataFrame(rows)
