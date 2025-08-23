# -*- coding: utf-8 -*-
"""
core.py — 競艇AIコア（安定版・オッズのみ）
- pyjpboatrace から 3連複/3連単オッズと結果を取得
- 3連複TopN→確率化→軽い順序バイアスで3連単へ分解（モデル確率 p）
- 市場確率 q は 1/オッズ を正規化
- EV判定（マージン一律）と資金配分（半ケリー＋上限）
- 入口の指標（堅さ/1号艇含有/推定頭率/市場頭率/ペア強度/カバレッジ）を提供
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict, Counter
from datetime import date
from functools import lru_cache

# 会場マスタ
VENUES: List[Tuple[int, str]] = [
    (1, "桐生"), (2, "戸田"), (3, "江戸川"), (4, "平和島"),
    (5, "多摩川"), (6, "浜名湖"), (7, "蒲郡"), (8, "常滑"),
    (9, "津"), (10, "三国"), (11, "びわこ"), (12, "住之江"),
    (13, "尼崎"), (14, "鳴門"), (15, "丸亀"), (16, "児島"),
    (17, "宮島"), (18, "徳山"), (19, "下関"), (20, "若松"),
    (21, "芦屋"), (22, "福岡"), (23, "唐津"), (24, "大村"),
]
VENUE_ID2NAME = {vid: name for vid, name in VENUES}

# 型
TrioSet = frozenset            # frozenset({a,b,c}) をキーに
Order3 = Tuple[int, int, int]  # 3連単の並び (i,j,k)

# 内枠寄りの軽い順序バイアス
HEAD_BIAS = {1: 1.35, 2: 1.05, 3: 0.95, 4: 0.85, 5: 0.80, 6: 0.75}
MID_BIAS  = {1: 1.20, 2: 1.05, 3: 1.00, 4: 0.95, 5: 0.90, 6: 0.85}
TAIL_BIAS = {1: 1.10, 2: 1.05, 3: 1.02, 4: 1.00, 5: 0.98, 6: 0.96}

# データ取得クライアント
try:
    from pyjpboatrace import PyJPBoatrace  # v0.4系
except Exception as e:
    PyJPBoatrace = None

_client_singleton: Optional["PyJPBoatrace"] = None
def _client() -> "PyJPBoatrace":
    global _client_singleton
    if PyJPBoatrace is None:
        raise RuntimeError("pyjpboatrace が利用できません。requirements を確認してください。")
    if _client_singleton is None:
        _client_singleton = PyJPBoatrace()
    return _client_singleton

# ---------- 取得系 ----------
def get_trio_odds(d: date, venue_id: int, rno: int) -> Tuple[Dict[TrioSet, float], Optional[str]]:
    """
    三連複オッズ → {frozenset({a,b,c}): 倍率}, update_tag
    """
    cli = _client()
    raw = cli.get_odds_trio(d, venue_id, rno)  # 例: {'1=2=3': 2.1, 'update': '締切時オッズ'}
    update = raw.get("update")
    odds: Dict[TrioSet, float] = {}
    for k, v in raw.items():
        if isinstance(k, str) and k.count("=") == 2:
            try:
                a, b, c = (int(x) for x in k.split("="))
                odds[frozenset((a, b, c))] = float(v)
            except Exception:
                pass
    return odds, update

def get_trifecta_odds(d: date, venue_id: int, rno: int) -> Dict[Order3, float]:
    """
    三連単オッズ → {(a,b,c): 倍率}
    """
    cli = _client()
    raw = cli.get_odds_trifecta(d, venue_id, rno)  # 例: {'1-2-3': 12.1, 'update': '...'}
    odds: Dict[Order3, float] = {}
    for k, v in raw.items():
        if isinstance(k, str) and k.count("-") == 2:
            try:
                a, b, c = (int(x) for x in k.split("-"))
                odds[(a, b, c)] = float(v)
            except Exception:
                pass
    return odds

def get_trifecta_result(d: date, venue_id: int, rno: int) -> Optional[Tuple[Order3, float]]:
    """
    確定三連単結果 ((a,b,c), オッズ=払戻/100) を返す。未確定は None。
    """
    try:
        cli = _client()
        res = cli.get_race_result(d, venue_id, rno)
        payoff = res.get("payoff") or {}
        tri = payoff.get("trifecta") or res.get("trifecta")
        if not tri:
            return None
        s = str(tri.get("result", "")).strip()
        if s.count("-") != 2:
            return None
        a, b, c = (int(x) for x in s.split("-"))
        yen = float(tri.get("payoff", 0.0))
        return ((a, b, c), yen / 100.0)
    except Exception:
        return None

# ---------- 開催検知 ----------
@lru_cache(maxsize=128)
def is_race_available(d: date, venue_id: int, rno: int = 1) -> bool:
    try:
        trio1, _ = get_trio_odds(d, venue_id, rno)
        if trio1:
            return True
        trio6, _ = get_trio_odds(d, venue_id, 6)
        return bool(trio6)
    except Exception:
        return False

@lru_cache(maxsize=32)
def venues_on(d: date) -> List[int]:
    active = []
    for vid, _ in VENUES:
        if is_race_available(d, vid, 1):
            active.append(vid)
    return active

# ---------- 3複TopN → 確率化 ----------
def top5_coverage(pmap: Dict[TrioSet, float]) -> float:
    return sum(sorted(pmap.values(), reverse=True)[:5])

def _choose_alpha_by_cov5(cov5: float) -> float:
    if cov5 >= 0.75:
        return 1.20
    elif cov5 >= 0.65:
        return 1.10
    elif cov5 >= 0.55:
        return 1.00
    else:
        return 0.90

def normalize_probs_from_odds(
    trio_sorted_items: List[Tuple[TrioSet, float]],
    top_n: int = 10,
    alpha: float = 1.0,
) -> Tuple[Dict[TrioSet, float], List[Tuple[TrioSet, float]]]:
    items = trio_sorted_items[:top_n]
    ws = []
    for S, o in items:
        w = (1.0 / o) ** alpha if (o and o > 0) else 0.0
        ws.append((S, w))
    den = sum(w for _, w in ws) or 1.0
    pmap = {S: (w / den) for S, w in ws}
    return pmap, items

def normalize_with_dynamic_alpha(
    trio_sorted_items: List[Tuple[TrioSet, float]],
    top_n: int = 10
) -> Tuple[Dict[TrioSet, float], List[Tuple[TrioSet, float]], float, float]:
    p_base, items = normalize_probs_from_odds(trio_sorted_items, top_n=top_n, alpha=1.0)
    cov5 = top5_coverage(p_base)
    alpha = _choose_alpha_by_cov5(cov5)
    pmap, items = normalize_probs_from_odds(trio_sorted_items, top_n=top_n, alpha=alpha)
    return pmap, items, alpha, cov5

# ---------- 指標 ----------
def inclusion_mass_for_boat(pmap: Dict[TrioSet, float], boat: int) -> float:
    return sum(p for S, p in pmap.items() if boat in S)

def pair_mass(pmap: Dict[TrioSet, float]) -> Dict[Tuple[int, int], float]:
    out = defaultdict(float)
    for S, p in pmap.items():
        a, b, c = sorted(list(S))
        for i, j in ((a, b), (a, c), (b, c)):
            out[(i, j)] += p
    return out

def _perm_weight(order: Order3) -> float:
    i, j, k = order
    return HEAD_BIAS[i] * MID_BIAS[j] * TAIL_BIAS[k]

def _split_to_orders(S: TrioSet, pS: float) -> List[Tuple[Order3, float]]:
    a, b, c = list(S)
    perms: List[Order3] = [(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)]
    ws = [_perm_weight(o) for o in perms]
    z = sum(ws) or 1.0
    return [(perms[t], pS * ws[t] / z) for t in range(6)]

def estimate_head_rate(pmap: Dict[TrioSet, float], head: int = 1) -> float:
    acc = 0.0
    for S, pS in pmap.items():
        if head not in S:
            continue
        for o, px in _split_to_orders(S, pS):
            if o[0] == head:
                acc += px
    return acc

# ---------- 市場確率 ----------
def market_prob_trifecta(tri_odds: Dict[Order3, float]) -> Dict[Order3, float]:
    w = {}
    s = 0.0
    for o, odds in tri_odds.items():
        if odds and odds > 0:
            r = 1.0 / odds
            w[o] = r
            s += r
    if s <= 0:
        return {o: 0.0 for o in tri_odds.keys()}
    return {o: (x / s) for o, x in w.items()}

def market_head_rate(tri_odds: Dict[Order3, float], head: int) -> float:
    q = market_prob_trifecta(tri_odds)
    tot = 0.0
    for j in range(1, 7):
        if j == head: 
            continue
        for k in range(1, 7):
            if k in (head, j):
                continue
            tot += q.get((head, j, k), 0.0)
    return tot

# ---------- モデル確率（3複→3単） ----------
def model_prob_trifecta(pmap: Dict[TrioSet, float]) -> Dict[Order3, float]:
    out = defaultdict(float)
    for S, pS in pmap.items():
        for o, px in _split_to_orders(S, pS):
            out[o] += px
    return dict(out)

# ---------- カバレッジ里程 ----------
def coverage_milestones(pmap: Dict[TrioSet, float]) -> Dict[str, List[Tuple[Tuple[int,int,int], float]]]:
    """
    3複上位から 25/50/75% に到達するまで、対応する3単代表（最大重みの順序）を列挙。
    """
    items = sorted(pmap.items(), key=lambda x: x[1], reverse=True)
    out = {"25": [], "50": [], "75": []}
    acc = 0.0
    for S, pS in items:
        # 代表順序（最大重み）
        orders = _split_to_orders(S, pS)
        rep = max(orders, key=lambda t: t[1])[0]
        acc += pS
        if acc <= 0.25:
            out["25"].append((rep, pS))
        if acc <= 0.50:
            out["50"].append((rep, pS))
        if acc <= 0.75:
            out["75"].append((rep, pS))
        if acc >= 0.75:
            break
    return out

# ---------- R決定・候補生成 ----------
def choose_R_by_coverage(pmap: Dict[TrioSet, float]) -> Tuple[int, str]:
    cov5 = top5_coverage(pmap)
    if cov5 >= 0.75:
        return 5, f"堅め（Top5={cov5:.1%}）"
    elif cov5 >= 0.65:
        return 6, f"やや本命（Top5={cov5:.1%}）"
    elif cov5 >= 0.55:
        return 7, f"中庸（Top5={cov5:.1%}）"
    else:
        return 8, f"やや荒れ（Top5={cov5:.1%}）"

def build_trifecta_candidates(
    pmap: Dict[TrioSet, float],
    R: int,
    avoid_top: bool = False,
    max_per_set: Optional[int] = None,
    cov5_hint: Optional[float] = None,
) -> List[Tuple[Order3, float, TrioSet]]:
    if max_per_set is None:
        max_per_set = 3 if (cov5_hint is not None and cov5_hint >= 0.75) else 2
    items = sorted(pmap.items(), key=lambda x: x[1], reverse=True)[:R]
    out: List[Tuple[Order3, float, TrioSet]] = []
    for S, pS in items:
        perms = _split_to_orders(S, pS)
        perms_sorted = sorted(perms, key=lambda t: t[1], reverse=True)
        picked = []
        skipped = False
        for o, pe in perms_sorted:
            if avoid_top and not skipped:
                skipped = True
                continue
            picked.append((o, pe, S))
            if len(picked) >= max_per_set:
                break
        out.extend(picked)
    return sorted(out, key=lambda x: x[1], reverse=True)

# ---------- EV・トリミング・配分 ----------
def evaluate_candidates_basic(
    cands: List[Tuple[Order3, float, TrioSet]],
    tri_odds: Dict[Order3, float],
    base_margin: float,
) -> List[Tuple[Order3, float, TrioSet, Optional[float], Optional[float], Optional[float], bool]]:
    """
    返り値: [(order, p_est, from_set, odds, req_p, ev, ok)]
    """
    out = []
    for o, p_est, S in cands:
        odds = tri_odds.get(o)
        if (odds is None) or (odds <= 0):
            out.append((o, p_est, S, None, None, None, False))
            continue
        req = (1.0 + base_margin) / odds
        ev = p_est * odds - 1.0
        ok = (p_est >= req)
        out.append((o, p_est, S, odds, req, ev, ok))
    return out

def trim_candidates(
    rows: List[Tuple[Order3, float, TrioSet, Optional[float], Optional[float], Optional[float], bool]],
    max_points: int = 8,
    max_same_pair_points: int = 2
) -> List[Tuple[Order3, float, TrioSet, float, float, float, bool]]:
    ok_rows = [r for r in rows if (r[6] is True) and (r[3] is not None)]
    ok_sorted = sorted(ok_rows, key=lambda x: (x[5], x[1] * (x[3] or 0.0)), reverse=True)
    pair_count = Counter()
    out = []
    for r in ok_sorted:
        o = r[0]
        pair = (o[0], o[1])
        if pair_count[pair] >= max_same_pair_points:
            continue
        out.append((o, r[1], r[2], float(r[3]), float(r[4] or 0.0), float(r[5] or 0.0), True))
        pair_count[pair] += 1
        if len(out) >= max_points:
            break
    return out

def half_kelly_fraction(p_est: float, odds: float) -> float:
    if (odds is None) or (odds <= 1.0):
        return 0.0
    b = odds - 1.0
    f = p_est - (1.0 - p_est) / b
    return max(0.0, 0.5 * f)

def allocate_budget(
    buys: List[Tuple[Order3, float, TrioSet, float, float, float, bool]],
    race_cap: int,
    min_unit: int = 100,
    per_bet_cap_ratio: float = 0.40
) -> Tuple[List[Tuple[Order3, float, TrioSet, int, float]], int]:
    """
    返り値: [(order, p_est, from_set, bet_yen, odds)], used_total
    """
    if race_cap <= 0 or not buys:
        return [], 0

    wishes = []
    for (o, p_est, S, odds, _req, _ev, _ok) in buys:
        f = half_kelly_fraction(p_est, odds)
        amt = f * race_cap
        wishes.append((o, p_est, S, odds, amt))

    tot = sum(max(0.0, w[4]) for w in wishes)
    per_cap = race_cap * per_bet_cap_ratio
    prelim = []
    if tot <= 0:
        # 最低限1単位ずつ
        units = max(1, race_cap // min_unit // max(1, len(wishes)))
        for o, p, S, od, _ in wishes:
            prelim.append((o, p, S, od, units))
    else:
        for o, p, S, od, amt in wishes:
            units = int(round(min(amt, per_cap) / min_unit))
            prelim.append((o, p, S, od, units))

    target_units = race_cap // min_unit
    cur_units = sum(u for *_, u in prelim)
    if cur_units == 0:
        prelim = [(o, p, S, od, 1) for (o, p, S, od, u) in prelim]
        cur_units = len(prelim)

    # 超過/不足の調整
    while cur_units > target_units and prelim:
        idx = min(range(len(prelim)), key=lambda i: prelim[i][1] if prelim[i][4] > 0 else 1e9)
        o, p, S, od, u = prelim[idx]
        if u > 0:
            prelim[idx] = (o, p, S, od, u - 1)
            cur_units -= 1
        else:
            break
    while cur_units < target_units:
        idx = max(range(len(prelim)), key=lambda i: prelim[i][1])
        o, p, S, od, u = prelim[idx]
        prelim[idx] = (o, p, S, od, u + 1)
        cur_units += 1

    out = [(o, p, S, u * min_unit, od) for (o, p, S, od, u) in prelim if u > 0]
    used = sum(x[3] for x in out)
    return out, used
