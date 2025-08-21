# -*- coding: utf-8 -*-
"""
core.py
共通ロジック：公式オッズの取得、3連複→確率化、指標算出、候補生成、EV判定、配分、
マーケット過熱度(head1_mkt, pair_overbet)と“紐荒れ”検出(value_ratio)。
"""

import warnings
from datetime import date
from itertools import permutations
from collections import defaultdict
from functools import lru_cache

# LibreSSL ノイズ警告の抑制（動作に影響なし）
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

# 公式データAPI
from pyjpboatrace import PyJPBoatrace  # get_odds_trio / get_odds_trifecta / get_race_result / get_stadiums

# 場コード（jcd=01..24）
VENUES = [
    (1, "桐生"), (2, "戸田"), (3, "江戸川"), (4, "平和島"), (5, "多摩川"), (6, "浜名湖"),
    (7, "蒲郡"), (8, "常滑"), (9, "津"), (10, "三国"), (11, "びわこ"), (12, "住之江"),
    (13, "尼崎"), (14, "鳴門"), (15, "丸亀"), (16, "児島"), (17, "宮島"), (18, "徳山"),
    (19, "下関"), (20, "若松"), (21, "芦屋"), (22, "福岡"), (23, "唐津"), (24, "大村"),
]
VENUE_ID2NAME = {vid: name for vid, name in VENUES}
VENUE_NAME2ID = {name: vid for vid, name in VENUES}

# ===== 内枠寄りの順序バイアス（3複→3単分解で使用） =====
HEAD_BIAS = {1: 0.60, 2: 0.18, 3: 0.12, 4: 0.07, 5: 0.02, 6: 0.01}
MID_BIAS  = {1: 1.00, 2: 0.95, 3: 0.90, 4: 0.85, 5: 0.80, 6: 0.75}
TAIL_BIAS = {1: 1.00, 2: 0.92, 3: 0.88, 4: 0.84, 5: 0.80, 6: 0.76}

# ===== 公式オッズ/結果 取得 =====

def get_trio_odds(d: date, stadium: int, rno: int):
    """
    3連複オッズ {frozenset({a,b,c}): odds} と更新タグを返す
    """
    api = PyJPBoatrace()
    raw = api.get_odds_trio(d, stadium, rno)  # {'1=2=3': 5.1, ..., 'update': '19:15'}
    trio = {}
    update_tag = raw.get("update", "")
    for k, v in raw.items():
        if "=" in k:
            try:
                a, b, c = map(int, k.split("="))
                trio[frozenset({a, b, c})] = float(v)
            except Exception:
                continue
    return trio, update_tag

def get_trifecta_odds(d: date, stadium: int, rno: int):
    """
    3連単オッズ {(a,b,c): odds}
    """
    api = PyJPBoatrace()
    raw = api.get_odds_trifecta(d, stadium, rno)  # {'1-2-3': 12.1, ..., 'update': '19:15'}
    tri = {}
    for k, v in raw.items():
        if "-" in k:
            try:
                a, b, c = map(int, k.split("-"))
                tri[(a, b, c)] = float(v)
            except Exception:
                continue
    return tri

def get_result_and_payout(d: date, stadium: int, rno: int):
    """
    確定結果（3連単）と払戻(100円あたり)
    戻り: (results:list[str], payouts:dict[str,int])
    """
    api = PyJPBoatrace()
    raw = api.get_race_result(d, stadium, rno)
    payouts = {}
    results = []
    if not isinstance(raw, dict):
        return results, payouts

    # 同着も網羅
    if "trifecta_all" in raw and isinstance(raw["trifecta_all"], list) and raw["trifecta_all"]:
        for item in raw["trifecta_all"]:
            res = str(item.get("result", "")).strip()
            if not res:
                continue
            results.append(res)
            pay = item.get("payoff", 0)
            try:
                pay = int(str(pay).replace(",", ""))
            except Exception:
                pay = int("".join(ch for ch in str(pay) if ch.isdigit()) or "0")
            payouts[res] = pay
    elif "trifecta" in raw and isinstance(raw["trifecta"], dict):
        item = raw["trifecta"]
        res = str(item.get("result", "")).strip()
        if res:
            results.append(res)
            pay = item.get("payoff", 0)
            try:
                pay = int(str(pay).replace(",", ""))
            except Exception:
                pay = int("".join(ch for ch in str(pay) if ch.isdigit()) or "0")
            payouts[res] = pay

    return results, payouts

# ===== 3複→確率化・指標 =====

def normalize_probs_from_odds(odds_items, top_n=None):
    """
    odds_items: list[(key, odds)] 低オッズ(人気)順
    上位top_nで 1/odds を合計1に正規化して擬似確率に
    """
    items = sorted(odds_items, key=lambda x: x[1])
    if top_n is not None:
        items = items[:top_n]
    q = [(k, 1.0/x if x > 0 else 0.0) for k, x in items]
    s = sum(v for _, v in q) or 1.0
    return {k: v/s for k, v in q}, items  # pmap, 人気順items

def top5_coverage(pmap):
    vals = sorted(pmap.values(), reverse=True)
    return sum(vals[:5])

def inclusion_mass_for_boat(pmap, boat):
    return sum(p for S, p in pmap.items() if boat in S)

def pair_mass(pmap):
    pm = defaultdict(float)
    for S, p in pmap.items():
        a, b, c = sorted(list(S))
        pm[(a, b)] += p; pm[(a, c)] += p; pm[(b, c)] += p
    return dict(pm)

# ===== 高速化：順序重みをキャッシュ =====

@lru_cache(maxsize=None)
def _simple_order_weights_cached(a, b, c):
    """
    {a,b,c}の全順序に対し、内枠補正の重みを正規化して返す
    """
    lst = []
    for i, j, k in permutations((a, b, c), 3):
        w = HEAD_BIAS[i] * MID_BIAS[j] * TAIL_BIAS[k]
        lst.append(((i, j, k), w))
    tot = sum(w for _, w in lst)
    return tuple((o, w/tot) for o, w in lst)

def simple_order_weights(S):
    """後方互換のため残す（内部はキャッシュ版）。"""
    a, b, c = sorted(list(S))
    return list(_simple_order_weights_cached(a, b, c))

def estimate_head_rate(pmap, head=1):
    pr = 0.0
    for S, pS in pmap.items():
        if head not in S:
            continue
        a, b, c = sorted(list(S))
        for (o, w) in _simple_order_weights_cached(a, b, c):
            if o[0] == head:
                pr += pS * w
    return pr

def choose_R_by_coverage(pmap):
    cov5 = top5_coverage(pmap)
    if cov5 >= 0.65:
        return 4, f"堅め（Top5合計={cov5:.1%}）"
    elif cov5 <= 0.55:
        return 6, f"流動（Top5合計={cov5:.1%}）"
    else:
        return 5, f"中庸（Top5合計={cov5:.1%}）"

def coverage_targets(pmap, targets=(0.25, 0.50, 0.75)):
    items = sorted(pmap.items(), key=lambda x: x[1], reverse=True)
    out = {}
    for t in targets:
        acc, pick = 0.0, []
        for S, p in items:
            pick.append((S, p))
            acc += p
            if acc >= t:
                out[t] = (len(pick), pick.copy()); break
        if t not in out:
            out[t] = (len(pick), pick.copy())
    return out

# ===== 候補生成・EV・配分 =====

def build_trifecta_candidates(pmap, R, avoid_top=True, max_per_set=2):
    top_sets = sorted(pmap.items(), key=lambda x: x[1], reverse=True)[:R]
    out = []
    for S, pS in top_sets:
        a, b, c = sorted(list(S))
        ow = sorted(_simple_order_weights_cached(a, b, c), key=lambda x: x[1], reverse=True)
        idx0 = 1 if (avoid_top and len(ow) > 1) else 0
        take = 0
        for (o, w) in ow[idx0:]:
            out.append((o, pS * w, S))
            take += 1
            if take >= max_per_set:
                break
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def add_pair_hedge_if_needed(cands, pmap, top_pairs, max_extra=1):
    exist = set()
    for (o, _, _) in cands:
        i, j = sorted((o[0], o[1]))
        exist.add((i, j))
    added = []
    for (i, j), _mass in top_pairs:
        if (i, j) in exist:
            continue
        bestS, bestp = None, -1.0
        for S, pS in pmap.items():
            if i in S and j in S and pS > bestp:
                bestS, bestp = S, pS
        if not bestS:
            continue
        a, b, c = sorted(list(bestS))
        ow = sorted(_simple_order_weights_cached(a, b, c), key=lambda x: x[1], reverse=True)
        start = 1 if len(ow) > 1 else 0
        o, w = ow[start]
        added.append((o, pmap[bestS] * w, bestS))
        if len(added) >= max_extra:
            break
    return cands + added

def ev_of(o, p_est, trifecta_odds, margin=0.10):
    odds = trifecta_odds.get(o)
    if odds is None or odds <= 0:
        return None, None, None, False
    req = (1.0 / odds) * (1.0 + margin)
    ev  = p_est * odds - 1.0
    ok  = (p_est > req)
    return odds, req, ev, ok

def allocate_budget_by_prob(cands, budget, min_unit=100):
    psum = sum(p for _, p, _ in cands)
    if psum <= 0:
        return [(o, p, S, 0) for (o, p, S) in cands], 0
    raw = [(o, p, S, budget * (p/psum)) for (o, p, S) in cands]
    bets, total = [], 0
    for (o, p, S, v) in raw:
        b = int(v // min_unit) * min_unit
        bets.append((o, p, S, b)); total += b
    rem = budget - total
    idx = 0
    while rem >= min_unit and idx < len(bets):
        o, p, S, b = bets[idx]
        bets[idx] = (o, p, S, b + min_unit)
        rem -= min_unit
        idx += 1
        if idx == len(bets): idx = 0
    return bets, budget - rem

# ===== マーケット過熱度 =====

def head_market_rate(trifecta_odds, head=1):
    """
    3連単オッズから暗黙の head頭 率（全点の1/oddsを正規化し、head==1の合計）
    """
    qs = []
    for (a,b,c), odd in trifecta_odds.items():
        if odd and odd > 0:
            qs.append((a, 1.0/odd))
    s = sum(q for _, q in qs) or 1.0
    return sum(q for a, q in qs if a == head) / s

def pair_overbet_ratio(pair, pmap, trifecta_odds, beta=1.0):
    """
    マーケット上の(頭=pair[0],2着=pair[1]) と (逆順) の暗黙合計を、3複ペア質量×β で割った比
    >1 なら過熱気味
    """
    i, j = sorted(pair)
    pm = pair_mass(pmap)
    mass = pm.get((i,j), 0.0) * beta
    q = 0.0
    denom = 0.0
    for (a,b,c), odd in trifecta_odds.items():
        if odd and odd > 0:
            val = 1.0/odd
            denom += val
            if (a, b) == (i, j) or (a, b) == (j, i):
                q += val
    if denom <= 0:
        return 0.0
    mkt = q / denom
    if mass <= 0:
        return float('inf') if mkt > 0 else 0.0
    return mkt / mass

# ===== “紐荒れ”検出（特定ペアの3着分布の歪み） =====

def value_ratios_for_pair(head, second, pmap, trifecta_odds):
    """
    指定ペア(head,second)に対する 3着k ごとの
    - 期待p（3複Top10→順序重み）と
    - マーケット暗黙p（3単オッズ→1/odds正規化）
    の比= mkt/exp を返す。低いほど“妙味”。
    戻り: list[(k, p_exp, p_mkt, ratio, odds_k)]
    """
    # 期待p
    exp_raw = defaultdict(float)
    exp_total = 0.0
    for S, pS in pmap.items():
        if head in S and second in S:
            a, b, c = sorted(list(S))
            for (o, w) in _simple_order_weights_cached(a, b, c):
                if o[0] == head and o[1] == second:
                    k = o[2]
                    exp_raw[k] += pS * w
                    exp_total += pS * w
    exp_dist = {}
    if exp_total > 0:
        for k, v in exp_raw.items():
            exp_dist[k] = v / exp_total

    # マーケット暗黙p
    mkt_raw = {}
    denom = 0.0
    for (a,b,c), odd in trifecta_odds.items():
        if odd and odd > 0:
            val = 1.0/odd
            denom += val
            if a == head and b == second:
                mkt_raw[c] = mkt_raw.get(c, 0.0) + val
    mkt_dist = {}
    if denom > 0:
        sub = sum(mkt_raw.values()) or 1.0
        for k, v in mkt_raw.items():
            mkt_dist[k] = v / sub

    keys = set(exp_dist.keys()) | set(mkt_dist.keys())
    out = []
    for k in sorted(keys):
        p_exp = exp_dist.get(k, 0.0)
        p_mkt = mkt_dist.get(k, 0.0)
        ratio = (p_mkt / p_exp) if p_exp > 0 else (0.0 if p_mkt == 0 else float('inf'))
        odds_k = trifecta_odds.get((head, second, k))
        out.append((k, p_exp, p_mkt, ratio, odds_k))
    out.sort(key=lambda x: (x[3], - (x[1] if x[1] else 0.0)))  # ratio昇順→期待p高い
    return out
