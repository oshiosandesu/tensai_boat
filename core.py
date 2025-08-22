# -*- coding: utf-8 -*-
"""
core.py
- 公式（ボートレース）由来のオッズ取得（pyjpboatrace）
- 3連複→（順序分解）→3連単の候補生成
- EVチェック、点数絞り（6〜8点推奨）、同一ペアの集中抑制
- 資金配分（縮小ケリー相当の確率按分 / 最小単位考慮）
- 指標（固さ・1号艇含有・1着見込み・ペア質量 etc）
- バックテスト用の結果取得（実配当が取れない場合はEVフォールバック）

※ 展示データは“表示用途”を想定し、ロジックにはまだ小さくしか入れていません
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, Optional, Set
from collections import defaultdict, Counter
from datetime import date

# --- 外部 ---
try:
    from pyjpboatrace import PyJPBoatrace  # v0.4.2
except Exception as e:
    PyJPBoatrace = None

# --- 型エイリアス ---
TrioSet = frozenset  # frozenset({a,b,c})
Order3 = Tuple[int, int, int]

# --- 会場テーブル（ID ↔ 名称）---
VENUES: List[Tuple[int, str]] = [
    (1, "桐生"), (2, "戸田"), (3, "江戸川"), (4, "平和島"),
    (5, "多摩川"), (6, "浜名湖"), (7, "蒲郡"), (8, "常滑"),
    (9, "津"), (10, "三国"), (11, "びわこ"), (12, "住之江"),
    (13, "尼崎"), (14, "鳴門"), (15, "丸亀"), (16, "児島"),
    (17, "宮島"), (18, "徳山"), (19, "下関"), (20, "若松"),
    (21, "芦屋"), (22, "福岡"), (23, "唐津"), (24, "大村"),
]
VENUE_ID2NAME = {vid: name for vid, name in VENUES}

# --- 順序分解用バイアス（内枠寄り）---
# 1着/2着/3着の“なりやすさ”の相対重み。必要に応じてチューニング可。
HEAD_BIAS = {1: 1.35, 2: 1.05, 3: 0.95, 4: 0.85, 5: 0.80, 6: 0.75}
MID_BIAS  = {1: 1.20, 2: 1.05, 3: 1.00, 4: 0.95, 5: 0.90, 6: 0.85}
TAIL_BIAS = {1: 1.10, 2: 1.05, 3: 1.02, 4: 1.00, 5: 0.98, 6: 0.96}

# --- 取得ラッパ ---
def _client() -> PyJPBoatrace:
    if PyJPBoatrace is None:
        raise RuntimeError("pyjpboatrace が利用できません。requirements とインポートをご確認ください。")
    return PyJPBoatrace()

def get_trio_odds(d: date, venue_id: int, rno: int):
    """
    3連複オッズ: {frozenset({a,b,c}): odds} と updateタグ(文字列) を返す
    """
    cli = _client()
    data = cli.get_trio_odds(d, venue_id, rno)  # 想定API（v0.4系）
    odds_map: Dict[TrioSet, float] = {}
    update_tag = None
    for row in data.get("odds", []):
        comb = row["comb"]  # 例: [1,2,3]
        o = row["odds"]
        odds_map[frozenset(comb)] = float(o)
    update_tag = data.get("update")
    return odds_map, update_tag

def get_trifecta_odds(d: date, venue_id: int, rno: int):
    """
    3連単オッズ: {(i,j,k): odds}
    """
    cli = _client()
    data = cli.get_trifecta_odds(d, venue_id, rno)
    odds_map: Dict[Order3, float] = {}
    for row in data.get("odds", []):
        order = tuple(row["comb"])  # [1,2,3]
        o = row["odds"]
        odds_map[(int(order[0]), int(order[1]), int(order[2]))] = float(o)
    return odds_map

def get_just_before_info(d: date, venue_id: int, rno: int) -> Optional[dict]:
    """
    直前情報（展示タイム等）。無ければ None。
    """
    try:
        cli = _client()
        info = cli.get_just_before_info(d, venue_id, rno)
        return info
    except Exception:
        return None

def get_trifecta_result(d: date, venue_id: int, rno: int) -> Optional[Tuple[Order3, float]]:
    """
    レース結果（3連単 的中並び と 100円あたり配当倍率）。
    取得できなければ None を返す。
    """
    try:
        cli = _client()
        result = cli.get_result(d, venue_id, rno)  # 想定API: { "trifecta": {"order":[1,3,4], "payout": 3340} }
        tri = result.get("trifecta", {})
        if not tri:
            return None
        order = tuple(int(x) for x in tri["order"])
        payout_yen = float(tri["payout"])  # 100円あたりの払戻金
        odds = payout_yen / 100.0          # 倍率へ変換
        return (order, odds)
    except Exception:
        return None

# --- 確率化（TopN・α導入可）---
def normalize_probs_from_odds(trio_sorted_items: List[Tuple[TrioSet, float]], top_n: int = 10, alpha: float = 1.0):
    """
    trio_sorted_items : (S, odds) をオッズ昇順で並べたもの
    返り値: pmap_topN({S: p}), topN_items(list)
    """
    items = trio_sorted_items[:top_n]
    weights = []
    for S, o in items:
        if o and o > 0:
            weights.append((S, (1.0 / o) ** alpha))
        else:
            weights.append((S, 0.0))
    denom = sum(w for _, w in weights) or 1.0
    pmap = {S: (w / denom) for S, w in weights}
    return pmap, items

# --- 各種指標 ---
def top5_coverage(pmap: Dict[TrioSet, float]) -> float:
    vals = sorted(pmap.values(), reverse=True)[:5]
    return sum(vals)

def inclusion_mass_for_boat(pmap: Dict[TrioSet, float], boat: int) -> float:
    return sum(p for S, p in pmap.items() if boat in S)

def pair_mass(pmap: Dict[TrioSet, float]) -> Dict[Tuple[int, int], float]:
    out = defaultdict(float)
    for S, p in pmap.items():
        a, b, c = sorted(list(S))
        pairs = [(a, b), (a, c), (b, c)]
        for i, j in pairs:
            out[(i, j)] += p
    return out

def _perm_weights(order: Order3) -> float:
    i, j, k = order
    return HEAD_BIAS[i] * MID_BIAS[j] * TAIL_BIAS[k]

def _split_to_orders(S: TrioSet, pS: float) -> List[Tuple[Order3, float]]:
    """
    S={a,b,c} の確率 pS を順序へ分配（重み比例）。
    """
    a, b, c = list(S)
    perms = [(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)]
    ws = [(_perm_weights(o)) for o in perms]
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

def choose_R_by_coverage(pmap: Dict[TrioSet, float]) -> Tuple[int, str]:
    """
    レース“固さ”から R（採用セット数）を決める雑把なルール。
    """
    cov5 = top5_coverage(pmap)
    if cov5 >= 0.75:
        return 4, f"堅め（Top5={cov5:.1%}）"
    elif cov5 >= 0.65:
        return 5, f"やや本命（Top5={cov5:.1%}）"
    elif cov5 >= 0.55:
        return 6, f"中庸（Top5={cov5:.1%}）"
    else:
        return 7, f"やや荒れ（Top5={cov5:.1%}）"

def coverage_targets(pmap: Dict[TrioSet, float], targets=(0.25,0.50,0.75)):
    items = sorted(pmap.items(), key=lambda x: x[1], reverse=True)
    out = {}
    for t in targets:
        acc, k = 0.0, 0
        for i, (S, p) in enumerate(items, start=1):
            acc += p
            if acc >= t:
                k = i
                break
        out[t] = (k, items)
    return out

# --- 候補生成・補強 ---
def build_trifecta_candidates(
    pmap: Dict[TrioSet, float],
    R: int,
    avoid_top: bool = True,
    max_per_set: int = 2
) -> List[Tuple[Order3, float, TrioSet]]:
    """
    上位Rセットから、順序分解で各セット max_per_set 点を返す。
    avoid_top=True の場合、各セットで最も“ありがち”な順序を1つスキップ。
    """
    items = sorted(pmap.items(), key=lambda x: x[1], reverse=True)[:R]
    candidates: List[Tuple[Order3, float, TrioSet]] = []
    for S, pS in items:
        perm = _split_to_orders(S, pS)  # [(order, p_est), ...] 6通り
        # ありがち順序 = 重み順トップ
        perm_sorted = sorted(perm, key=lambda x: _perm_weights(x[0]), reverse=True)
        pick_list = []
        for (o, pe) in perm_sorted:
            if avoid_top and len(pick_list) == 0:
                # 1番“ありがち”はスキップ
                avoid_top = True
                # ただしスキップするだけで、以降は採用
                pass
            else:
                pick_list.append((o, pe, S))
            if len(pick_list) >= max_per_set:
                break
        candidates.extend(pick_list)
    # 全候補を“p_est×オッズなし”で降順（素の当たりやすさ重視）
    return sorted(candidates, key=lambda x: x[1], reverse=True)

def add_pair_hedge_if_needed(
    cands: List[Tuple[Order3, float, TrioSet]],
    pmap: Dict[TrioSet, float],
    top_pairs: List[Tuple[Tuple[int,int], float]],
    max_extra: int = 1
) -> List[Tuple[Order3, float, TrioSet]]:
    """
    上位ペア（頭-2着の順序いずれか）が一つも含まれていないとき、1点だけ追加。
    """
    if not top_pairs:
        return cands
    need = []
    present_pairs = {(o[0], o[1]) for (o, _, _) in cands} | {(o[1], o[0]) for (o, _, _) in cands}
    for (i, j), _m in top_pairs:
        if (i, j) not in present_pairs and (j, i) not in present_pairs:
            need.append((i, j))
    if not need:
        return cands
    # 最上位セットから、そのペアを満たす順序を1つ作る
    items = sorted(pmap.items(), key=lambda x: x[1], reverse=True)
    extra = []
    for (i, j) in need[:max_extra]:
        for S, pS in items:
            if i in S and j in S:
                for (o, pe) in _split_to_orders(S, pS):
                    if (o[0], o[1]) in [(i, j), (j, i)]:
                        extra.append((o, pe, S))
                        break
                break
    # 末尾に追加
    return cands + extra

# --- 市場由来の過熱判定 ---
def head_market_rate(tri_odds: Dict[Order3, float], head: int = 1) -> float:
    mass = 0.0
    den = 0.0
    for (i,j,k), o in tri_odds.items():
        if o <= 0: 
            continue
        w = 1.0 / o
        den += w
        if i == head:
            mass += w
    return (mass / den) if den > 0 else 0.0

def pair_overbet_ratio(
    pair: Tuple[int,int],
    pmap: Dict[TrioSet, float],
    tri_odds: Dict[Order3, float],
    beta: float = 1.0
) -> float:
    # 期待（モデル）
    (i, j) = pair
    exp_mass = 0.0
    for S, pS in pmap.items():
        if i in S and j in S:
            for (o, px) in _split_to_orders(S, pS):
                if (o[0], o[1]) in [(i, j), (j, i)]:
                    exp_mass += px
    # 市場
    mkt_mass = 0.0
    den = 0.0
    for (a,b,c), o in tri_odds.items():
        if o <= 0: 
            continue
        w = 1.0 / o
        den += w
        if (a,b) in [(i,j),(j,i)]:
            mkt_mass += w
    mkt = (mkt_mass / den) if den > 0 else 0.0
    exp = exp_mass * beta
    return (mkt / exp) if exp > 0 else 0.0

def value_ratios_for_pair(
    head: int, second: int,
    pmap: Dict[TrioSet, float],
    tri_odds: Dict[Order3, float]
) -> List[Tuple[int, float, float, float, Optional[float]]]:
    """
    並び head-second 固定で、3着候補ごとの
    (3着, 期待（モデル）, 市場（人気）, 市場/期待, オッズ) を返す
    """
    # 期待（モデル）
    exp = defaultdict(float)
    for S, pS in pmap.items():
        if head in S and second in S:
            for (o, px) in _split_to_orders(S, pS):
                if o[0] == head and o[1] == second:
                    exp[o[2]] += px
    # 市場（3着ごと）
    mkt = defaultdict(float)
    den = 0.0
    for (i,j,k), o in tri_odds.items():
        if o <= 0: 
            continue
        w = 1.0 / o
        den += w
        if i == head and j == second:
            mkt[k] += w
    for k in list(mkt.keys()):
        mkt[k] = mkt[k] / den if den > 0 else 0.0

    out = []
    for k in range(1,7):
        odds = tri_odds.get((head, second, k))  # 無い時は None
        ex = exp.get(k, 0.0)
        mk = mkt.get(k, 0.0)
        ratio = (mk / ex) if ex > 0 else float("inf")
        out.append((k, ex, mk, ratio, odds))
    return out

# --- EV・資金配分 ---
def ev_of(order: Order3, p_est: float, tri_odds: Dict[Order3, float], margin: float):
    odds = tri_odds.get(order)
    if odds is None or odds <= 0:
        return (None, None, None, False)
    req = (1.0 + margin) / odds  # 必要p
    ev = p_est * odds - 1.0
    ok = (p_est >= req)
    return (odds, req, ev, ok)

def allocate_budget_by_prob(
    buys: List[Tuple[Order3, float, TrioSet]],
    race_cap: int,
    min_unit: int = 100
) -> Tuple[List[Tuple[Order3, float, TrioSet, int]], int]:
    """
    buys: (order, p_est, S)
    返り値: [(order, p_est, S, bet_amount)], used_total
    """
    if race_cap <= 0 or not buys:
        return [], 0
    total_p = sum(p for _, p, _ in buys) or 1.0
    units = race_cap // min_unit
    if units <= 0:
        return [], 0

    # 初期配分：p比率で丸め
    raw = [(o, p, S, int(round((p / total_p) * units))) for (o, p, S) in buys]
    # 0 単位は1単位に底上げ（取捨は上位を優先）
    for idx, (o, p, S, u) in enumerate(raw):
        if u <= 0:
            raw[idx] = (o, p, S, 1)
    # 合計調整
    cur = sum(u for *_, u in raw)
    while cur > units:
        # pが最小のものから1ずつ削る
        mi = min(range(len(raw)), key=lambda i: raw[i][1] if raw[i][3] > 0 else 1e9)
        o, p, S, u = raw[mi]
        if u > 0:
            raw[mi] = (o, p, S, u - 1)
            cur -= 1
        else:
            break
    while cur < units:
        ma = max(range(len(raw)), key=lambda i: raw[i][1])
        o, p, S, u = raw[ma]
        raw[ma] = (o, p, S, u + 1)
        cur += 1

    out = [(o, p, S, u * min_unit) for (o, p, S, u) in raw if u > 0]
    used = sum(x[3] for x in out)
    return out, used

# --- 点数絞り（6〜8推奨）と同一ペア抑制 ---
def trim_candidates_with_rules(
    ok_rows: List[Tuple[Order3, float, TrioSet, float, float, float, bool]],
    max_points: int = 8,
    max_same_pair_points: int = 2,
    add_margin_pp: int = 0
):
    """
    ok_rows: [(order, p_est, S, odds, req, ev, ok=True), ...]
    ルール:
      - EV降順で並び替え
      - 同一(頭,2着)ペアは最大 max_same_pair_points 点まで
      - 多すぎる時は上から max_points まで
      - それでも多い場合は add_margin_pp で再フィルタ（外側で使う）
    """
    # EV降順 → 期待リターン降順（p*odds）で安定ソート
    ok_rows = sorted(ok_rows, key=lambda x: (x[5], x[1]*x[3]), reverse=True)
    pair_count = Counter()
    trimmed = []
    for row in ok_rows:
        o, p_est, S, odds, req, ev, ok = row
        pair = (o[0], o[1])
        if pair_count[pair] >= max_same_pair_points:
            continue
        trimmed.append(row)
        pair_count[pair] += 1
        if len(trimmed) >= max_points:
            break
    return trimmed
