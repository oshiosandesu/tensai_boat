# -*- coding: utf-8 -*-
"""
copy.py  —  “オッズだけで読む”競艇AIコア（市場の歪み=過大/過小評価の可視化 版・全文）

このファイルは、現行の core.py を置き換え/併用できるように設計。
主なポイント（ROI重視のチューニングを網羅）:

  A) 開催会場の自動検出
     - venues_on(date): その日に“開催していそうな”会場IDを返す（高速）

  B) 三連複TopN → 確率化 → 三連単順序分解
     - α（1/odds^α）の “自動選択”（堅いほどα↑）
     - 内枠・中/外枠の軽い順序バイアス（HEAD/MID/TAIL）

  C) 市場の確率（q）とモデルの確率（p）の比較
     - 三連単ごとの割安スコア: VR_tri = p(o) / q(o)
     - (頭,2着)ペアの過熱度: Over_pair = 市場/期待
     - 1着の過熱度（頭率）: Over_head = 市場/期待

  D) EV（期待値）判定を “オッズ帯で出し分け”
     - 短配当（<12）: 緩めない/やや厳しめ
     - 中配当（12–25）: 主戦場（余裕+6〜10%）
     - 長配当（25–60）: さらに厳しめ（+8〜12%）
     - 超長配当（>60）: 原則不採用
     - 過熱ペアは “課金” （余裕+3%）で自然に落とす
     - スリッページ（約定オッズ低下）を見込んだEV判定（例：-7%）

  E) レース入口フィルター（まず地雷を踏まない）
     - Top5カバレッジ帯で選別
     - 1号艇の市場頭率レンジで選別
     - 過熱ペアしきい値（>1.4）で除外

  F) 点数上限・ペア上限・資金配分
     - 総点数上限（例：6–8点）
     - 同一(頭,2着)ペア上限（例：2点）
     - 半ケリー + 1点あたり上限 + 総上限

依存:
  - Python 3.9+
  - pyjpboatrace (オッズ/結果/直前データの取得)
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict, Counter
from datetime import date
from functools import lru_cache
import math

# ========= 型 =========
TrioSet = frozenset            # frozenset({a,b,c}) を3連複セットのキーに
Order3 = Tuple[int, int, int]  # (i, j, k) = 3連単の順序タプル

# ========= 会場マスタ =========
VENUES: List[Tuple[int, str]] = [
    (1, "桐生"), (2, "戸田"), (3, "江戸川"), (4, "平和島"),
    (5, "多摩川"), (6, "浜名湖"), (7, "蒲郡"), (8, "常滑"),
    (9, "津"), (10, "三国"), (11, "びわこ"), (12, "住之江"),
    (13, "尼崎"), (14, "鳴門"), (15, "丸亀"), (16, "児島"),
    (17, "宮島"), (18, "徳山"), (19, "下関"), (20, "若松"),
    (21, "芦屋"), (22, "福岡"), (23, "唐津"), (24, "大村"),
]
VENUE_ID2NAME = {vid: name for vid, name in VENUES}

# ========= 順序分解の軽いバイアス（内枠優勢を控えめに反映） =========
HEAD_BIAS = {1: 1.35, 2: 1.05, 3: 0.95, 4: 0.85, 5: 0.80, 6: 0.75}
MID_BIAS  = {1: 1.20, 2: 1.05, 3: 1.00, 4: 0.95, 5: 0.90, 6: 0.85}
TAIL_BIAS = {1: 1.10, 2: 1.05, 3: 1.02, 4: 1.00, 5: 0.98, 6: 0.96}

# ========= データ取得（pyjpboatrace） =========
try:
    from pyjpboatrace import PyJPBoatrace  # v0.4.x を想定
except Exception:
    PyJPBoatrace = None

_client_singleton: Optional["PyJPBoatrace"] = None

def _client() -> "PyJPBoatrace":
    global _client_singleton
    if PyJPBoatrace is None:
        raise RuntimeError("pyjpboatrace が利用できません。requirements とインストール状況を確認してください。")
    if _client_singleton is None:
        _client_singleton = PyJPBoatrace()
    return _client_singleton

# ========= オッズ/結果/直前 =========
def get_trio_odds(d: date, venue_id: int, rno: int) -> Tuple[Dict[TrioSet, float], Optional[str]]:
    """
    三連複オッズ → {frozenset({a,b,c}): 倍率} に整形。
    返り値: (odds_map, update_tag)
    """
    cli = _client()
    raw = cli.get_odds_trio(d, venue_id, rno)  # 例: {'1=2=3': 5.1, 'update': '締切時オッズ', ...}
    update = raw.get("update")
    odds: Dict[TrioSet, float] = {}
    for k, v in raw.items():
        if isinstance(k, str) and k.count("=") == 2:
            try:
                a, b, c = (int(x) for x in k.split("="))
                odds[frozenset((a, b, c))] = float(v)
            except Exception:
                continue
    return odds, update

def get_trifecta_odds(d: date, venue_id: int, rno: int) -> Dict[Order3, float]:
    """
    三連単オッズ → {(a,b,c): 倍率} に整形。
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
                continue
    return odds

def get_trifecta_result(d: date, venue_id: int, rno: int) -> Optional[Tuple[Order3, float]]:
    """
    三連単の確定結果（((a,b,c), 倍率)）を取得。
    倍率 = 100円あたり払戻金 / 100。
    """
    try:
        cli = _client()
        res = cli.get_race_result(d, venue_id, rno)
        payoff = res.get("payoff") or {}
        tri = payoff.get("trifecta") or res.get("trifecta")
        if not tri:
            return None
        order_str = str(tri.get("result", "")).strip()
        if order_str.count("-") != 2:
            return None
        a, b, c = (int(x) for x in order_str.split("-"))
        payoff_yen = float(tri.get("payoff", 0.0))
        odds = payoff_yen / 100.0
        return ((a, b, c), odds)
    except Exception:
        return None

def get_just_before_info(d: date, venue_id: int, rno: int) -> Optional[dict]:
    """
    直前情報（展示タイム・風・波など）を辞書で返す。取得不可なら None。
    """
    try:
        cli = _client()
        return cli.get_just_before_info(d, venue_id, rno)
    except Exception:
        return None

# ========= 開催会場の自動検出（高速） =========
@lru_cache(maxsize=128)
def is_race_available(d: date, venue_id: int, rno: int = 1) -> bool:
    """
    指定日の場・レースに三連複オッズが存在するかを軽く確認。
    1Rが空の場合は 6R を再チェックして True/False を返す。
    """
    try:
        odds1, _ = get_trio_odds(d, venue_id, rno)
        if odds1:
            return True
        odds6, _ = get_trio_odds(d, venue_id, 6)
        return bool(odds6)
    except Exception:
        return False

@lru_cache(maxsize=32)
def venues_on(d: date) -> List[int]:
    """
    指定日に開催していそうな会場IDのリスト。
    """
    active: List[int] = []
    for vid, _name in VENUES:
        if is_race_available(d, vid, 1):
            active.append(vid)
    return active

# ========= 3複TopN → 確率化 =========
def _choose_alpha_by_cov5(cov5: float) -> float:
    # 固いほど α を大きく（上位強調）→ ヒット率寄り
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
    alpha: float = 0.9,
) -> Tuple[Dict[TrioSet, float], List[Tuple[TrioSet, float]]]:
    """
    trio_sorted_items : (S, odds) の“オッズ昇順”リスト（外で sort 済み）
    返り値:
      - pmap_topN: {S: p}（TopNを 1/odds^α で正規化）
      - items    : TopNの (S, odds) リスト
    """
    items = trio_sorted_items[:top_n]
    weights: List[Tuple[TrioSet, float]] = []
    for S, o in items:
        w = 0.0
        if o and o > 0:
            w = (1.0 / o) ** alpha
        weights.append((S, w))
    denom = sum(w for _, w in weights) or 1.0
    pmap = {S: (w / denom) for S, w in weights}
    return pmap, items

def normalize_with_dynamic_alpha(
    trio_sorted_items: List[Tuple[TrioSet, float]],
    top_n: int = 10
) -> Tuple[Dict[TrioSet, float], List[Tuple[TrioSet, float]], float, float]:
    """
    trio_sorted_items: (S, odds) のオッズ昇順リスト
    返り値: (pmap, items, alpha_used, cov5_preview)
      - cov5_preview: まず α=1.0 で概算した Top5カバレッジ
      - alpha_used  : cov5_preview に応じて選んだ α
    """
    base, items = normalize_probs_from_odds(trio_sorted_items, top_n=top_n, alpha=1.0)
    cov5_preview = top5_coverage(base)
    alpha = _choose_alpha_by_cov5(cov5_preview)
    pmap, items = normalize_probs_from_odds(trio_sorted_items, top_n=top_n, alpha=alpha)
    return pmap, items, alpha, cov5_preview

# ========= 指標 =========
def top5_coverage(pmap: Dict[TrioSet, float]) -> float:
    vals = sorted(pmap.values(), reverse=True)[:5]
    return sum(vals)

def inclusion_mass_for_boat(pmap: Dict[TrioSet, float], boat: int) -> float:
    return sum(p for S, p in pmap.items() if boat in S)

def pair_mass(pmap: Dict[TrioSet, float]) -> Dict[Tuple[int, int], float]:
    """
    3複TopNの確率から、2艇コンビの“同舟券になりやすさ”を集計。
    """
    out = defaultdict(float)
    for S, p in pmap.items():
        a, b, c = sorted(list(S))
        for i, j in ((a, b), (a, c), (b, c)):
            out[(i, j)] += p
    return out

# ========= 3複 → 3単 順序分解 =========
def _perm_weights(order: Order3) -> float:
    i, j, k = order
    return HEAD_BIAS[i] * MID_BIAS[j] * TAIL_BIAS[k]

def _split_to_orders(S: TrioSet, pS: float) -> List[Tuple[Order3, float]]:
    """
    S={a,b,c} の確率 pS を、重み比例で 6通りの順序へ分配。
    """
    a, b, c = list(S)
    perms: List[Order3] = [(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)]
    ws = [_perm_weights(o) for o in perms]
    z = sum(ws) or 1.0
    return [(perms[t], pS * ws[t] / z) for t in range(6)]

def estimate_head_rate(pmap: Dict[TrioSet, float], head: int = 1) -> float:
    """
    “◯号艇が1着になる見込み”を、3複pから順序分解して推定。
    """
    acc = 0.0
    for S, pS in pmap.items():
        if head not in S:
            continue
        for o, px in _split_to_orders(S, pS):
            if o[0] == head:
                acc += px
    return acc

# ========= “市場の確率” q（正規化された 1/odds） =========
def market_prob_trifecta(tri_odds: Dict[Order3, float]) -> Dict[Order3, float]:
    """
    三連単の市場確率 q(o) を作る（1/odds を全体正規化）。
    """
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
    return sum(q.get((head, j, k), 0.0) for j in range(1, 7) for k in range(1, 7) if j != head and k not in (head, j))

def market_pair_rate(tri_odds: Dict[Order3, float], i: int, j: int) -> float:
    q = market_prob_trifecta(tri_odds)
    s = 0.0
    for k in range(1, 7):
        if k in (i, j): 
            continue
        s += q.get((i, j, k), 0.0)
        s += q.get((j, i, k), 0.0)
    return s

# ========= “モデルの確率” p（3複→順序分解） =========
def model_prob_trifecta(pmap: Dict[TrioSet, float]) -> Dict[Order3, float]:
    """
    3複セット確率 pmap から三連単確率 p(o) を生成。
    """
    out = defaultdict(float)
    for S, pS in pmap.items():
        for o, px in _split_to_orders(S, pS):
            out[o] += px
    return dict(out)

def model_head_rate(pmap: Dict[TrioSet, float], head: int) -> float:
    ptri = model_prob_trifecta(pmap)
    return sum(ptri.get((head, j, k), 0.0) for j in range(1, 7) for k in range(1, 7) if j != head and k not in (head, j))

def model_pair_rate(pmap: Dict[TrioSet, float], i: int, j: int) -> float:
    ptri = model_prob_trifecta(pmap)
    s = 0.0
    for k in range(1, 7):
        if k in (i, j): 
            continue
        s += ptri.get((i, j, k), 0.0)
        s += ptri.get((j, i, k), 0.0)
    return s

# ========= 歪みの指標（割安/過熱） =========
def value_ratio_tri(p_tri: Dict[Order3, float], q_tri: Dict[Order3, float]) -> Dict[Order3, float]:
    """
    並びごとの割安スコア VR = p(o)/q(o) を返す（q(o)=0は +inf）。
    """
    out = {}
    for o, p in p_tri.items():
        q = q_tri.get(o, 0.0)
        if q <= 0:
            out[o] = float("inf") if p > 0 else 1.0
        else:
            out[o] = p / q
    return out

def pair_overbet_ratio(pair: Tuple[int, int], pmap: Dict[TrioSet, float], tri_odds: Dict[Order3, float]) -> float:
    """
    (頭,2着) ペアの過熱度 = 市場/期待。
    """
    i, j = pair
    # 期待（モデル）
    exp_mass = 0.0
    for S, pS in pmap.items():
        if i in S and j in S:
            for (o, px) in _split_to_orders(S, pS):
                if (o[0], o[1]) in ((i, j), (j, i)):
                    exp_mass += px
    # 市場
    mkt = market_pair_rate(tri_odds, i, j)
    return (mkt / exp_mass) if exp_mass > 0 else float("inf")

def head_overbet_ratio(head: int, pmap: Dict[TrioSet, float], tri_odds: Dict[Order3, float]) -> float:
    """
    1着の過熱度 = 市場/期待。
    """
    p_head = model_head_rate(pmap, head)
    q_head = market_head_rate(tri_odds, head)
    return (q_head / p_head) if p_head > 0 else float("inf")

# ========= 採用セット数 R（固さによる粗い制御） =========
def choose_R_by_coverage(pmap: Dict[TrioSet, float]) -> Tuple[int, str]:
    """
    レースの“固さ”から、採用する 3複セット数 R を決定（候補母集団を少し広げ気味）。
    """
    cov5 = top5_coverage(pmap)
    if cov5 >= 0.75:
        return 5, f"堅め（Top5={cov5:.1%}）"
    elif cov5 >= 0.65:
        return 6, f"やや本命（Top5={cov5:.1%}）"
    elif cov5 >= 0.55:
        return 7, f"中庸（Top5={cov5:.1%}）"
    else:
        return 8, f"やや荒れ（Top5={cov5:.1%}）"

# ========= 候補生成 =========
def build_trifecta_candidates(
    pmap: Dict[TrioSet, float],
    R: int,
    avoid_top: bool = False,          # 最有力も採用（本命を素直に残す）
    max_per_set: Optional[int] = None,
    cov5_hint: Optional[float] = None,
) -> List[Tuple[Order3, float, TrioSet]]:
    """
    上位Rセットから順序分解を行い、各セットごとに max_per_set 点を抽出。
    cov5_hint があれば:
      - cov5 >= 0.75 → 3点/セット
      - それ以外      → 2点/セット
    返り値: [(order, p_est, from_set), ...] を “当たりやすさ p_est” 降順。
    """
    if max_per_set is None:
        if (cov5_hint is not None) and (cov5_hint >= 0.75):
            max_per_set = 3
        else:
            max_per_set = 2

    items = sorted(pmap.items(), key=lambda x: x[1], reverse=True)[:R]
    candidates: List[Tuple[Order3, float, TrioSet]] = []

    for S, pS in items:
        perm = _split_to_orders(S, pS)
        perm_sorted = sorted(perm, key=lambda x: _perm_weights(x[0]), reverse=True)

        picked = []
        skipped_once = False
        for (o, pe) in perm_sorted:
            if avoid_top and not skipped_once:
                skipped_once = True
                continue
            picked.append((o, pe, S))
            if len(picked) >= max_per_set:
                break

        candidates.extend(picked)

    return sorted(candidates, key=lambda x: x[1], reverse=True)

def add_pair_hedge_if_needed(
    cands: List[Tuple[Order3, float, TrioSet]],
    pmap: Dict[TrioSet, float],
    top_pairs: List[Tuple[Tuple[int, int], float]],
    max_extra: int = 1
) -> List[Tuple[Order3, float, TrioSet]]:
    """
    上位ペア（頭-2着いずれかの順序）がひとつも含まれていない場合、
    そのペアを満たす並びを “最大 max_extra 点” だけ追加する。
    """
    if not top_pairs:
        return cands[:]

    present_pairs = {(o[0], o[1]) for (o, _, _) in cands} | {(o[1], o[0]) for (o, _, _) in cands}
    need_pairs: List[Tuple[int, int]] = []
    for (i, j), _mass in top_pairs:
        if (i, j) not in present_pairs and (j, i) not in present_pairs:
            need_pairs.append((i, j))

    if not need_pairs:
        return cands[:]

    items = sorted(pmap.items(), key=lambda x: x[1], reverse=True)
    extra: List[Tuple[Order3, float, TrioSet]] = []
    for (i, j) in need_pairs[:max_extra]:
        for S, pS in items:
            if i in S and j in S:
                for (o, pe) in _split_to_orders(S, pS):
                    if (o[0], o[1]) in ((i, j), (j, i)):
                        extra.append((o, pe, S))
                        break
                break

    return cands + extra

# ========= EV（期待値）判定・オッズ帯出し分け・過熱課金・スリッページ =========
def adjust_for_slippage(odds: float, slip_rate: float = 0.07) -> float:
    """
    スリッページ（約定時にオッズが悪化する想定）を反映。
    例: slip_rate=0.07 なら、表示オッズ×(1-0.07) で評価。
    """
    if odds is None:
        return None
    return max(1.0, odds * (1.0 - max(0.0, slip_rate)))

def ev_of_band(
    order: Order3,
    p_est: float,
    tri_odds: Dict[Order3, float],
    base_margin: float,
    *,
    long_odds_extra: float = 0.10,     # 25–60倍は +10% 程度を推奨
    short_odds_relax: float = -0.00,   # <12倍は緩めない（0〜+2%を推奨なら正に）
    long_odds_threshold: float = 25.0,
    short_odds_threshold: float = 12.0,
    max_odds: float = 60.0,            # >60倍は原則不採用
    slippage: float = 0.07,            # スリッページ率
    extra_margin: float = 0.0          # 過熱ペアなどの課金加算
) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
    """
    オッズ帯によって余裕%を出し分けし、スリッページを反映したEV判定を行う。
    返り値: (odds_adj, req_p, ev, ok)
    """
    odds = tri_odds.get(order)
    if (odds is None) or (odds <= 0):
        return (None, None, None, False)
    if odds > max_odds:
        return (odds, None, None, False)

    # スリッページを反映した評価用オッズ
    odds_eval = adjust_for_slippage(odds, slippage)

    # 余裕%の出し分け
    m = base_margin + extra_margin
    if odds_eval < short_odds_threshold:
        m = max(0.0, base_margin + short_odds_relax + extra_margin)
    elif odds_eval >= long_odds_threshold:
        m = base_margin + long_odds_extra + extra_margin

    req = (1.0 + m) / odds_eval
    ev = p_est * odds_eval - 1.0
    ok = (p_est >= req)
    return (odds_eval, req, ev, ok)

def evaluate_candidates_with_overbet(
    cands: List[Tuple[Order3, float, TrioSet]],
    pmap: Dict[TrioSet, float],
    tri_odds: Dict[Order3, float],
    base_margin: float,
    *,
    overbet_thresh: float = 1.30,
    overbet_cut: float = 1.50,
    overbet_extra: float = 0.03,
    long_odds_extra: float = 0.10,
    short_odds_relax: float = 0.00,
    long_odds_threshold: float = 25.0,
    short_odds_threshold: float = 12.0,
    max_odds: float = 60.0,
    slippage: float = 0.07
) -> List[Tuple[Order3, float, TrioSet, float, float, float, bool, float]]:
    """
    候補ごとに EV 判定を行う。過熱ペアには余裕+α を上乗せ、超過熱は切る。
    返り値: [(order, p_est, from_set, odds_eval, req, ev, ok, over_ratio)]
    """
    out = []
    cache_over = {}
    for (o, p_est, S) in cands:
        pair = (o[0], o[1])
        if pair not in cache_over:
            cache_over[pair] = pair_overbet_ratio(pair, pmap, tri_odds)
        over = cache_over[pair]

        # 超過熱は切る
        if over >= overbet_cut:
            out.append((o, p_est, S, tri_odds.get(o, None), None, None, False, over))
            continue

        # しきい値以上は課金
        extra = overbet_extra if over >= overbet_thresh else 0.0

        odds_eval, req, ev, ok = ev_of_band(
            o, p_est, tri_odds, base_margin,
            long_odds_extra=long_odds_extra,
            short_odds_relax=short_odds_relax,
            long_odds_threshold=long_odds_threshold,
            short_odds_threshold=short_odds_threshold,
            max_odds=max_odds,
            slippage=slippage,
            extra_margin=extra
        )
        out.append((o, p_est, S, odds_eval, req, ev, ok, over))
    return out

# ========= トリミング（点数上限/ペア上限） =========
def trim_candidates_with_rules(
    rows: List[Tuple[Order3, float, TrioSet, float, float, float, bool, float]],
    max_points: int = 8,
    max_same_pair_points: int = 2
) -> List[Tuple[Order3, float, TrioSet, float, float, float, bool, float]]:
    """
    EV達成（ok=True）の中から、
      - EV降順（同点なら 期待リターン p*odds）で優先
      - 同一 (頭,2着) ペアは最大 max_same_pair_points まで
      - 上から max_points まで
    """
    ok_rows = [r for r in rows if r[6] is True and (r[3] is not None)]
    ok_rows_sorted = sorted(ok_rows, key=lambda x: (x[5], x[1] * (x[3] or 0.0)), reverse=True)

    pair_count = Counter()
    trimmed: List[Tuple[Order3, float, TrioSet, float, float, float, bool, float]] = []
    for row in ok_rows_sorted:
        o = row[0]
        pair = (o[0], o[1])
        if pair_count[pair] >= max_same_pair_points:
            continue
        trimmed.append(row)
        pair_count[pair] += 1
        if len(trimmed) >= max_points:
            break
    return trimmed

# ========= 資金配分（半ケリー + 上限リミッター） =========
def half_kelly_bet_fraction(p_est: float, odds_eval: float) -> float:
    """
    ケリー基準の 1/2。p*(b) - (1-p) / b を 1/2 に縮小。
    ここで b = odds-1。負値は 0 に丸め。
    """
    if (odds_eval is None) or (odds_eval <= 1.0):
        return 0.0
    b = odds_eval - 1.0
    f = (p_est - (1.0 - p_est) / b)
    return max(0.0, 0.5 * f)

def allocate_budget_safely(
    buys: List[Tuple[Order3, float, TrioSet, float, float, float, bool, float]],
    race_cap: int,
    min_unit: int = 100,
    per_bet_cap_ratio: float = 0.40  # 1点あたり上限（総上限の40%など）
) -> Tuple[List[Tuple[Order3, float, TrioSet, int, float]], int]:
    """
    半ケリーに基づいて“望ましい額”を出し、最小刻みに丸め、かつ各点に上限をかけて配分。
    返り値: [(order, p_est, from_set, bet_yen, odds_eval)], used_total
    """
    if race_cap <= 0 or not buys:
        return [], 0

    # 各点の理想額（半ケリー × race_cap）を計算
    wish = []
    for (o, p_est, S, odds_eval, req, ev, ok, over) in buys:
        f = half_kelly_bet_fraction(p_est, odds_eval)
        amt = f * race_cap
        wish.append((o, p_est, S, odds_eval, amt))

    # 全額が小さすぎる場合に備えて比率再配分（ゼロが多発すると配れないため）
    total_wish = sum(max(0.0, x[4]) for x in wish)
    if total_wish <= 0:
        # すべて同額で1単位ずつ配る（最低限）
        units = race_cap // min_unit
        if units <= 0:
            return [], 0
        unit_each = max(1, units // len(wish))
        out = []
        used_units = 0
        for (o, p, S, odds_eval, _) in wish:
            out.append((o, p, S, unit_each * min_unit, odds_eval))
            used_units += unit_each
            if used_units >= units:
                break
        used = sum(x[3] for x in out)
        return out, used

    # 各点の上限
    per_cap = race_cap * per_bet_cap_ratio

    # 希望額に基づいて丸め＋上限適用
    prelim = []
    for (o, p, S, odds_eval, amt) in wish:
        amt = min(amt, per_cap)
        units = int(round(amt / min_unit))
        prelim.append((o, p, S, odds_eval, units))

    # 合計ユニットを race_cap に合わせる
    target_units = race_cap // min_unit
    cur_units = sum(u for *_, u in prelim)
    # 足りない/超過を調整
    if cur_units == 0:
        # 最低限1単位ずつ配る
        prelim = [(o, p, S, od, 1) for (o, p, S, od, u) in prelim]
        cur_units = len(prelim)
    while cur_units > target_units and prelim:
        # pが小さいものから削る
        idx = min(range(len(prelim)), key=lambda i: prelim[i][1] if prelim[i][4] > 0 else 1e9)
        o, p, S, od, u = prelim[idx]
        if u > 0:
            prelim[idx] = (o, p, S, od, u - 1)
            cur_units -= 1
        else:
            break
    while cur_units < target_units:
        # pが大きいものに足す（ただし per_cap を超えないように）
        idx = max(range(len(prelim)), key=lambda i: prelim[i][1])
        o, p, S, od, u = prelim[idx]
        prelim[idx] = (o, p, S, od, u + 1)
        cur_units += 1

    out = [(o, p, S, u * min_unit, od) for (o, p, S, od, u) in prelim if u > 0]
    used = sum(x[3] for x in out)
    return out, used

# ========= レース入口フィルター（地雷回避） =========
def should_skip_race_by_entry_filters(
    pmap_top10: Dict[TrioSet, float],
    tri_odds: Dict[Order3, float],
    *,
    cov5_min: float = 0.55,
    cov5_max: float = 0.75,
    head1_min: float = 0.25,
    head1_max: float = 0.38,
    over_pair_drop: float = 1.40
) -> Tuple[bool, Dict[str, Any]]:
    """
    レースを“最初から回さない”判断。True=見送り。
      - Top5カバレッジが極端すぎる（>0.85 or <0.45）を避ける前に、まず狙い目帯に限定
      - 1号艇の市場頭率が高すぎ/低すぎるレースを避ける
      - 過熱ペアが極端に強いレースを避ける
    """
    cov5 = top5_coverage(pmap_top10)
    if not (cov5_min <= cov5 <= cov5_max):
        return True, {"reason": "cov5_out", "cov5": cov5}

    # 1号艇の市場頭率
    q_head1 = market_head_rate(tri_odds, 1)
    if not (head1_min <= q_head1 <= head1_max):
        return True, {"reason": "head1_out", "q_head1": q_head1, "cov5": cov5}

    # 過熱ペアのチェック（上位ペア3つ程度）
    p_pairs = pair_mass(pmap_top10)
    top_pairs = sorted(p_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
    for (i, j), _mass in top_pairs:
        over = pair_overbet_ratio((i, j), pmap_top10, tri_odds)
        if over >= over_pair_drop:
            return True, {"reason": "overpair_out", "pair": (i, j), "over": over, "cov5": cov5}

    return False, {"cov5": cov5, "q_head1": q_head1}

# ========= 3着の妙味ヒント =========
def value_ratios_for_pair(
    head: int, second: int,
    pmap: Dict[TrioSet, float],
    tri_odds: Dict[Order3, float]
) -> List[Tuple[int, float, float, float, Optional[float]]]:
    """
    固定 (head, second) のとき、3着ごとの
    (3着, モデル期待, 市場人気, 市場/期待, オッズ) を返す（“3着の妙味”ヒント用）
    """
    # モデル期待
    exp = defaultdict(float)
    for S, pS in pmap.items():
        if head in S and second in S:
            for (o, px) in _split_to_orders(S, pS):
                if o[0] == head and o[1] == second:
                    exp[o[2]] += px

    # 市場人気
    q = market_prob_trifecta(tri_odds)
    mkt = defaultdict(float)
    den = 0.0
    for (i, j, k), qq in q.items():
        den += qq
        if i == head and j == second:
            mkt[k] += qq
    for k in list(mkt.keys()):
        mkt[k] = (mkt[k] / den) if den > 0 else 0.0

    out: List[Tuple[int, float, float, float, Optional[float]]] = []
    for k in range(1, 7):
        odds = tri_odds.get((head, second, k))
        ex = exp.get(k, 0.0)
        mk = mkt.get(k, 0.0)
        ratio = (mk / ex) if ex > 0 else float("inf")
        out.append((k, ex, mk, ratio, odds))
    return out
