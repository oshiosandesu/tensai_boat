# -*- coding: utf-8 -*-
"""
core.py  —  “オッズだけで読む”競艇AIコア（開催会場自動検出 付き・全文）

この版の主なポイント:
  A) 開催会場だけを自動検出して日次バックテスト側で使えるようにするヘルパーを追加
     - detect_active_venues(date) … その日「開催していそうな会場IDのリスト」を返す
     - venues_on(date)            … detect_active_venues の別名（読みやすさ用）
     - is_race_available(date, vid, rno) … レース単位の可用性チェック（内部で使用）
     ※ 1R（→空なら6R）だけを軽く叩いて判断（全レース走査より劇的に高速）

  B) 予想ロジックのチューニング（“当たりが付きにくい”状況を緩和）
     1) 3連複TopNの確率化 α=0.9（上位セットをやや重視）
     2) 3複→3単の候補生成で最有力スキップを廃止（avoid_top=False）
     3) EV判定で長配当（25倍以上）の margin を 2pp だけ軽減
     4) “固さ”→R（採用セット数）を全体に +1（候補母集団を少し広げる）

アプリ側（1レース診断/日次バックテスト）からの I/F は互換のままです。
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict, Counter
from datetime import date
from functools import lru_cache

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

# ========= 三連複/三連単/結果/直前 情報取得 =========
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

# ========= 開催会場の自動検出（高速・軽量） =========
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
        # 1Rが空＝開催なし or 未公開の可能性。6Rも軽く見る。
        odds6, _ = get_trio_odds(d, venue_id, 6)
        return bool(odds6)
    except Exception:
        return False

@lru_cache(maxsize=32)
def detect_active_venues(d: date) -> List[int]:
    """
    指定日に“開催していそうな”会場IDのリストを返す。
    1R（ダメなら6R）だけを見るので全レース走査より大幅に高速。
    """
    active: List[int] = []
    for vid, _name in VENUES:
        if is_race_available(d, vid, 1):
            active.append(vid)
    return active

# 読みやすさのための別名
def venues_on(d: date) -> List[int]:
    """detect_active_venues のエイリアス。"""
    return detect_active_venues(d)

# ========= 3複TopN → 確率化 =========
def normalize_probs_from_odds(
    trio_sorted_items: List[Tuple[TrioSet, float]],
    top_n: int = 10,
    alpha: float = 0.9,  # ← 既定を 0.9 に（従来は 1.0）
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
        if head not in S:  # 含まれないセットは飛ばす
            continue
        for o, px in _split_to_orders(S, pS):
            if o[0] == head:
                acc += px
    return acc

def choose_R_by_coverage(pmap: Dict[TrioSet, float]) -> Tuple[int, str]:
    """
    レースの“固さ”から、採用する 3複セット数 R を大づかみに決定。
    ※ 従来より +1（候補母集団を少し広げる）
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

def coverage_targets(pmap: Dict[TrioSet, float], targets=(0.25, 0.50, 0.75)):
    """
    “上位から積んだとき、どこまでで t% に到達するか”を返す。
    """
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

# ========= 候補生成・補強 =========
def build_trifecta_candidates(
    pmap: Dict[TrioSet, float],
    R: int,
    avoid_top: bool = False,  # ← 既定を False（最有力も採用）
    max_per_set: int = 2
) -> List[Tuple[Order3, float, TrioSet]]:
    """
    上位Rセットから順序分解を行い、各セットごとに max_per_set 点を抽出。
    返り値: [(order, p_est, from_set), ...] を “当たりやすさ p_est” 降順で整列。
    """
    items = sorted(pmap.items(), key=lambda x: x[1], reverse=True)[:R]
    candidates: List[Tuple[Order3, float, TrioSet]] = []

    for S, pS in items:
        perm = _split_to_orders(S, pS)
        perm_sorted = sorted(perm, key=lambda x: _perm_weights(x[0]), reverse=True)

        picked = []
        skipped_once = False
        for (o, pe) in perm_sorted:
            if avoid_top and not skipped_once:
                skipped_once = True  # 古い動作互換用（avoid_top=Trueのときのみ1点スキップ）
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

# ========= 市場バイアス関連（“人気の集まり”） =========
def head_market_rate(tri_odds: Dict[Order3, float], head: int = 1) -> float:
    """
    三連単オッズ全体から “iが頭になる市場確率” を概算（1/オッズで正規化）。
    """
    mass, den = 0.0, 0.0
    for (i, j, k), o in tri_odds.items():
        if o and o > 0:
            w = 1.0 / o
            den += w
            if i == head:
                mass += w
    return (mass / den) if den > 0 else 0.0

def pair_overbet_ratio(
    pair: Tuple[int, int],
    pmap: Dict[TrioSet, float],
    tri_odds: Dict[Order3, float],
    beta: float = 1.0
) -> float:
    """
    “モデル期待”に対する“市場の過熱度”（市場/期待）を、頭-2着ペアで推定。
    1.0 より大きいほど“買われすぎ”の傾向。
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
    mkt_mass, den = 0.0, 0.0
    for (a, b, _c), o in tri_odds.items():
        if o and o > 0:
            w = 1.0 / o
            den += w
            if (a, b) in ((i, j), (j, i)):
                mkt_mass += w

    mkt = (mkt_mass / den) if den > 0 else 0.0
    exp = exp_mass * beta
    return (mkt / exp) if exp > 0 else float("inf")

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
    mkt = defaultdict(float)
    den = 0.0
    for (i, j, k), o in tri_odds.items():
        if o and o > 0:
            w = 1.0 / o
            den += w
            if i == head and j == second:
                mkt[k] += w
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

# ========= EV（割に合うか） & 資金配分 =========
def ev_of(order: Order3, p_est: float, tri_odds: Dict[Order3, float], margin: float):
    """
    EV（期待値）と“必要p”（=(1+margin)/odds）を計算して判定。
    返り値: (odds, req_p, ev, ok)
    ※ 長配当（25倍以上）は margin を 2pp だけ軽減（過度な厳格で全落ちを防止）
    """
    odds = tri_odds.get(order)
    if (odds is None) or (odds <= 0):
        return (None, None, None, False)
    m = margin
    if odds >= 25.0:
        m = max(0.0, margin - 0.02)
    req = (1.0 + m) / odds
    ev = p_est * odds - 1.0
    ok = (p_est >= req)
    return (odds, req, ev, ok)

def allocate_budget_by_prob(
    buys: List[Tuple[Order3, float, TrioSet]],
    race_cap: int,
    min_unit: int = 100
) -> Tuple[List[Tuple[Order3, float, TrioSet, int]], int]:
    """
    “確率按分”で race_cap を min_unit 刻みに配分。
    返り値: [(order, p_est, from_set, bet_yen)], used_total
    """
    if race_cap <= 0 or not buys:
        return [], 0

    total_p = sum(p for _, p, _ in buys) or 1.0
    units = race_cap // min_unit
    if units <= 0:
        return [], 0

    # 初期配分：p比率で丸め
    raw = [(o, p, S, int(round((p / total_p) * units))) for (o, p, S) in buys]

    # ゼロ配分は 1 単位に底上げ（のちに全体調整）
    for idx, (o, p, S, u) in enumerate(raw):
        if u <= 0:
            raw[idx] = (o, p, S, 1)

    # 合計ユニットを合わせる（最小pから削る／最大pに足す）
    cur = sum(u for *_, u in raw)
    while cur > units:
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

def trim_candidates_with_rules(
    ok_rows: List[Tuple[Order3, float, TrioSet, float, float, float, bool]],
    max_points: int = 8,
    max_same_pair_points: int = 2,
    add_margin_pp: int = 0
) -> List[Tuple[Order3, float, TrioSet, float, float, float, bool]]:
    """
    OK候補（EV達成）の中から、
      - EV降順（同点なら期待リターン p*odds）で優先
      - 同一 (頭,2着) ペアは最大 max_same_pair_points まで
      - 上から max_points まで
    を満たすように絞り込む。
    """
    # EV降順 → 期待リターン（p*odds）降順の安定ソート
    ok_rows_sorted = sorted(ok_rows, key=lambda x: (x[5], x[1] * (x[3] or 0.0)), reverse=True)

    pair_count = Counter()
    trimmed: List[Tuple[Order3, float, TrioSet, float, float, float, bool]] = []
    for row in ok_rows_sorted:
        o, p_est, S, odds, req, ev, ok = row
        pair = (o[0], o[1])
        if pair_count[pair] >= max_same_pair_points:
            continue
        trimmed.append(row)
        pair_count[pair] += 1
        if len(trimmed) >= max_points:
            break
    return trimmed
