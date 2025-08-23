# -*- coding: utf-8 -*-
"""
core_bridge.py
- app.py / pages からの import 互換レイヤー。
- まず core を読み込み、必要な関数を re-export（再公開）。
- core に存在しない関数名は、近い代替やフォールバック実装を与える。
- これにより from core_bridge import (...) を使えば、core 側の差分で ImportError になりにくい。
"""

from __future__ import annotations
from importlib import import_module

# core を読み込む
C = import_module("core")

# ==== ヘルパー ====
def _has(name: str) -> bool:
    return hasattr(C, name)

def _get(name: str):
    return getattr(C, name)

def _export(name: str, obj):
    globals()[name] = obj

# ==== まず core にあるものはそのまま re-export ====
BASIC_EXPORTS = [
    # 取得系
    "get_trio_odds", "get_trifecta_odds", "get_trifecta_result",
    "get_just_before_info",
    # 会場
    "VENUES", "VENUE_ID2NAME", "venues_on", "is_race_available",
    # 確率化・指標
    "normalize_probs_from_odds", "normalize_with_dynamic_alpha",
    "top5_coverage", "pair_mass", "estimate_head_rate",
    # モデル/市場 確率
    "model_prob_trifecta", "market_prob_trifecta",
    "model_head_rate", "market_head_rate", "market_pair_rate",
    # 価値/過熱
    "value_ratio_tri", "pair_overbet_ratio", "head_overbet_ratio",
    # 候補生成・R決定
    "choose_R_by_coverage", "build_trifecta_candidates",
    "add_pair_hedge_if_needed",
    # EV/評価
    "evaluate_candidates_with_overbet", "ev_of_band", "adjust_for_slippage",
    # トリミング・配分
    "trim_candidates_with_rules", "allocate_budget_safely",
    # ヒント類
    "value_ratios_for_pair",
]

for n in BASIC_EXPORTS:
    if _has(n):
        _export(n, _get(n))

# ==== フォールバック実装（core に無いときだけ定義） ====

# normalize_with_dynamic_alpha が無い場合の簡易互換
if not _has("normalize_with_dynamic_alpha") and _has("normalize_probs_from_odds") and _has("top5_coverage"):
    def _choose_alpha_by_cov5(cov5: float) -> float:
        if cov5 >= 0.75:
            return 1.20
        elif cov5 >= 0.65:
            return 1.10
        elif cov5 >= 0.55:
            return 1.00
        else:
            return 0.90

    def normalize_with_dynamic_alpha(trio_sorted_items, top_n: int = 10):
        """
        trio_sorted_items: (TrioSet, odds) の“オッズ昇順”リスト
        返り値: (pmap, items, alpha_used, cov5_preview)
        """
        p1, items = C.normalize_probs_from_odds(trio_sorted_items, top_n=top_n, alpha=1.0)
        cov5 = C.top5_coverage(p1)
        alpha = _choose_alpha_by_cov5(cov5)
        p2, items = C.normalize_probs_from_odds(trio_sorted_items, top_n=top_n, alpha=alpha)
        return p2, items, alpha, cov5

    _export("normalize_with_dynamic_alpha", normalize_with_dynamic_alpha)

# evaluate_candidates_with_overbet が無い場合の簡易互換
if not _has("evaluate_candidates_with_overbet"):
    # ev_of_band が core にあるならそれを使う
    def evaluate_candidates_with_overbet(
        cands, pmap, tri_odds, base_margin,
        *, overbet_thresh: float = 1.30, overbet_cut: float = 1.50,
        overbet_extra: float = 0.03, long_odds_extra: float = 0.10,
        short_odds_relax: float = 0.00, long_odds_threshold: float = 25.0,
        short_odds_threshold: float = 12.0, max_odds: float = 60.0,
        slippage: float = 0.07
    ):
        """
        最低限の互換（過熱度は参照するが、無ければ課金なしで ev 判定のみ）
        返り値: [(order, p_est, from_set, odds_eval, req, ev, ok, over_ratio)]
        """
        out = []
        # overbet が使えなければ全て1.0扱い
        _pair_over = _get("pair_overbet_ratio") if _has("pair_overbet_ratio") else (lambda pair, *_: 1.0)
        _ev = _get("ev_of_band") if _has("ev_of_band") else None

        for (o, p_est, S) in cands:
            over = _pair_over((o[0], o[1]), pmap, tri_odds)
            # 切り条件
            if overbet_cut and over >= overbet_cut:
                odds_raw = tri_odds.get(o)
                out.append((o, p_est, S, odds_raw, None, None, False, over))
                continue
            extra = overbet_extra if (overbet_thresh and over >= overbet_thresh) else 0.0

            if _ev is not None:
                odds_eval, req, ev, ok = _ev(
                    o, p_est, tri_odds, base_margin,
                    long_odds_extra=long_odds_extra,
                    short_odds_relax=short_odds_relax,
                    long_odds_threshold=long_odds_threshold,
                    short_odds_threshold=short_odds_threshold,
                    max_odds=max_odds,
                    slippage=slippage,
                    extra_margin=extra
                )
            else:
                odds_eval = tri_odds.get(o)
                req = None
                ev = (p_est * odds_eval - 1.0) if (odds_eval and odds_eval > 0) else None
                ok = True if (ev is not None and ev >= 0) else False

            out.append((o, p_est, S, odds_eval, req, ev, ok, over))
        return out

    _export("evaluate_candidates_with_overbet", evaluate_candidates_with_overbet)

# trim_candidates_with_rules が無い場合の簡易互換
if not _has("trim_candidates_with_rules"):
    from collections import Counter
    def trim_candidates_with_rules(
        rows, max_points: int = 8, max_same_pair_points: int = 2
    ):
        ok_rows = [r for r in rows if len(r) >= 7 and r[6] is True and (r[3] is not None)]
        ok_rows_sorted = sorted(ok_rows, key=lambda x: (x[5], x[1] * (x[3] or 0.0)), reverse=True)
        pair_count = Counter()
        out = []
        for r in ok_rows_sorted:
            o = r[0]
            pair = (o[0], o[1])
            if pair_count[pair] >= max_same_pair_points:
                continue
            out.append(r)
            pair_count[pair] += 1
            if len(out) >= max_points:
                break
        return out

    _export("trim_candidates_with_rules", trim_candidates_with_rules)

# allocate_budget_safely が無い場合の簡易互換
if not _has("allocate_budget_safely"):
    def half_kelly_bet_fraction(p_est: float, odds_eval: float) -> float:
        if (odds_eval is None) or (odds_eval <= 1.0):
            return 0.0
        b = odds_eval - 1.0
        f = (p_est - (1.0 - p_est) / b)
        return max(0.0, 0.5 * f)

    def allocate_budget_safely(
        buys, race_cap: int, min_unit: int = 100, per_bet_cap_ratio: float = 0.40
    ):
        if race_cap <= 0 or not buys:
            return [], 0
        wish = []
        for (o, p_est, S, odds_eval, req, ev, ok, over) in buys:
            f = half_kelly_bet_fraction(p_est, odds_eval)
            amt = f * race_cap
            wish.append((o, p_est, S, odds_eval, amt))

        total_wish = sum(max(0.0, x[4]) for x in wish)
        if total_wish <= 0:
            units = race_cap // min_unit
            if units <= 0:
                return [], 0
            unit_each = max(1, units // len(wish))
            out = []
            used_units = 0
            for (o, p, S, od, _) in wish:
                out.append((o, p, S, unit_each * min_unit, od))
                used_units += unit_each
                if used_units >= units:
                    break
            used = sum(x[3] for x in out)
            return out, used

        per_cap = race_cap * per_bet_cap_ratio
        prelim = []
        for (o, p, S, od, amt) in wish:
            amt = min(amt, per_cap)
            units = int(round(amt / min_unit))
            prelim.append((o, p, S, od, units))

        target_units = race_cap // min_unit
        cur_units = sum(u for *_, u in prelim)
        if cur_units == 0:
            prelim = [(o, p, S, od, 1) for (o, p, S, od, u) in prelim]
            cur_units = len(prelim)
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

    _export("allocate_budget_safely", allocate_budget_safely)

# value_ratios_for_pair が無い場合の簡易互換
if not _has("value_ratios_for_pair") and _has("model_prob_trifecta") and _has("market_prob_trifecta"):
    from collections import defaultdict
    def value_ratios_for_pair(head: int, second: int, pmap, tri_odds):
        ptri = C.model_prob_trifecta(pmap)
        qtri = C.market_prob_trifecta(tri_odds)
        exp = defaultdict(float)
        mkt = defaultdict(float)

        # 期待
        for (i, j, k), px in ptri.items():
            if i == head and j == second:
                exp[k] += px
        # 市場
        for (i, j, k), qq in qtri.items():
            if i == head and j == second:
                mkt[k] += qq

        out = []
        for k in range(1, 7):
            ex = exp.get(k, 0.0)
            mk = mkt.get(k, 0.0)
            ratio = (mk / ex) if ex > 0 else float("inf")
            odds = tri_odds.get((head, second, k))
            out.append((k, ex, mk, ratio, odds))
        return out

    _export("value_ratios_for_pair", value_ratios_for_pair)

# should_skip_race_by_entry_filters が無い場合の簡易互換
if not _has("should_skip_race_by_entry_filters") and _has("top5_coverage") and _has("pair_overbet_ratio") and _has("market_head_rate") and _has("pair_mass"):
    def should_skip_race_by_entry_filters(
        pmap_top10, tri_odds,
        *, cov5_min=0.55, cov5_max=0.75, head1_min=0.25, head1_max=0.38, over_pair_drop=1.40
    ):
        cov5 = C.top5_coverage(pmap_top10)
        if not (cov5_min <= cov5 <= cov5_max):
            return True, {"reason": "cov5_out", "cov5": cov5}
        q_head1 = C.market_head_rate(tri_odds, 1)
        if not (head1_min <= q_head1 <= head1_max):
            return True, {"reason": "head1_out", "q_head1": q_head1, "cov5": cov5}
        p_pairs = C.pair_mass(pmap_top10)
        top_pairs = sorted(p_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
        for (i, j), _mass in top_pairs:
            over = C.pair_overbet_ratio((i, j), pmap_top10, tri_odds)
            if over >= over_pair_drop:
                return True, {"reason": "overpair_out", "pair": (i, j), "over": over, "cov5": cov5}
        return False, {"cov5": cov5, "q_head1": q_head1}

    _export("should_skip_race_by_entry_filters", should_skip_race_by_entry_filters)
