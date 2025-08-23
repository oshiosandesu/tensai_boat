# -*- coding: utf-8 -*-
"""
core_bridge.py — import 互換レイヤー（衝突に強い版）
- まず 'core' を試し、無ければ 'boatcore' を読む
- 見つかったコアから必要関数を re-export
- 無い関数はフォールバックで補う
"""
from __future__ import annotations
from importlib import import_module

# ---- どのコアを使うか決定（core.py 優先。無ければ boatcore.py） ----
_last_err = None
C = None
for modname in ("core", "boatcore"):
    try:
        C = import_module(modname)
        CORE_MODULE_NAME = modname
        break
    except Exception as e:
        _last_err = e
if C is None:
    raise ImportError(
        "Neither 'core.py' nor 'boatcore.py' could be imported. "
        "Place one of them next to app.py. Last error: %r" % _last_err
    )

def _has(name: str) -> bool:
    return hasattr(C, name)

def _get(name: str):
    return getattr(C, name)

def _export(name: str, obj):
    globals()[name] = obj

BASIC_EXPORTS = [
    # 取得系
    "get_trio_odds", "get_trifecta_odds", "get_trifecta_result", "get_just_before_info",
    # 会場
    "VENUES", "VENUE_ID2NAME", "venues_on", "is_race_available",
    # 確率化・指標
    "normalize_probs_from_odds", "normalize_with_dynamic_alpha",
    "top5_coverage", "inclusion_mass_for_boat", "pair_mass", "estimate_head_rate",
    # モデル/市場 確率
    "model_prob_trifecta", "market_prob_trifecta",
    "model_head_rate", "market_head_rate", "market_pair_rate",
    # 価値/過熱
    "value_ratio_tri", "pair_overbet_ratio", "head_overbet_ratio",
    # 候補生成・R決定
    "choose_R_by_coverage", "build_trifecta_candidates", "add_pair_hedge_if_needed",
    # EV/評価
    "evaluate_candidates_with_overbet", "ev_of_band", "adjust_for_slippage",
    # トリミング・配分
    "trim_candidates_with_rules", "allocate_budget_safely",
    # ヒント
    "value_ratios_for_pair",
]
for n in BASIC_EXPORTS:
    if _has(n):
        _export(n, _get(n))

# --------- フォールバック実装（無い時だけ定義） ----------
if not _has("normalize_with_dynamic_alpha") and _has("normalize_probs_from_odds") and _has("top5_coverage"):
    def _choose_alpha_by_cov5(cov5: float) -> float:
        if cov5 >= 0.75: return 1.20
        if cov5 >= 0.65: return 1.10
        if cov5 >= 0.55: return 1.00
        return 0.90
    def normalize_with_dynamic_alpha(trio_sorted_items, top_n: int = 10):
        p1, items = C.normalize_probs_from_odds(trio_sorted_items, top_n=top_n, alpha=1.0)
        cov5 = C.top5_coverage(p1)
        alpha = _choose_alpha_by_cov5(cov5)
        p2, items = C.normalize_probs_from_odds(trio_sorted_items, top_n=top_n, alpha=alpha)
        return p2, items, alpha, cov5
    _export("normalize_with_dynamic_alpha", normalize_with_dynamic_alpha)

if not _has("evaluate_candidates_with_overbet"):
    def evaluate_candidates_with_overbet(
        cands, pmap, tri_odds, base_margin, *,
        overbet_thresh: float = 1.30, overbet_cut: float = 1.50,
        overbet_extra: float = 0.03, long_odds_extra: float = 0.10,
        short_odds_relax: float = 0.00, long_odds_threshold: float = 25.0,
        short_odds_threshold: float = 12.0, max_odds: float = 60.0, slippage: float = 0.07
    ):
        out = []
        _pair_over = _get("pair_overbet_ratio") if _has("pair_overbet_ratio") else (lambda pair, *_: 1.0)
        _ev = _get("ev_of_band") if _has("ev_of_band") else None
        for (o, p_est, S) in cands:
            over = _pair_over((o[0], o[1]), pmap, tri_odds)
            if overbet_cut and over >= overbet_cut:
                odds_raw = tri_odds.get(o)
                out.append((o, p_est, S, odds_raw, None, None, False, over))
                continue
            extra = overbet_extra if (overbet_thresh and over >= overbet_thresh) else 0.0
            if _ev is not None:
                odds_eval, req, ev, ok = _ev(
                    o, p_est, tri_odds, base_margin,
                    long_odds_extra=long_odds_extra, short_odds_relax=short_odds_relax,
                    long_odds_threshold=long_odds_threshold, short_odds_threshold=short_odds_threshold,
                    max_odds=max_odds, slippage=slippage, extra_margin=extra
                )
            else:
                odds_eval = tri_odds.get(o); req = None
                ev = (p_est * odds_eval - 1.0) if odds_eval else None
                ok = bool(ev is not None and ev >= 0)
            out.append((o, p_est, S, odds_eval, req, ev, ok, over))
        return out
    _export("evaluate_candidates_with_overbet", evaluate_candidates_with_overbet)

if not _has("trim_candidates_with_rules"):
    from collections import Counter
    def trim_candidates_with_rules(rows, max_points: int = 8, max_same_pair_points: int = 2):
        ok_rows = [r for r in rows if len(r) >= 7 and r[6] is True and (r[3] is not None)]
        ok_rows_sorted = sorted(ok_rows, key=lambda x: (x[5], x[1] * (x[3] or 0.0)), reverse=True)
        pair_count = Counter(); out = []
        for r in ok_rows_sorted:
            o = r[0]; pair = (o[0], o[1])
            if pair_count[pair] >= max_same_pair_points: continue
            out.append(r); pair_count[pair] += 1
            if len(out) >= max_points: break
        return out
    _export("trim_candidates_with_rules", trim_candidates_with_rules)

if not _has("allocate_budget_safely"):
    def half_kelly_bet_fraction(p_est: float, odds_eval: float) -> float:
        if (odds_eval is None) or (odds_eval <= 1.0): return 0.0
        b = odds_eval - 1.0; f = (p_est - (1.0 - p_est) / b)
        return max(0.0, 0.5 * f)
    def allocate_budget_safely(buys, race_cap: int, min_unit: int = 100, per_bet_cap_ratio: float = 0.40):
        if race_cap <= 0 or not buys: return [], 0
        wish = []
        for (o, p_est, S, odds_eval, req, ev, ok, over) in buys:
            f = half_kelly_bet_fraction(p_est, odds_eval); amt = f * race_cap
            wish.append((o, p_est, S, odds_eval, amt))
        total_wish = sum(max(0.0, x[4]) for x in wish)
        if total_wish <= 0:
            units = race_cap // min_unit
            if units <= 0: return [], 0
            unit_each = max(1, units // len(wish)); out = []; used_units = 0
            for (o, p, S, od, _) in wish:
                out.append((o, p, S, unit_each * min_unit, od))
                used_units += unit_each
                if used_units >= units: break
            used = sum(x[3] for x in out); return out, used
        per_cap = race_cap * per_bet_cap_ratio
        prelim = []
        for (o, p, S, od, amt) in wish:
            amt = min(amt, per_cap); units = int(round(amt / min_unit))
            prelim.append((o, p, S, od, units))
        target_units = race_cap // min_unit
        cur_units = sum(u for *_, u in prelim)
        if cur_units == 0:
            prelim = [(o, p, S, od, 1) for (o, p, S, od, u) in prelim]; cur_units = len(prelim)
        while cur_units > target_units and prelim:
            idx = min(range(len(prelim)), key=lambda i: prelim[i][1] if prelim[i][4] > 0 else 1e9)
            o, p, S, od, u = prelim[idx]
            if u > 0: prelim[idx] = (o, p, S, od, u - 1); cur_units -= 1
            else: break
        while cur_units < target_units:
            idx = max(range(len(prelim)), key=lambda i: prelim[i][1])
            o, p, S, od, u = prelim[idx]; prelim[idx] = (o, p, S, od, u + 1); cur_units += 1
        out = [(o, p, S, u * min_unit, od) for (o, p, S, od, u) in prelim if u > 0]
        used = sum(x[3] for x in out); return out, used
    _export("allocate_budget_safely", allocate_budget_safely)
