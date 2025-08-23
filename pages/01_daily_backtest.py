# -*- coding: utf-8 -*-
"""
pages/01_daily_backtest.py — 日次バックテスト
- 当日開催場だけを回して ROI を集計
- app.py と同じロジックで意思決定 → 実結果と突き合わせ
- 並列なし（まずは正しさ重視）。必要なら ThreadPoolExecutor 化も容易。
"""
# --- ここを pages/01_daily_backtest.py の import より前に差し込む ---
import sys
from pathlib import Path
APP_DIR = Path(__file__).resolve().parents[1]   # pages の 1 つ上 = app.py の場所
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
# --------------------------------------------------------------------

from datetime import date
import pandas as pd
import streamlit as st

from core_bridge import (
    VENUES, VENUE_ID2NAME, venues_on,
    get_trio_odds, get_trifecta_odds, get_trifecta_result,
    normalize_with_dynamic_alpha, top5_coverage, inclusion_mass_for_boat,
    pair_mass, estimate_head_rate, choose_R_by_coverage,
    build_trifecta_candidates, add_pair_hedge_if_needed,
    market_head_rate, pair_overbet_ratio,
    evaluate_candidates_with_overbet, trim_candidates_with_rules, allocate_budget_safely,
)

# ===== ページ設定 =====
st.set_page_config(page_title="日次バックテスト", page_icon="📊", layout="wide")
st.title("📊 日次バックテスト（オッズベース）")
st.caption("app.py と同じ判定ロジックで日単位のROIを検証します。")

# ===== サイドバー =====
with st.sidebar:
    today = date.today()
    d = st.date_input("対象日", value=today, format="YYYY-MM-DD")

    only_active = st.checkbox("開催会場のみを対象にする（推奨）", value=True)
    venue_choices = [f"{vid:02d} - {name}" for vid, name in VENUES]
    sel_all = st.multiselect("対象会場（空なら上の設定に従う）", venue_choices, default=[])

    r_from = st.number_input("開始R", min_value=1, max_value=12, value=1, step=1)
    r_to   = st.number_input("終了R", min_value=1, max_value=12, value=12, step=1)

    st.divider()
    st.header("売買ルール（app.py と同一）")
    race_cap = st.number_input("1レース上限（円）", min_value=100, value=600, step=100)
    min_unit = st.number_input("最小購入単位（円）", min_value=100, value=100, step=100)
    margin_pct = st.slider("基本の余裕％", 0, 30, 10, 1)
    margin = margin_pct / 100.0
    max_candidates = st.slider("候補の最大点数", 4, 10, 8, 1)
    add_hedge = st.checkbox("保険（条件付きで1点追加）", value=True)

    st.divider()
    do_run = st.button("この条件で回す", type="primary", use_container_width=True)

# ===== 実行 =====
if not do_run:
    st.info("左の条件を設定して「この条件で回す」を押してください。")
    st.stop()

# 対象会場の決定
if sel_all:
    target_vids = [int(x.split(" - ")[0]) for x in sel_all]
else:
    target_vids = venues_on(d) if only_active else [vid for vid, _ in VENUES]

if not target_vids:
    st.warning("対象会場が見つかりません。日付や開催可否の判定を確認してください。")
    st.stop()

st.write(f"対象日: **{d.strftime('%Y-%m-%d')}** / 対象会場: {len(target_vids)} / R: {r_from}〜{r_to}")

# 集計バッファ
rows = []
total_bet = 0
total_return = 0

prog = st.progress(0)
done = 0
total_tasks = len(target_vids) * max(0, r_to - r_from + 1)

for vid in target_vids:
    vname = VENUE_ID2NAME.get(vid, f"場{vid}")
    for rno in range(r_from, r_to + 1):
        done += 1
        prog.progress(min(1.0, done / max(1, total_tasks)))

        try:
            trio_odds, update_tag = get_trio_odds(d, vid, rno)
            if not trio_odds:
                continue
            trifecta_odds = get_trifecta_odds(d, vid, rno)
            if not trifecta_odds:
                continue

            # 確率化（動的α）
            trio_sorted = sorted(trio_odds.items(), key=lambda x: x[1])
            pmap_top10, top10_items, alpha_used, cov5_preview = normalize_with_dynamic_alpha(trio_sorted, top_n=10)

            # 指標
            cov5 = top5_coverage(pmap_top10)
            inc1 = inclusion_mass_for_boat(pmap_top10, 1)
            head1_est = estimate_head_rate(pmap_top10, head=1)
            head1_mkt = market_head_rate(trifecta_odds, head=1)
            mass_pairs = pair_mass(pmap_top10)
            top_pairs = sorted(mass_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
            R, _label_cov = choose_R_by_coverage(pmap_top10)

            # 候補生成（プレビュー用は省略し、本番候補のみ作成）
            base_cands = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2, cov5_hint=cov5)

            # 条件付きヘッジ（堅くて過熱気味の時だけ）
            primary_pair = top_pairs[0][0] if top_pairs else (1, 2)
            over_ratio = pair_overbet_ratio(primary_pair, pmap_top10, trifecta_odds)
            if add_hedge and (cov5 >= 0.68 and over_ratio >= 1.12):
                base_cands = add_pair_hedge_if_needed(base_cands, pmap_top10, top_pairs, max_extra=1)

            # 判定（過熱課金・帯で余裕%出し分け・スリッページ）
            judged = evaluate_candidates_with_overbet(
                base_cands, pmap_top10, trifecta_odds, base_margin=margin,
                overbet_thresh=1.30, overbet_cut=1.50, overbet_extra=0.03,
                long_odds_extra=0.10, short_odds_relax=0.00,
                long_odds_threshold=25.0, short_odds_threshold=12.0,
                max_odds=60.0, slippage=0.07
            )

            # トリミング
            trimmed = trim_candidates_with_rules(judged, max_points=max_candidates, max_same_pair_points=2)

            # 多すぎる場合、余裕+5pp で再判定
            if len(trimmed) > max_candidates:
                judged2 = evaluate_candidates_with_overbet(
                    base_cands, pmap_top10, trifecta_odds, base_margin=min(0.30, margin + 0.05),
                    overbet_thresh=1.30, overbet_cut=1.50, overbet_extra=0.03,
                    long_odds_extra=0.10, short_odds_relax=0.00,
                    long_odds_threshold=25.0, short_odds_threshold=12.0,
                    max_odds=60.0, slippage=0.07
                )
                trimmed = trim_candidates_with_rules(judged2, max_points=max_candidates, max_same_pair_points=2)

            # 配分（半ケリー＋上限）
            bets, used = allocate_budget_safely(trimmed, race_cap, min_unit=min_unit, per_bet_cap_ratio=0.40)
            used_amt = sum(b for (_o, _p, _S, b, _od) in bets)

            # 結果取得
            res = get_trifecta_result(d, vid, rno)
            hit = False
            ret_amt = 0
            if res:
                (win_o, win_odds) = res
                # 的中金額 = (当該並びの購入額 / 100) * 払戻金額（= 100×オッズ）
                # ここでは “購入額 × オッズ” として評価（100円単位の扱いは各自の約定仕様に依存）
                for (o, p, S, bet_yen, od_eval) in bets:
                    if o == win_o:
                        hit = True
                        ret_amt += int(round(bet_yen * win_odds))
            total_bet += used_amt
            total_return += ret_amt

            rows.append({
                "date": d.strftime("%Y%m%d"),
                "venue_id": vid, "venue": vname, "rno": rno,
                "cov5": cov5, "inc1": inc1, "head1_est": head1_est, "head1_mkt": head1_mkt,
                "alpha_used": alpha_used,
                "n_bets": len(bets), "bet_total": used_amt, "return_total": ret_amt,
                "hit": hit
            })

        except Exception as e:
            rows.append({
                "date": d.strftime("%Y%m%d"),
                "venue_id": vid, "venue": vname, "rno": rno,
                "error": str(e)
            })
            continue

# ===== 結果表示 =====
df = pd.DataFrame(rows)
if df.empty:
    st.warning("結果が空でした。対象日のデータ有無をご確認ください。")
    st.stop()

c1, c2, c3 = st.columns(3)
bet_sum = int(df["bet_total"].fillna(0).sum())
ret_sum = int(df["return_total"].fillna(0).sum())
roi = (ret_sum / bet_sum) if bet_sum > 0 else 0.0
hit_rate = (df["hit"].fillna(False).mean()) if "hit" in df else 0.0

c1.metric("総購入", f"{bet_sum:,} 円")
c2.metric("総払戻", f"{ret_sum:,} 円")
c3.metric("ROI", f"{roi:.2f} 倍")

st.metric("的中率（レース単位）", f"{hit_rate:.1%}")

st.subheader("レース別サマリ")
show_cols = ["date", "venue", "rno", "n_bets", "bet_total", "return_total", "hit", "cov5", "head1_mkt", "head1_est", "alpha_used"]
st.dataframe(df[show_cols].sort_values(["venue", "rno"]).reset_index(drop=True), use_container_width=True)

st.subheader("ダウンロード")
st.download_button("全明細（CSV）", df.to_csv(index=False).encode("utf-8"), file_name=f"daily_backtest_{d.strftime('%Y%m%d')}.csv", mime="text/csv")
