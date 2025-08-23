# -*- coding: utf-8 -*-
"""
01_daily_backtest.py
- 指定日の全会場・全レースを走査してバックテスト
- ROIや的中率を集計
- 処理時間短縮のため開催中の会場だけ選べる
"""

# --- パス補正 ---
import sys
from pathlib import Path
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from datetime import date
import streamlit as st
import pandas as pd
import time

from core_bridge import (
    VENUES, VENUE_ID2NAME,
    get_trio_odds, get_trifecta_odds, get_trifecta_result,
    normalize_probs_from_odds, top5_coverage, pair_mass,
    estimate_head_rate, market_head_rate,
    choose_R_by_coverage, build_trifecta_candidates, add_pair_hedge_if_needed,
    evaluate_candidates_with_overbet, trim_candidates_with_rules, allocate_budget_safely,
)

# ---------- ページ設定 ----------
st.set_page_config(page_title="日次バックテスト", page_icon="📊", layout="wide")

st.title("📊 日次バックテスト")
st.caption("指定日の全会場・全レースを対象に ROI を検証します。")

# ---------- サイドバー ----------
with st.sidebar:
    st.header("設定")
    today = date.today()
    d = st.date_input("開催日", value=today, format="YYYY-MM-DD")
    race_cap = st.number_input("1レースあたりの上限（円）", min_value=100, value=600, step=100)
    min_unit = st.number_input("最小購入単位（円）", min_value=100, value=100, step=100)
    margin_pct = st.slider("余裕（%）", 0, 30, 10, 1)
    margin = margin_pct / 100.0
    max_candidates = st.slider("候補の最大点数", 4, 10, 8, 1)
    add_hedge = st.checkbox("保険を1点足す", value=True)
    do_run = st.button("この条件でバックテスト", type="primary", use_container_width=True)

# ---------- 実行 ----------
if do_run:
    st.markdown(f"## {d.strftime('%Y-%m-%d')} のバックテスト開始")

    results = []
    total_bet = 0
    total_return = 0
    total_hit = 0
    total_cnt = 0

    progress = st.progress(0)
    venues = VENUES
    vlen = len(venues)

    for vi, (vid, vname) in enumerate(venues, start=1):
        for rno in range(1, 13):
            try:
                # オッズ取得
                trio_odds, _ = get_trio_odds(d, vid, rno)
                trifecta_odds = get_trifecta_odds(d, vid, rno)

                # 確率化
                trio_sorted = sorted(trio_odds.items(), key=lambda x: x[1])
                pmap_top10, top10_items = normalize_probs_from_odds(trio_sorted, top_n=10, alpha=1.0)

                # R決定
                R, _ = choose_R_by_coverage(pmap_top10)

                # 候補生成
                base_cands = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2)
                if add_hedge:
                    mass_pairs = pair_mass(pmap_top10)
                    top_pairs = sorted(mass_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
                    base_cands = add_pair_hedge_if_needed(base_cands, pmap_top10, top_pairs, max_extra=1)

                # EV評価
                judged = evaluate_candidates_with_overbet(
                    base_cands, pmap_top10, trifecta_odds, base_margin=margin
                )

                # 点数絞り
                trimmed = trim_candidates_with_rules(judged, max_points=max_candidates, max_same_pair_points=2)

                # 資金配分
                bets, used = allocate_budget_safely(trimmed, race_cap, min_unit=min_unit)
                bet_map = {o: b for (o, p, S, b, od) in bets}

                # 払戻計算
                hit = False
                payout = 0
                res = get_trifecta_result(d, vid, rno)
                if res:
                    res_tuple = tuple(res)
                    if res_tuple in bet_map:
                        hit = True
                        payout = bet_map[res_tuple] * trifecta_odds.get(res_tuple, 0)

                total_bet += sum(bet_map.values())
                total_return += payout
                total_hit += 1 if hit else 0
                total_cnt += 1

                results.append({
                    "date": d.strftime("%Y%m%d"),
                    "venue": vname,
                    "rno": rno,
                    "bet": sum(bet_map.values()),
                    "return": payout,
                    "hit": hit
                })

            except Exception as e:
                results.append({
                    "date": d.strftime("%Y%m%d"),
                    "venue": vname,
                    "rno": rno,
                    "error": str(e)
                })

        progress.progress(vi / vlen)

    # 集計
    roi = (total_return / total_bet) if total_bet > 0 else 0
    hit_rate = (total_hit / total_cnt) if total_cnt > 0 else 0

    st.subheader("結果サマリー")
    c1, c2, c3 = st.columns(3)
    c1.metric("総投資額", f"{total_bet:,} 円")
    c2.metric("総払戻", f"{total_return:,} 円")
    c3.metric("ROI", f"{roi:.2f} 倍")

    st.metric("的中率", f"{hit_rate:.1%} （{total_hit}/{total_cnt}R）")

    st.subheader("レースごとの結果")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    st.download_button("結果をCSVで保存", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"backtest_{d.strftime('%Y%m%d')}.csv", mime="text/csv")
else:
    st.info("左の条件を設定して「この条件でバックテスト」を押してください。")
