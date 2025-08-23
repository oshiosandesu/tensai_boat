# -*- coding: utf-8 -*-
import streamlit as st
from datetime import date
import time
from typing import Dict, Tuple, List

from core import (
    VENUES, VENUE_ID2NAME, venues_on,
    get_trio_odds, get_trifecta_odds, get_trifecta_result,
    normalize_with_dynamic_alpha, top5_coverage, choose_R_by_coverage,
    build_trifecta_candidates, evaluate_candidates_basic, trim_candidates, allocate_budget
)

st.set_page_config(page_title="⚡ 日次バックテスト（安定版）", layout="wide")

st.title("⚡ 日次バックテスト（安定版・オッズのみ）")

with st.sidebar:
    st.header("実行条件")
    d = st.date_input("対象日", value=date.today(), format="YYYY-MM-DD")
    only_active = st.checkbox("本日開催の会場のみで実行", value=True)
    st.caption("※ “本日開催のみ”をONにすると取得が速く安定します。")

    st.header("買い方・資金（共通）")
    bank = st.number_input("バンク（円）", min_value=0, value=100_000, step=10_000)
    race_cap = st.number_input("1レース上限（円）", min_value=0, value=600, step=100)
    min_unit = st.number_input("最小購入単位（円）", min_value=100, value=100, step=100)
    margin_pct = st.slider("安全マージン（+%）", min_value=0, max_value=20, value=10, step=1)
    max_points = st.slider("候補上限（点）", min_value=2, max_value=15, value=8, step=1)
    max_same_pair = st.slider("同一(頭-2着)ペア上限", min_value=1, max_value=4, value=2, step=1)

    run = st.button("この条件でバックテスト実行", type="primary")

st.caption("※ このページは確定結果がある日付向けです（オッズのみ使用）。")

def run_one_race(d, vid, rno, race_cap, min_unit, margin_pct, max_points, max_same_pair):
    try:
        trio_odds, _ = get_trio_odds(d, vid, rno)
        tri_odds = get_trifecta_odds(d, vid, rno)
        if not trio_odds or not tri_odds:
            return {"status": "no_odds"}

        # 3複Top10
        trio_sorted = sorted(trio_odds.items(), key=lambda x: x[1])
        pmap, items, alpha_used, cov5_preview = normalize_with_dynamic_alpha(trio_sorted, top_n=10)

        # 候補→EV→トリミング→配分
        R, _ = choose_R_by_coverage(pmap)
        cands = build_trifecta_candidates(pmap, R, cov5_hint=top5_coverage(pmap))
        rows = evaluate_candidates_basic(cands, tri_odds, base_margin=margin_pct/100.0)
        buys = trim_candidates(rows, max_points=max_points, max_same_pair_points=max_same_pair)

        if not buys:
            return {"status": "no_buy"}

        allocs, used = allocate_budget(buys, race_cap=race_cap, min_unit=min_unit, per_bet_cap_ratio=0.40)

        # 結果取得
        res = get_trifecta_result(d, vid, rno)
        if not res:
            # 未確定の場合は集計不能
            return {
                "status": "bought_noresult",
                "used": used,
                "buys": buys,
                "allocs": allocs
            }
        (win_order, win_odds) = res
        # 払戻計算
        pay = 0
        hit = False
        for (o, p, S, bet_yen, odds_eval) in allocs:
            if o == win_order:
                hit = True
                pay += int(bet_yen * win_odds)  # 100円あたりオッズ→金額
        pl = pay - used
        return {
            "status": "done",
            "used": used,
            "pay": pay,
            "pl": pl,
            "hit": hit,
            "win": f"{win_order[0]}-{win_order[1]}-{win_order[2]}",
            "win_odds": win_odds,
            "buys": buys,
            "allocs": allocs,
        }
    except Exception as e:
        return {"status": "error", "error": repr(e)}

if run:
    if only_active:
        vids = venues_on(d)
        if not vids:
            st.warning("有効な開催場が見つかりませんでした。日付を変更するか、“本日開催のみ”をOFFにしてください。")
            st.stop()
    else:
        vids = [vid for vid, _ in VENUES]

    st.write("対象:", ", ".join([f"{VENUE_ID2NAME[v]}" for v in vids]))

    total_used = 0
    total_pay = 0
    total_pl = 0
    total_buy = 0
    total_hit = 0
    total_error = 0
    total_no_odds = 0
    total_no_buy = 0

    progress = st.progress(0)
    steps = len(vids) * 12
    done = 0

    results: List[Dict] = []
    for vid in vids:
        for rno in range(1, 13):
            r = run_one_race(d, vid, rno, race_cap, min_unit, margin_pct, max_points, max_same_pair)
            results.append((vid, rno, r))
            done += 1
            progress.progress(min(1.0, done/steps))
            time.sleep(0.02)  # 負荷を少し下げる

            st.write(f"{d.strftime('%Y-%m-%d')}　{VENUE_ID2NAME[vid]}　{rno}R … {r.get('status')}")

            if r["status"] == "done":
                total_used += r["used"]
                total_pay += r["pay"]
                total_pl += r["pl"]
                total_buy += 1
                if r["hit"]:
                    total_hit += 1
            elif r["status"] == "error":
                total_error += 1
            elif r["status"] == "no_odds":
                total_no_odds += 1
            elif r["status"] == "no_buy":
                total_no_buy += 1

    st.subheader("集計結果")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("総投下", f"{total_used:,} 円")
    col2.metric("総払戻", f"{total_pay:,} 円")
    col3.metric("損益", f"{total_pl:,} 円")
    roi = (total_pay / total_used) if total_used > 0 else 0.0
    col4.metric("ROI", f"{roi:.2f}")
    col5.metric("的中数", f"{total_hit}/{total_buy}")
    col6.metric("エラー", f"{total_error}")

    st.caption(f"見送り(EV未達)={total_no_buy} / オッズ未取得={total_no_odds} / 合計={len(results)}")
