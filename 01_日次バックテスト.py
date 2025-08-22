# -*- coding: utf-8 -*-
"""
pages/01_日次バックテスト.py
- 指定日・会場（or 全場）で 1R〜12R を機械的に走査
- 現行ロジックで「買うならコレ（割に合うものだけ）」を算出
- 実配当（取得できる場合）で損益を集計。取れない場合は“EVフォールバック”
- ROI・的中率・累積残高を可視化（CSV保存付き）
"""

from datetime import date
import streamlit as st
import pandas as pd
import altair as alt

from core import (
    VENUES, VENUE_ID2NAME,
    get_trio_odds, get_trifecta_odds, get_trifecta_result,
    normalize_probs_from_odds, build_trifecta_candidates, add_pair_hedge_if_needed,
    top5_coverage, inclusion_mass_for_boat, estimate_head_rate, head_market_rate,
    pair_mass, pair_overbet_ratio, ev_of, allocate_budget_by_prob, choose_R_by_coverage,
    trim_candidates_with_rules
)

st.set_page_config(page_title="日次バックテスト", page_icon="📈", layout="wide")
st.title("📈 日次バックテスト（1〜12R）")

with st.sidebar:
    today = date.today()
    d = st.date_input("対象日", value=today, format="YYYY-MM-DD")
    venue_names = ["全場"] + [f"{vid:02d} - {name}" for vid, name in VENUES]
    vsel = st.selectbox("開催場", venue_names, index=0)
    if vsel == "全場":
        venue_ids = [vid for vid, _ in VENUES]
        vtitle = "全場"
    else:
        vid = int(vsel.split(" - ")[0])
        venue_ids = [vid]
        vtitle = VENUE_ID2NAME.get(vid, f"場{vid}")

    st.divider()
    st.header("買い方パラメータ")
    race_cap = st.number_input("1レース上限（円）", min_value=100, value=600, step=100)
    min_unit = st.number_input("最小購入単位（円）", min_value=100, value=100, step=100)
    margin_pct = st.slider("余裕（%）", 0, 30, 10, 1)
    margin = margin_pct / 100.0
    add_hedge = st.checkbox("保険を1点足す", value=True)
    max_points = st.slider("点数上限（自動絞り）", 4, 12, 8, 1)
    same_pair_max = st.slider("同一ペア上限（頭-2着）", 1, 3, 2, 1)

    do_run = st.button("この条件でバックテスト", type="primary", use_container_width=True)

if not do_run:
    st.info("左の条件を設定して「この条件でバックテスト」を押してください。")
    st.stop()

rows = []
progress = st.progress(0.0)
total_jobs = len(venue_ids) * 12
done = 0

for vid in venue_ids:
    vname = VENUE_ID2NAME.get(vid, f"場{vid}")
    st.markdown(f"### {d.strftime('%Y-%m-%d')}　{vname}　1〜12R")
    for rno in range(1, 13):
        try:
            trio_odds, _ = get_trio_odds(d, vid, rno)
            tri_odds = get_trifecta_odds(d, vid, rno)
            if not trio_odds or not tri_odds:
                # オッズ取れない場合はスキップ
                rows.append({
                    "date": d.strftime("%Y%m%d"), "venue": vname, "rno": rno,
                    "status": "no-odds", "bet": 0, "payout": 0, "pnl": 0, "hit": 0
                })
                done += 1
                progress.progress(done/total_jobs)
                continue

            # 確率化（Top10）
            trio_sorted = sorted(trio_odds.items(), key=lambda x: x[1])
            pmap_top10, _ = normalize_probs_from_odds(trio_sorted, top_n=10, alpha=1.0)

            # R決定・候補生成
            R, _ = choose_R_by_coverage(pmap_top10)
            cands = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2)
            if add_hedge:
                mass_pairs = pair_mass(pmap_top10)
                top_pairs = sorted(mass_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
                cands = add_pair_hedge_if_needed(cands, pmap_top10, top_pairs, max_extra=1)

            # EVチェック
            ok_rows = []
            for (o, p_est, S) in cands:
                odds, req, ev, ok = ev_of(o, p_est, tri_odds, margin=margin)
                if ok:
                    ok_rows.append((o, p_est, S, odds, req, ev, True))
            # 絞り
            ok_rows = trim_candidates_with_rules(ok_rows, max_points=max_points, max_same_pair_points=same_pair_max)

            # 資金配分
            buys_input = [(o, p, S) for (o,p,S,odds,req,ev,ok) in ok_rows]
            bets, used = allocate_budget_by_prob(buys_input, race_cap, min_unit=min_unit)
            bet_map = {o: b for (o,p,S,b) in bets}

            # 実結果（取れない場合は EVフォールバック）
            res = get_trifecta_result(d, vid, rno)
            if res:
                (win_order, win_odds) = res
                hit_amt = bet_map.get(win_order, 0)
                payout = int(round(hit_amt * tri_odds.get(win_order, win_odds)))
                hit = 1 if hit_amt > 0 else 0
            else:
                # EVフォールバック（期待値払い戻し）
                payout = 0
                hit = 0
                for (o,p,S,odds,req,ev,ok) in ok_rows:
                    payout += int(round(bet_map.get(o,0) * (p * tri_odds.get(o,odds))))
                # 期待値なのでヒット数は 0 のまま（参考として扱う）

            pnl = payout - used
            rows.append({
                "date": d.strftime("%Y%m%d"),
                "venue": vname, "venue_id": vid, "rno": rno,
                "bet": used, "payout": payout, "pnl": pnl,
                "n_points": len(ok_rows), "hit": hit, "mode": "real" if res else "ev"
            })
        except Exception as e:
            rows.append({
                "date": d.strftime("%Y%m%d"), "venue": vname, "rno": rno,
                "status": f"err:{e}", "bet": 0, "payout": 0, "pnl": 0, "hit": 0
            })
        finally:
            done += 1
            progress.progress(done/total_jobs)

# 集計
df = pd.DataFrame(rows)
df_ok = df[df["bet"] > 0].copy()
if df_ok.empty:
    st.warning("有効なレースがありませんでした。")
    st.stop()

df_ok["cum_pnl"] = df_ok["pnl"].cumsum()
df_ok["roi"] = (df_ok["payout"] / df_ok["bet"]).fillna(0.0)
df_ok["hit_cum"] = df_ok["hit"].cumsum()

st.subheader(f"集計（{vtitle} / {d.strftime('%Y-%m-%d')}）")
left, right = st.columns(2)
with left:
    total_bet = int(df_ok["bet"].sum())
    total_payout = int(df_ok["payout"].sum())
    total_pnl = int(df_ok["pnl"].sum())
    st.metric("総投下", f"{total_bet} 円")
    st.metric("総払戻", f"{total_payout} 円")
    st.metric("損益", f"{total_pnl:+,} 円")
with right:
    hits = int(df_ok["hit"].sum())
    n = int(len(df_ok))
    st.metric("的中数（実結果ベース）", f"{hits} / {n}")
    roi = (total_payout / total_bet) if total_bet>0 else 0.0
    st.metric("ROI", f"{roi:.2f}")

st.subheader("累積損益")
chart = alt.Chart(df_ok).mark_line().encode(
    x=alt.X("rno:Q", title="R（通し）"),
    y=alt.Y("cum_pnl:Q", title="累積損益（円）"),
    tooltip=["venue","rno","bet","payout","pnl","cum_pnl"]
).properties(height=280)
st.altair_chart(chart, use_container_width=True)

st.subheader("明細（CSV保存可）")
st.dataframe(df_ok[["venue","rno","n_points","bet","payout","pnl","hit","mode"]],
             use_container_width=True)
st.download_button(
    "明細を保存（CSV）",
    df_ok.to_csv(index=False).encode("utf-8"),
    file_name=f"bt_{d.strftime('%Y%m%d')}_{vtitle}.csv",
    mime="text/csv"
)
