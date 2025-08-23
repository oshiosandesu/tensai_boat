# -*- coding: utf-8 -*-
import streamlit as st
from datetime import date, datetime
from typing import List, Tuple
from core import (
    VENUES, VENUE_ID2NAME, get_trio_odds, get_trifecta_odds,
    normalize_with_dynamic_alpha, top5_coverage, inclusion_mass_for_boat,
    estimate_head_rate, market_head_rate, pair_mass, coverage_milestones,
    choose_R_by_coverage, build_trifecta_candidates,
    evaluate_candidates_basic, trim_candidates, allocate_budget
)

st.set_page_config(page_title="⛵ 単レース診断（オッズのみ）", layout="wide")

st.title("⛵ 単レース診断（わかりやすい表示・オッズのみ）")

with st.sidebar:
    st.header("レース選択")
    d = st.date_input("開催日", value=date.today(), format="YYYY-MM-DD")
    venue_name_to_id = {name: vid for vid, name in VENUES}
    vname = st.selectbox("開催場", options=[name for _, name in VENUES])
    vid = venue_name_to_id[vname]
    rno = st.selectbox("レース番号", options=list(range(1, 13)), index=0)

    st.header("買い方・資金")
    bank = st.number_input("バンク（円）", min_value=0, value=100_000, step=10_000)
    race_cap = st.number_input("1レース上限（円）", min_value=0, value=600, step=100)
    min_unit = st.number_input("最小購入単位（円）", min_value=100, value=100, step=100)
    margin_pct = st.slider("安全マージン（+%）", min_value=0, max_value=20, value=10, step=1)
    max_points = st.slider("候補上限（点）", min_value=2, max_value=15, value=8, step=1)
    max_same_pair = st.slider("同一(頭-2着)ペア上限", min_value=1, max_value=4, value=2, step=1)

    run = st.button("この条件で診断する", type="primary")

st.caption("※ このページはオッズのみで指標・買い目を出します（成績・展示は未使用）")

if run:
    with st.spinner("オッズ取得中..."):
        trio_odds, update_tag = get_trio_odds(d, vid, rno)
        tri_odds = get_trifecta_odds(d, vid, rno)

    if not trio_odds or not tri_odds:
        st.error("オッズが取得できませんでした（発売前/回線混雑/開催なしの可能性）。")
        st.stop()

    # 3連複Top10（オッズ昇順）
    trio_sorted = sorted(trio_odds.items(), key=lambda x: x[1])
    pmap, items, alpha_used, cov5_preview = normalize_with_dynamic_alpha(trio_sorted, top_n=10)

    # 指標
    cov5 = top5_coverage(pmap)
    mass1 = inclusion_mass_for_boat(pmap, 1)
    head1_est = estimate_head_rate(pmap, head=1)
    head1_mkt = market_head_rate(tri_odds, head=1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("堅さ（上位5組の合計）", f"{cov5*100:.1f}%")
    col2.metric("1号艇の絡みやすさ（Top10）", f"{mass1*100:.1f}%")
    col3.metric("1号艇の1着見込み（モデル）", f"{head1_est*100:.1f}%")
    col4.metric("1号艇の人気集中（市場）", f"{head1_mkt*100:.1f}%")

    st.caption(f"更新タグ: {update_tag or '不明'} / 開催: {d.strftime('%Y-%m-%d')} {VENUE_ID2NAME[vid]} {rno}R / α={alpha_used:.2f}")

    # ペア強度
    st.subheader("ペア強度（3連複Top10から）")
    pm = pair_mass(pmap)
    top_pairs = sorted(pm.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_pairs:
        st.table({
            "ペア": [f"{i}-{j}" for (i, j), _ in top_pairs],
            "同舟券になりやすさ（推定）": [f"{mass*100:.1f}%" for _, mass in top_pairs],
        })

    # カバレッジ
    st.subheader("3連複カバレッジ（上位から貪欲に積む）")
    cov = coverage_milestones(pmap)
    c25 = ", ".join([f"{a}-{b}-{c}" for (a,b,c), _ in cov["25"]]) or "—"
    c50 = ", ".join([f"{a}-{b}-{c}" for (a,b,c), _ in cov["50"]]) or "—"
    c75 = ", ".join([f"{a}-{b}-{c}" for (a,b,c), _ in cov["75"]]) or "—"
    st.markdown(f"**25% 到達**: {c25}")
    st.markdown(f"**50% 到達**: {c50}")
    st.markdown(f"**75% 到達**: {c75}")

    # 候補生成 → EV判定 → トリミング
    R, note = choose_R_by_coverage(pmap)
    st.caption(f"採用集合 R = {R}（{note}）")

    cands = build_trifecta_candidates(pmap, R, cov5_hint=cov5)
    rows = evaluate_candidates_basic(cands, tri_odds, base_margin=margin_pct/100.0)
    buys = trim_candidates(rows, max_points=max_points, max_same_pair_points=max_same_pair)

    # プレビュー：投資を考えず当てやすさ（候補pの合算）
    st.subheader("当てやすさプレビュー（投資は考えない）")
    preview_hit = sum(x[1] for x in cands[:max_points])
    st.metric("想定的中率（プレビュー）", f"{preview_hit*100:.1f}%")

    # 最終の買い目（資金配分）
    st.subheader("おトク重視の買い目（最終）")
    if not buys:
        st.info("EV条件を満たす買い目がありませんでした。マージン/点数上限/ペア上限を調整してみてください。")
    else:
        allocs, used = allocate_budget(buys, race_cap=race_cap, min_unit=min_unit, per_bet_cap_ratio=0.40)
        # 表示
        import pandas as pd
        df = pd.DataFrame([{
            "買い目": f"{o[0]}-{o[1]}-{o[2]}",
            "推定ヒット率": f"{p*100:.2f}%",
            "オッズ": f"{odds:.1f}倍",
            "必要ヒット率": f"{req*100:.2f}%",
            "期待値": f"{ev*100:.1f}%",
            "購入金額": int(next((a for (oo, pp, S, a, od) in allocs if oo==o), 0)),
        } for (o, p, S, odds, req, ev, ok) in buys])
        st.dataframe(df, use_container_width=True)
        st.caption(f"合計購入: {used} 円 / 上限: {race_cap} 円")
