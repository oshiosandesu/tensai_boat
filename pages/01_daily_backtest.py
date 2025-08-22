# -*- coding: utf-8 -*-
"""
pages/01_daily_backtest.py
- 指定日・会場（or 全場）で 1R〜12R を走査
- オッズ取得の可否、EV判定の結果（買い/見送り）、例外を可視化
- “発注あり”と“発注なし（見送り理由）”の両方を表示
- CSV保存、累積損益グラフ
"""

from datetime import date
import streamlit as st
import pandas as pd
import altair as alt

# ---- core の取り込み（最新版に合わせる） ----
try:
    from core import (
        VENUES, VENUE_ID2NAME,
        get_trio_odds, get_trifecta_odds, get_trifecta_result,
        normalize_probs_from_odds, build_trifecta_candidates, add_pair_hedge_if_needed,
        top5_coverage, inclusion_mass_for_boat, estimate_head_rate, head_market_rate,
        pair_mass, pair_overbet_ratio, ev_of, allocate_budget_by_prob, choose_R_by_coverage,
        trim_candidates_with_rules
    )
    _has_result_api = True
except ImportError:
    from core import (
        VENUES, VENUE_ID2NAME,
        get_trio_odds, get_trifecta_odds,
        normalize_probs_from_odds, build_trifecta_candidates, add_pair_hedge_if_needed,
        top5_coverage, inclusion_mass_for_boat, estimate_head_rate, head_market_rate,
        pair_mass, pair_overbet_ratio, ev_of, allocate_budget_by_prob, choose_R_by_coverage,
        trim_candidates_with_rules
    )
    get_trifecta_result = None
    _has_result_api = False

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

    st.divider()
    relax = st.checkbox("EV閾値を少し緩めて再評価（+5ppを無効化）", value=False,
                        help="候補が多すぎる時の絞り込みで余裕%を+5ppする処理を止め、やや緩めに採用します。")
    show_all_table = st.checkbox("発注なしのレースも明細に表示", value=True)

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
            # ---- オッズ取得 ----
            trio_odds, _ = get_trio_odds(d, vid, rno)
            tri_odds = get_trifecta_odds(d, vid, rno)

            if not trio_odds or not tri_odds:
                rows.append({
                    "date": d.strftime("%Y%m%d"), "venue": vname, "venue_id": vid, "rno": rno,
                    "status": "no-odds", "reason": "オッズ未取得（未公開/休催/エラー）",
                    "bet": 0, "payout": 0, "pnl": 0, "hit": 0, "n_points": 0, "mode": ""
                })
                done += 1
                progress.progress(done/total_jobs)
                continue

            # ---- 確率化（Top10）→ R決定 → 候補生成 ----
            trio_sorted = sorted(trio_odds.items(), key=lambda x: x[1])
            pmap_top10, _ = normalize_probs_from_odds(trio_sorted, top_n=10, alpha=1.0)
            R, _ = choose_R_by_coverage(pmap_top10)
            cands = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2)

            if add_hedge:
                mass_pairs = pair_mass(pmap_top10)
                top_pairs = sorted(mass_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
                cands = add_pair_hedge_if_needed(cands, pmap_top10, top_pairs, max_extra=1)

            # ---- EVチェック ----
            ok_rows = []
            for (o, p_est, S) in cands:
                odds, req, ev, ok = ev_of(o, p_est, tri_odds, margin=margin)
                if ok:
                    ok_rows.append((o, p_est, S, odds, req, ev, True))

            # ---- 点数絞り ----
            trimmed = trim_candidates_with_rules(
                ok_rows, max_points=max_points, max_same_pair_points=same_pair_max
            )

            # 候補多すぎ時の余裕%+5pp（緩める設定ならスキップ）
            if (not relax) and len(trimmed) > max_points:
                margin2 = min(0.30, margin + 0.05)
                ok2 = []
                for (o, p_est, S, _, _, _, _) in ok_rows:
                    odds, req, ev, ok = ev_of(o, p_est, tri_odds, margin=margin2)
                    if ok:
                        ok2.append((o, p_est, S, odds, req, ev, True))
                trimmed = trim_candidates_with_rules(ok2, max_points=max_points, max_same_pair_points=same_pair_max)

            # ---- 資金配分 ----
            buys_input = [(o, p, S) for (o, p, S, *_ ) in trimmed]
            bets, used = allocate_budget_by_prob(buys_input, race_cap, min_unit=min_unit)
            bet_map = {o: b for (o, p, S, b) in bets}

            # ---- 結果 or EVフォールバック ----
            if used == 0:
                rows.append({
                    "date": d.strftime("%Y%m%d"),
                    "venue": vname, "venue_id": vid, "rno": rno,
                    "status": "no-buy", "reason": "EV未達（割に合う買い目なし）",
                    "bet": 0, "payout": 0, "pnl": 0, "hit": 0, "n_points": 0, "mode": ""
                })
            else:
                if _has_result_api:
                    res = get_trifecta_result(d, vid, rno)
                else:
                    res = None
                if res:
                    (win_order, win_odds) = res
                    hit_amt = bet_map.get(win_order, 0)
                    payout = int(round(hit_amt * tri_odds.get(win_order, win_odds)))
                    hit = 1 if hit_amt > 0 else 0
                    mode = "real"
                else:
                    # 期待値の払い戻し（参考）
                    payout = 0
                    hit = 0
                    for (o, p, S, odds, req, ev, ok) in trimmed:
                        payout += int(round(bet_map.get(o, 0) * (p * tri_odds.get(o, odds))))
                    mode = "ev"

                pnl = payout - used
                rows.append({
                    "date": d.strftime("%Y%m%d"),
                    "venue": vname, "venue_id": vid, "rno": rno,
                    "status": "buy", "reason": "",
                    "bet": used, "payout": payout, "pnl": pnl,
                    "hit": hit, "n_points": len(trimmed), "mode": mode
                })

        except Exception as e:
            rows.append({
                "date": d.strftime("%Y%m%d"), "venue": vname, "venue_id": vid, "rno": rno,
                "status": "error", "reason": f"{e}", "bet": 0, "payout": 0, "pnl": 0, "hit": 0, "n_points": 0, "mode": ""
            })
        finally:
            done += 1
            progress.progress(done/total_jobs)

# ---- 集計・可視化 ----
df = pd.DataFrame(rows)

# サマリーバッジ
n_total = len(df)
n_no_odds = int((df["status"] == "no-odds").sum())
n_no_buy = int((df["status"] == "no-buy").sum())
n_error = int((df["status"] == "error").sum())
n_buy = int((df["status"] == "buy").sum())
st.markdown(
    f"**処理サマリ**　"
    f"発注あり: **{n_buy}**　/　"
    f"見送り(EV未達): **{n_no_buy}**　/　"
    f"オッズ未取得: **{n_no_odds}**　/　"
    f"エラー: **{n_error}**　/　"
    f"合計: **{n_total}**"
)

# 発注あり（集計）
df_buy = df[df["status"] == "buy"].copy()
if not df_buy.empty:
    df_buy["cum_pnl"] = df_buy["pnl"].cumsum()
    total_bet = int(df_buy["bet"].sum())
    total_payout = int(df_buy["payout"].sum())
    total_pnl = int(df_buy["pnl"].sum())
    hits = int(df_buy["hit"].sum())
    n = int(len(df_buy))
    roi = (total_payout / total_bet) if total_bet > 0 else 0.0

    st.subheader(f"集計（{vtitle} / {d.strftime('%Y-%m-%d')}）")
    left, right = st.columns(2)
    with left:
        st.metric("総投下", f"{total_bet} 円")
        st.metric("総払戻", f"{total_payout} 円")
        st.metric("損益", f"{total_pnl:+,} 円")
    with right:
        st.metric("的中数（実結果ベース）", f"{hits} / {n}")
        st.metric("ROI", f"{roi:.2f}")

    st.subheader("累積損益")
    chart = alt.Chart(df_buy).mark_line().encode(
        x=alt.X("rno:Q", title="R（通し）"),
        y=alt.Y("cum_pnl:Q", title="累積損益（円）"),
        tooltip=["venue","rno","bet","payout","pnl","cum_pnl","mode"]
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("明細（発注あり）")
    st.dataframe(
        df_buy[["venue","rno","n_points","bet","payout","pnl","hit","mode"]],
        use_container_width=True
    )
    st.download_button(
        "明細（発注あり）を保存（CSV）",
        df_buy.to_csv(index=False).encode("utf-8"),
        file_name=f"bt_buy_{d.strftime('%Y%m%d')}_{vtitle}.csv",
        mime="text/csv"
    )
else:
    st.warning("この条件では『発注あり』のレースがありませんでした。サイドバーで余裕%を下げる/点数上限を増やす/保険ONなどをお試しください。")

# 発注なし（診断目的の一覧）
if show_all_table:
    df_nobuy = df[df["status"].isin(["no-buy", "no-odds", "error"])].copy()
    if not df_nobuy.empty:
        st.subheader("明細（発注なし）")
        st.dataframe(
            df_nobuy[["venue","rno","status","reason"]],
            use_container_width=True
        )
        st.download_button(
            "明細（発注なし）を保存（CSV）",
            df_nobuy.to_csv(index=False).encode("utf-8"),
            file_name=f"bt_nobuy_{d.strftime('%Y%m%d')}_{vtitle}.csv",
            mime="text/csv"
        )
