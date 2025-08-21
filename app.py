# -*- coding: utf-8 -*-
"""
app.py
Streamlit版：単レース可視化を“ユーザー言語”で。DB不要・CSV保存あり。
- レースの“形”をカード/グラフで直感表示（堅さ、1頭人気の過熱、ペア強度、25/50/75%）
- “紐荒れ”ヒント（選択ペアの3着分布：期待 vs 市場）
- 「当てにいく / バランス / 一撃狙い」のプリセット（投資を離れた当てやすさプレビュー）
- 最終の「買い目（おトク重視）」はEVでふるい、必要オッズも提示
"""

import json
from datetime import date

import streamlit as st
import altair as alt
import pandas as pd

from core import (
    VENUES, VENUE_ID2NAME,
    get_trio_odds, get_trifecta_odds,
    normalize_probs_from_odds, top5_coverage, inclusion_mass_for_boat,
    pair_mass, estimate_head_rate, choose_R_by_coverage,
    coverage_targets, build_trifecta_candidates, add_pair_hedge_if_needed,
    head_market_rate, pair_overbet_ratio, value_ratios_for_pair,
    ev_of, allocate_budget_by_prob
)

# ---------- ページ設定 ----------
st.set_page_config(
    page_title="ボートレース：単レース可視化",
    page_icon="⛵",
    layout="wide"
)

# ---------- 軽いスタイル ----------
st.markdown("""
<style>
.small { font-size: 0.92rem; color: #666; }
.big { font-size: 1.1rem; font-weight: 700; }
.metric-ok { color:#0a7a0a; font-weight:600; }
.metric-warn { color:#a66f00; font-weight:600; }
.metric-bad { color:#b00020; font-weight:600; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; background:#eef; margin-right:6px; }
.badge-strong { background:#e8f5e9; }
.badge-mid { background:#fff8e1; }
.badge-weak { background:#ffebee; }
.tbl th, .tbl td { font-size: 0.95rem; }
.hint { background:#f7f7ff; padding:8px 10px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.title("⛵ 単レース可視化（わかりやすい表示版）")
st.caption("当てやすさの見える化＋おトク重視の買い目。DB不要、ローカル実行。")

# ---------- サイドバー（入力） ----------
with st.sidebar:
    st.header("レース選択")
    today = date.today()
    d = st.date_input("開催日", value=today, format="YYYY-MM-DD")
    venue_display = [f"{vid:02d} - {name}" for vid, name in VENUES]
    vsel = st.selectbox("開催場", venue_display, index=len(VENUES)-1)  # 既定：最後=大村
    vid = int(vsel.split(" - ")[0])
    vname = VENUE_ID2NAME.get(vid, f"場{vid}")

    st.write("レース番号")
    # ボタン風のセグメント：Streamlit 1.31+ の segmented_control 相当を radio で代用
    rno = st.radio(" ", list(range(1, 13)), index=7, horizontal=True, label_visibility="collapsed")

    st.divider()
    st.header("買い方プリセット（投資を離れて当てやすさを見る）")
    preset = st.radio(" ",
        options=["🟢 当てにいく", "🟡 バランス", "🔴 一撃狙い"],
        index=1, horizontal=True, label_visibility="collapsed"
    )
    # 当てにいく の目標（50% / 75%）を選べる
    if preset == "🟢 当てにいく":
        target_cover = st.radio("狙う当たりやすさ", ["50% 目標", "75% 目標"], index=0, horizontal=True)
    else:
        target_cover = None

    st.divider()
    st.header("最終の“おトク重視”設定")
    race_cap = st.number_input("1レース上限（円）", min_value=100, value=600, step=100)
    min_unit = st.number_input("最小購入単位（円）", min_value=100, value=100, step=100)
    margin_pct = st.slider("安全マージン（+%）", min_value=0, max_value=30, value=10, step=1)
    margin = margin_pct / 100.0
    anti_mode = st.radio("アンチ本命ペア（自動判定）",
                         ["オフ（M0）", "自動・部分外し（M1）", "自動・強外し（M2）"], horizontal=True)
    add_hedge = st.checkbox("足りないペアを1点だけ補強（ヘッジ）", value=True)
    max_candidates = st.slider("候補上限（点）", min_value=4, max_value=10, value=6, step=1)

    st.divider()
    do_run = st.button("この条件で可視化する", type="primary")

# ---------- キャッシュ付きの取得 ----------
@st.cache_data(ttl=60, show_spinner=False)
def cached_trio(d, vid, rno):
    return get_trio_odds(d, vid, rno)

@st.cache_data(ttl=60, show_spinner=False)
def cached_trifecta(d, vid, rno):
    return get_trifecta_odds(d, vid, rno)

# ---------- 実行 ----------
if do_run:
    with st.spinner("公式オッズを取得中…"):
        trio_odds, update_tag = cached_trio(d, vid, rno)
        trifecta_odds = cached_trifecta(d, vid, rno)
        if not trio_odds or not trifecta_odds:
            st.error("オッズが取得できません。時間をおいて再試行してください。")
            st.stop()

    # 3複Top10正規化
    trio_sorted = sorted(trio_odds.items(), key=lambda x: x[1])
    pmap_top10, top10_items = normalize_probs_from_odds(trio_sorted, top_n=10)
    ssum = sum((1.0/x for _, x in top10_items)) or 1.0

    # 指標
    cov5 = top5_coverage(pmap_top10)                         # 堅さ
    inc1 = inclusion_mass_for_boat(pmap_top10, 1)            # 1含有率
    head1_est = estimate_head_rate(pmap_top10, head=1)       # 1頭の当たり見込み
    head1_mkt = head_market_rate(trifecta_odds, head=1)      # 市場の1頭人気
    mass_pairs = pair_mass(pmap_top10)
    top_pairs = sorted(mass_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
    R, label_cov = choose_R_by_coverage(pmap_top10)
    cov_targets = coverage_targets(pmap_top10, (0.25, 0.50, 0.75))

    # ---- 上段：レース診断カード ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("堅さ（上位5組の合計）", f"{cov5:.1%}", help="高いほど堅め／低いほど荒れ気味")
    c2.metric("1の絡みやすさ（Top10）", f"{inc1:.1%}", help="3連複Top10における1番の含有率")
    c3.metric("1の1着見込み", f"{head1_est:.1%}")
    diff = (head1_mkt / head1_est) if head1_est > 0 else 0.0
    delta_txt = f"人気の偏り：市場/見込み = {diff:.2f}×"
    c4.metric("1の人気集中（市場）", f"{head1_mkt:.1%}", delta=delta_txt)

    # 一言コメント
    comments = []
    if cov5 >= 0.70 and head1_est >= 0.60:
        comments.append("頭は固め。紐は広めに拾うのが無難。")
    elif cov5 <= 0.55:
        comments.append("やや荒れ気配。点数は絞って妙味サイドへ。")
    if head1_est > 0 and diff >= 1.15:
        comments.append("1の人気がやや過熱。1-◯-◯の主要順序は買い控え。")
    elif head1_est > 0 and diff <= 0.90:
        comments.append("1の人気は控えめ。1軸の妙味が出やすい。")
    if comments:
        st.markdown(f"<div class='hint'>{'　'.join(comments)}</div>", unsafe_allow_html=True)

    st.caption(f"更新タグ: {update_tag or '不明'}　/　開催: {d.strftime('%Y-%m-%d')} {vname} {rno}R")

    # ---- ペア強度 Top3（棒） ----
    st.subheader("ペア強度 Top3（3連複Top10から算出）")
    df_pairs = pd.DataFrame([{"ペア": f"{i}-{j}", "強度": m} for (i,j), m in top_pairs])
    chart_pairs = alt.Chart(df_pairs).mark_bar().encode(
        x=alt.X("ペア:N", title="ペア"),
        y=alt.Y("強度:Q", title="強度", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("強度:Q", title="強度", format=".1%")]
    ).properties(height=180)
    st.altair_chart(chart_pairs, use_container_width=True)

    # ---- 3連複カバレッジ（25/50/75%） ----
    st.subheader("3連複カバレッジ（上位から貪欲に積む）")
    chips = []
    for tval in (0.25, 0.50, 0.75):
        k, picks = cov_targets[tval]
        sets = ", ".join(
            f"{min(S)}={sorted(list(S))[1]}={max(S)}" for S, _ in picks[:k]
        )
        chips.append((int(tval*100), k, sets))
    cols = st.columns(3)
    for (pct, k, sets), col in zip(chips, cols):
        col.markdown(f"<span class='badge badge-strong'>{pct}% 到達</span> 上位{k}組", unsafe_allow_html=True)
        col.write(sets if sets else "-")

    # ---- 「当てやすい買い方」プリセット（投資を離れてプレビュー） ----
    st.subheader("当てやすさプレビュー（投資は考えない）")
    # 上位集合から順序へ展開（最上位順序は回避）
    base_preview = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2)
    # プリセット別フィルタ
    preview = base_preview[:]
    primary_pair = top_pairs[0][0] if top_pairs else (1, 2)
    over_ratio = pair_overbet_ratio(primary_pair, pmap_top10, trifecta_odds, beta=1.0)

    def is_pair_head2(o, pair):
        i, j = pair
        return (o[0], o[1]) == (i, j) or (o[0], o[1]) == (j, i)

    if preset == "🟢 当てにいく":
        # 目標カバレッジを満たすまで集合を増やす（Rを動的拡張）
        target = 0.50 if (target_cover == "50% 目標") else 0.75
        # 必要なら R を広げる（最大6程度まで）
        Rt = R
        items = sorted(pmap_top10.items(), key=lambda x: x[1], reverse=True)
        acc = sum(v for _, v in items[:Rt])
        while acc < target and Rt < min(10, len(items)):
            Rt += 1
            acc = sum(v for _, v in items[:Rt])
        preview = build_trifecta_candidates(pmap_top10, R=Rt, avoid_top=True, max_per_set=2)
        preview = preview[: min(12, len(preview))]  # 表示点数は控えめ
        hit_est = sum(p for _, p, _ in preview)
        st.metric("想定的中率（プレビュー）", f"{hit_est:.1%}")
        st.caption(f"上位{Rt}組から順序展開（最上位順序は除外）")

    elif preset == "🟡 バランス":
        # 過熱ペアは軽く回避（M1に近い）
        if (over_ratio > 1.10 and cov5 <= 0.65):
            preview = [x for x in preview if not is_pair_head2(x[0], primary_pair)]
            st.caption("過熱ペアの頭-2着順序を軽く回避（M1相当）")
        hit_est = sum(p for _, p, _ in preview[:10])
        st.metric("想定的中率（プレビュー）", f"{hit_est:.1%}")

    else:  # 🔴 一撃狙い
        # 過熱ペアを強く回避（M2に近い）＋点数を控える
        if (over_ratio > 1.15 and cov5 <= 0.55):
            preview = [x for x in preview if not is_pair_head2(x[0], primary_pair)]
            st.caption("過熱ペアの頭-2着順序を強く回避（M2相当）")
        preview = preview[:6]
        hit_est = sum(p for _, p, _ in preview)
        st.metric("想定的中率（プレビュー）", f"{hit_est:.1%}")

    # 簡易テーブル（買い目と当たり見込みだけ）
    if preview:
        df_prev = pd.DataFrame([{"買い目": f"{o[0]}-{o[1]}-{o[2]}", "当たり見込み": p} for (o, p, _) in preview])
        st.dataframe(df_prev.style.format({"当たり見込み":"{:.2%}"}), use_container_width=True)

    # ---- “紐荒れ”ヒント（選択ペアの3着分布：期待 vs 市場） ----
    with st.expander("“紐荒れ”ヒント（開いて計算）", expanded=False):
        opt_order = st.radio("順序を選択", [f"{primary_pair[0]}-{primary_pair[1]}", f"{primary_pair[1]}-{primary_pair[0]}"],
                             horizontal=True, index=0)
        head, second = [int(x) for x in opt_order.split("-")]
        vr = value_ratios_for_pair(head, second, pmap_top10, trifecta_odds)
        df_vr = pd.DataFrame([
            {"3着": k, "期待": ex, "市場": mk, "市場/期待": r, "オッズ": odds} for (k, ex, mk, r, odds) in vr
        ])
        cL, cR = st.columns([1.2, 1.2])
        chart_vr = alt.Chart(df_vr).mark_bar().encode(
            x=alt.X("3着:N", title="3着"),
            y=alt.Y("市場/期待:Q", title="市場/期待（低い=妙味）"),
            tooltip=["3着", alt.Tooltip("期待:Q", format=".1%"), alt.Tooltip("市場:Q", format=".1%"), alt.Tooltip("市場/期待:Q", format=".2f"), "オッズ"]
        ).properties(height=220)
        cL.altair_chart(chart_vr, use_container_width=True)

        chart_vr2 = alt.Chart(df_vr.melt(id_vars=["3着"], value_vars=["期待","市場"], var_name="種別", value_name="p")).mark_bar().encode(
            x=alt.X("3着:N"),
            y=alt.Y("p:Q", axis=alt.Axis(format="%")),
            color=alt.Color("種別:N"),
            column=alt.Column("種別:N", header=alt.Header(title=None))
        ).properties(height=220)
        cR.altair_chart(chart_vr2, use_container_width=True)

    st.divider()

    # ---- 最終：おトク重視の買い目（EV判定＋配分） ----
    st.subheader("おトク重視の買い目（最終）")

    # 候補（Rは堅さに応じて）
    base_cands = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2)
    if add_hedge:
        base_cands = add_pair_hedge_if_needed(base_cands, pmap_top10, top_pairs, max_extra=1)
    cands = base_cands[:max_candidates]

    # アンチ本命の自動発動条件
    def is_pair_head2(o, pair):
        i, j = pair
        return (o[0], o[1]) == (i, j) or (o[0], o[1]) == (j, i)

    def need_partial_filter():
        return (over_ratio > 1.10 and cov5 <= 0.65)

    def need_strong_filter():
        return (over_ratio > 1.15 and cov5 <= 0.55)

    if anti_mode == "自動・部分外し（M1）" and need_partial_filter():
        cands = [x for x in cands if not is_pair_head2(x[0], primary_pair)]
        st.warning("M1発動：過熱ペアの“頭-2着”順序を除外しました。")
    elif anti_mode == "自動・強外し（M2）" and need_strong_filter():
        cands = [x for x in cands if not is_pair_head2(x[0], primary_pair)]
        st.error("M2発動：過熱ペアの“頭-2着”順序を完全除外しました。")

    # EVチェック
    ok_rows, ng_rows = [], []
    for (o, p_est, S) in cands:
        odds, req, ev, ok = ev_of(o, p_est, trifecta_odds, margin=margin)
        (ok_rows if ok else ng_rows).append((o, p_est, S, odds, req, ev, ok))

    # 必要オッズ（ = (1+margin)/p_est ）を列として追加するための関数
    def needed_odds(p_est, margin):
        return (1.0 + margin) / p_est if p_est > 0 else None

    # 配分（OKのみ）
    buys_input = [(o, p, S) for (o, p, S, *_ ) in ok_rows]
    bets, used = allocate_budget_by_prob(buys_input, race_cap, min_unit=min_unit)
    buy_map = {o: b for (o, p, S, b) in bets}

    # OKテーブル
    if ok_rows:
        df_ok = pd.DataFrame([
            {
                "買い目": f"{o[0]}-{o[1]}-{o[2]}",
                "由来セット": f"{min(S)}={sorted(list(S))[1]}={max(S)}",
                "当たり見込み": p_est,
                "オッズ": odds,
                "必要ライン": req,
                "おトク度": ev,
                "必要オッズ": needed_odds(p_est, margin),
                "購入": buy_map.get(o, 0)
            }
            for (o, p_est, S, odds, req, ev, ok) in ok_rows
        ])
        df_ok = df_ok[["買い目","由来セット","当たり見込み","オッズ","必要オッズ","必要ライン","おトク度","購入"]]
        st.dataframe(
            df_ok.style
                .format({"当たり見込み":"{:.2%}","必要ライン":"{:.2%}","おトク度":"{:+.1%}","必要オッズ":"{:.2f}倍"})
                .background_gradient(subset=["おトク度"], cmap="Greens"),
            use_container_width=True
        )
    else:
        st.info("おトク度の条件を満たす買い目がありません。")

    # NGテーブル（参考）
    with st.expander("参考：見送り候補（おトク度NG）"):
        if ng_rows:
            df_ng = pd.DataFrame([
                {
                    "買い目": f"{o[0]}-{o[1]}-{o[2]}",
                    "由来セット": f"{min(S)}={sorted(list(S))[1]}={max(S)}",
                    "当たり見込み": p_est,
                    "オッズ": odds,
                    "必要オッズ": needed_odds(p_est, margin),
                    "必要ライン": req,
                    "おトク度": ev
                }
                for (o, p_est, S, odds, req, ev, ok) in ng_rows
            ])
            df_ng = df_ng[["買い目","由来セット","当たり見込み","オッズ","必要オッズ","必要ライン","おトク度"]]
            st.dataframe(df_ng.style.format({"当たり見込み":"{:.2%}","必要ライン":"{:.2%}","おトク度":"{:+.1%}","必要オッズ":"{:.2f}倍"}),
                         use_container_width=True)
        else:
            st.write("（なし）")

    # サマリー
    hit_rate_est = sum(p for (o, p, *_ ) in [(o,p,S) for (o,p,S,odds,req,ev,ok) in ok_rows])
    total_bet = sum(buy_map.get(o,0) for (o, *_ ) in [(o,p,S,odds,req,ev,ok) for (o,p,S,odds,req,ev,ok) in ok_rows])
    cA, cB, cC = st.columns(3)
    cA.metric("想定的中率（OK合算）", f"{hit_rate_est:.1%}")
    cB.metric("合計購入", f"{total_bet} 円")
    cC.metric("レース上限", f"{race_cap} 円")

    # ダウンロード（1レース分）
    top10_list = []
    acc = 0.0
    for (S, odd) in top10_items:
        p = (1.0/odd)/ssum; acc += p
        a,b,c = sorted(list(S))
        top10_list.append({"set": f"{a}={b}={c}", "odds": float(odd), "p": float(p), "cum": float(acc)})

    race_record = {
        "date": d.strftime("%Y%m%d"), "venue": vname, "rno": rno, "odds_update": update_tag or "",
        "top10_list": json.dumps(top10_list, ensure_ascii=False),
        "top5_coverage": round(cov5,6), "inc_mass_1": round(inc1,6), "head1_est": round(head1_est,6),
        "pair_mass_top3": json.dumps([{"pair": f"({i},{j})", "mass": float(m)} for (i,j), m in top_pairs], ensure_ascii=False),
        "R": R, "max_candidates": len(cands), "hedge_used": 1 if add_hedge else 0,
        "race_cap": race_cap, "margin": margin
    }
    ticket_records = []
    for (o, p_est, S, odds, req, ev, ok) in ok_rows:
        ticket_records.append({
            "date": d.strftime("%Y%m%d"), "venue": vname, "rno": rno,
            "selection": f"{o[0]}-{o[1]}-{o[2]}",
            "from_set": f"{min(S)}={sorted(list(S))[1]}={max(S)}",
            "p_est": round(p_est,6), "odds_close": odds,
            "req_p": round(req,6), "ev_est": round(ev,6),
            "needed_odds": round(((1.0+margin)/p_est), 6) if p_est>0 else None,
            "bet_amount": buy_map.get(o,0)
        })

    race_df = pd.DataFrame([race_record])
    ticket_df = pd.DataFrame(ticket_records)
    c1, c2 = st.columns(2)
    c1.download_button("race_level（このレース）CSVを保存", race_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"race_{d.strftime('%Y%m%d')}_{vname}_{rno}.csv", mime="text/csv")
    c2.download_button("ticket_level（このレース）CSVを保存", ticket_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"ticket_{d.strftime('%Y%m%d')}_{vname}_{rno}.csv", mime="text/csv")

else:
    st.info("左の条件を設定して「この条件で可視化する」を押してください。")
