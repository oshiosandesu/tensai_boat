# -*- coding: utf-8 -*-
"""
app.py
スマホ向け・白ベース・日本語UI最適化版
- 見出し&用語：誰でも直感で使える表現に統一
- 「レースのようす」→「レースの傾向」に変更（ご要望反映）
- 的中目標/予想の方針/最小購入単位 など、ラベルと小さな説明を徹底
- スマホ前提: layout="centered"、2列×2行の指標カード、簡潔な表とグラフ
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

# ---------- ページ設定（スマホ想定） ----------
st.set_page_config(
    page_title="レース診断（1レース）",
    page_icon="⛵",
    layout="centered"   # スマホで見やすい縦積みレイアウト
)

# ---------- シンプルCSS（白ベース＋小説明） ----------
st.markdown("""
<style>
/* ===== グローバル（白ベースで文字は濃く） ===== */
html, body, [data-testid="stAppViewContainer"] * {
  color: #111111 !important;
}
[data-testid="stAppViewContainer"] {
  background: #FFFFFF !important;
}
[data-testid="stHeader"] {
  background: #FFFFFF !important;
}
a, a:visited {
  color: #0E7AFE !important;
  text-decoration: none;
}

/* ===== セクションの薄背景など ===== */
.hint {
  background:#f7f7ff;
  padding:10px 12px;
  border-radius:10px;
}

/* ===== 小さな説明・見出し装飾 ===== */
.small { font-size: 0.92rem; color: #444 !important; }
.big   { font-size: 1.08rem; font-weight: 700; color:#111 !important; }

.badge { display:inline-block; padding:2px 10px; border-radius:999px; background:#eef; margin-right:6px; color:#111 !important; }
.badge-strong { background:#e8f5e9; }

/* ===== 表（モバイル読みやすさ） ===== */
.tbl th, .tbl td { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)
st.title("⛵ レース診断（1レース）")
st.caption("このレースの“全体像”を、人気の偏りと当てやすさで見える化します。")

# ---------- サイドバー（入力：スマホでも操作しやすい配置） ----------
with st.sidebar:
    st.header("レース選択")
    today = date.today()
    d = st.date_input("開催日", value=today, format="YYYY-MM-DD")
    st.caption("例：2025-08-22")

    venue_display = [f"{vid:02d} - {name}" for vid, name in VENUES]
    vsel = st.selectbox("開催場", venue_display, index=len(VENUES)-1)
    vid = int(vsel.split(" - ")[0])
    vname = VENUE_ID2NAME.get(vid, f"場{vid}")
    st.caption("検索できます")

    st.write("レース番号")
    rno = st.radio(" ", list(range(1, 13)), index=7, horizontal=True, label_visibility="collapsed")
    st.caption("1〜12をボタンで選択")

    st.divider()
    st.header("予想の方針（お試し）")
    preset = st.radio(" ",
        options=["🟢 当たり重視", "🟡 ほどよく", "🔴 高配当狙い"],
        index=1, horizontal=True, label_visibility="collapsed"
    )
    if preset == "🟢 当たり重視":
        target_cover = st.radio("的中目標", ["50% 目標", "75% 目標"], index=0, horizontal=True)
        st.caption("当てにいくときの目安。どこまで押さえるかの参考です。")
    else:
        target_cover = None

    st.divider()
    st.header("最終の買い方（割に合うものだけ採用）")
    race_cap = st.number_input("このレースに使う上限（円）", min_value=100, value=600, step=100)
    st.caption("最終的に“買う”と決めた場合の合計上限です。")

    min_unit = st.number_input("最小購入単位（円）", min_value=100, value=100, step=100)

    margin_pct = st.slider("余裕（%）", min_value=0, max_value=30, value=10, step=1)
    margin = margin_pct / 100.0
    st.caption("数値を上げるほど条件が厳しくなり、本当に割に合うものだけ残ります（目安 5〜15%）。")

    anti_str = st.radio("本命に偏った並びを避ける",
                        ["使わない", "少し避ける", "だいぶ避ける"], horizontal=True)
    add_hedge = st.checkbox("保険を1点足す", value=True)
    st.caption("上位ペアが候補に無ければ1点だけ追加して“取りこぼし”を減らします。")

    max_candidates = st.slider("候補の最大点数", min_value=4, max_value=10, value=6, step=1)

    st.divider()
    do_run = st.button("この条件で診断する", type="primary", use_container_width=True)

# ---------- キャッシュ取得 ----------
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
            st.error("オッズを取得できませんでした。時間をおいてやり直してください。")
            st.stop()

    # 3複Top10 正規化
    trio_sorted = sorted(trio_odds.items(), key=lambda x: x[1])
    pmap_top10, top10_items = normalize_probs_from_odds(trio_sorted, top_n=10)
    ssum = sum((1.0/x for _, x in top10_items)) or 1.0

    # 指標
    cov5 = top5_coverage(pmap_top10)                         # 固さ
    inc1 = inclusion_mass_for_boat(pmap_top10, 1)            # 1号艇含有率
    head1_est = estimate_head_rate(pmap_top10, head=1)       # 1号艇の1着見込み
    head1_mkt = head_market_rate(trifecta_odds, head=1)      # 市場の1号艇人気
    mass_pairs = pair_mass(pmap_top10)
    top_pairs = sorted(mass_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
    R, label_cov = choose_R_by_coverage(pmap_top10)
    cov_targets = coverage_targets(pmap_top10, (0.25, 0.50, 0.75))

    # ===== レースの傾向（上段カード：2列×2行） =====
    st.subheader("レースの傾向")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("レースの固さ", f"{cov5:.1%}")
        st.caption("上位5組の合計。高い=本命寄り／低い=荒れ気味（目安: 70%↑=固い）")
    with c2:
        st.metric("1号艇の絡みやすさ", f"{inc1:.1%}")
        st.caption("3連複Top10に1号艇が含まれる割合。")

    c3, c4 = st.columns(2)
    with c3:
        st.metric("1号艇が1着になりやすさ", f"{head1_est:.1%}")
        st.caption("推定。あくまで目安。")
    with c4:
        diff = (head1_mkt / head1_est) if head1_est > 0 else 0.0
        st.metric("1号艇への人気の集まり", f"{head1_mkt:.1%}", delta=f"市場/見込み = {diff:.2f}×")
        st.caption("市場の人気が推定より強い/弱いかの目安。")

    # 一言コメント（状況に応じて複数表示）
    comments = []
    # 固さ
    if cov5 >= 0.70:
        comments.append("本命寄りで固め。紐は広めに拾うのが無難。")
    elif cov5 >= 0.60:
        comments.append("やや本命寄り。的中目標50%が現実的。")
    elif cov5 >= 0.55:
        comments.append("中庸。妙味サイドも混ぜるとバランス良し。")
    else:
        comments.append("やや荒れ気配。点数は絞って妙味を優先。")
    # 1号艇の人気偏り
    if head1_est > 0 and (head1_mkt / head1_est) >= 1.15:
        comments.append("1号艇が買われ過ぎ。1-◯-◯の本命順序は控えめに。")
    elif head1_est > 0 and (head1_mkt / head1_est) <= 0.90:
        comments.append("1号艇の人気は控えめ。1軸の妙味が出やすい。")
    # コンビ過熱（ざっくり）
    primary_pair = top_pairs[0][0] if top_pairs else (1, 2)
    over_ratio = pair_overbet_ratio(primary_pair, pmap_top10, trifecta_odds, beta=1.0)
    if over_ratio > 1.10:
        comments.append("このコンビは人気先行ぎみ。3着は“おいしい方”へ振ると◎。")

    if comments:
        st.markdown(f"<div class='hint'>{'　'.join(comments)}</div>", unsafe_allow_html=True)

    st.caption(f"オッズ更新時刻: {update_tag or '不明'}　/　開催: {d.strftime('%Y-%m-%d')} {vname} {rno}R")

    # ===== 同舟券コンビ TOP3（2艇） =====
    st.subheader("同舟券コンビ TOP3（2艇）")
    df_pairs = pd.DataFrame([{"コンビ": f"{i}号艇-{j}号艇", "一緒に来やすさ": m} for (i,j), m in top_pairs])
    chart_pairs = alt.Chart(df_pairs).mark_bar().encode(
        x=alt.X("コンビ:N", title="コンビ"),
        y=alt.Y("一緒に来やすさ:Q", title="一緒に来やすさ", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("一緒に来やすさ:Q", title="一緒に来やすさ", format=".1%")]
    ).properties(height=180)
    st.altair_chart(chart_pairs, use_container_width=True)
    st.caption("同じ舟券に絡みやすい2艇の組み合わせ。高いほど一緒に来やすい傾向。")

    # ===== どこまで押さえれば 25/50/75%？ =====
    st.subheader("どこまで押さえれば 25/50/75%？")
    chips = []
    for tval in (0.25, 0.50, 0.75):
        k, picks = cov_targets[tval]
        sets = ", ".join(
            f"{min(S)}={sorted(list(S))[1]}={max(S)}" for S, _ in picks[:k]
        )
        chips.append((int(tval*100), k, sets))
    for pct, k, sets in chips:
        st.markdown(f"<span class='badge badge-strong'>{pct}% 到達</span> 上位{k}組", unsafe_allow_html=True)
        st.write(sets if sets else "-")

    # ===== 予想の方針（お試し） =====
    st.subheader("当てやすさの目安（お試し並べ）")
    base_preview = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2)

    def is_pair_head2(o, pair):
        i, j = pair
        return (o[0], o[1]) == (i, j) or (o[0], o[1]) == (j, i)

    preview = base_preview[:]
    if preset == "🟢 当たり重視":
        target = 0.50 if (target_cover == "50% 目標") else 0.75
        Rt = R
        items = sorted(pmap_top10.items(), key=lambda x: x[1], reverse=True)
        acc = sum(v for _, v in items[:Rt])
        while acc < target and Rt < min(10, len(items)):
            Rt += 1
            acc = sum(v for _, v in items[:Rt])
        preview = build_trifecta_candidates(pmap_top10, R=Rt, avoid_top=True, max_per_set=2)
        preview = preview[: min(12, len(preview))]
        hit_est = sum(p for _, p, _ in preview)
        st.metric("想定の当たりやすさ（プレビュー）", f"{hit_est:.1%}")
        st.caption(f"上位{Rt}組から順序展開（最上位順序は除外）")
    elif preset == "🟡 ほどよく":
        if (over_ratio > 1.10 and cov5 <= 0.65):
            preview = [x for x in preview if not is_pair_head2(x[0], primary_pair)]
            st.caption("本命に偏った“頭-2着”の順序は少し回避。")
        hit_est = sum(p for _, p, _ in preview[:10])
        st.metric("想定の当たりやすさ（プレビュー）", f"{hit_est:.1%}")
    else:  # 🔴 高配当狙い
        if (over_ratio > 1.15 and cov5 <= 0.55):
            preview = [x for x in preview if not is_pair_head2(x[0], primary_pair)]
            st.caption("本命に偏った“頭-2着”の順序はだいぶ回避。")
        preview = preview[:6]
        hit_est = sum(p for _, p, _ in preview)
        st.metric("想定の当たりやすさ（プレビュー）", f"{hit_est:.1%}")

    if preview:
        df_prev = pd.DataFrame([{"買い目": f"{o[0]}-{o[1]}-{o[2]}", "当たりやすさ": p} for (o, p, _) in preview])
        st.dataframe(df_prev.style.format({"当たりやすさ":"{:.2%}"}), use_container_width=True)

    # ===== 3着どれが“おいしい”？（ヒント） =====
    with st.expander("3着どれが“おいしい”？（ヒント）", expanded=False):
        opt_order = st.radio("並びを選ぶ", [f"{primary_pair[0]}-{primary_pair[1]}", f"{primary_pair[1]}-{primary_pair[0]}"],
                             horizontal=True, index=0)
        head, second = [int(x) for x in opt_order.split("-")]
        vr = value_ratios_for_pair(head, second, pmap_top10, trifecta_odds)
        df_vr = pd.DataFrame([
            {"3着": k, "期待（モデル）": ex, "市場（人気）": mk, "市場/期待（低い=おいしい）": r, "オッズ": odds} for (k, ex, mk, r, odds) in vr
        ])
        cL, cR = st.columns(2)
        chart_vr = alt.Chart(df_vr).mark_bar().encode(
            x=alt.X("3着:N", title="3着"),
            y=alt.Y("市場/期待（低い=おいしい）:Q", title="市場/期待（低い=おいしい）"),
            tooltip=["3着", alt.Tooltip("期待（モデル）:Q", format=".1%"), alt.Tooltip("市場（人気）:Q", format=".1%"), alt.Tooltip("市場/期待（低い=おいしい）:Q", format=".2f"), "オッズ"]
        ).properties(height=220)
        cL.altair_chart(chart_vr, use_container_width=True)

        chart_vr2 = alt.Chart(df_vr.melt(id_vars=["3着"], value_vars=["期待（モデル）","市場（人気）"], var_name="種別", value_name="p")).mark_bar().encode(
            x=alt.X("3着:N"),
            y=alt.Y("p:Q", axis=alt.Axis(format="%")),
            color=alt.Color("種別:N"),
            column=alt.Column("種別:N", header=alt.Header(title=None))
        ).properties(height=220)
        cR.altair_chart(chart_vr2, use_container_width=True)
        st.caption("棒が低いほど“まだ買われすぎていない＝おいしい3着”の目安です。")

    st.divider()

    # ===== 買うならコレ（割に合うものだけ） =====
    st.subheader("買うならコレ（割に合うものだけ）")

    # UIの「本命に偏った並びを避ける」を内部モードに写像
    if anti_str == "使わない":
        anti_mode = "M0"
    elif anti_str == "少し避ける":
        anti_mode = "M1"
    else:
        anti_mode = "M2"

    base_cands = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2)
    if add_hedge:
        base_cands = add_pair_hedge_if_needed(base_cands, pmap_top10, top_pairs, max_extra=1)
    cands = base_cands[:max_candidates]

    def is_pair_head2(o, pair):
        i, j = pair
        return (o[0], o[1]) == (i, j) or (o[0], o[1]) == (j, i)

    def need_partial_filter():
        return (over_ratio > 1.10 and cov5 <= 0.65)

    def need_strong_filter():
        return (over_ratio > 1.15 and cov5 <= 0.55)

    if anti_mode == "M1" and need_partial_filter():
        cands = [x for x in cands if not is_pair_head2(x[0], primary_pair)]
        st.warning("“本命に偏った並び”を少し回避しました。")
    elif anti_mode == "M2" and need_strong_filter():
        cands = [x for x in cands if not is_pair_head2(x[0], primary_pair)]
        st.error("“本命に偏った並び”をだいぶ回避しました。")

    # EVチェック
    ok_rows, ng_rows = [], []
    for (o, p_est, S) in cands:
        odds, req, ev, ok = ev_of(o, p_est, trifecta_odds, margin=margin)
        (ok_rows if ok else ng_rows).append((o, p_est, S, odds, req, ev, ok))

    def needed_odds(p_est, margin):
        return (1.0 + margin) / p_est if p_est > 0 else None

    # 配分（OKのみ）
    buys_input = [(o, p, S) for (o, p, S, *_ ) in ok_rows]
    bets, used = allocate_budget_by_prob(buys_input, race_cap, min_unit=min_unit)
    buy_map = {o: b for (o, p, S, b) in bets}

    # OKテーブル
    if ok_rows:
        # 印（◎○△×）付与
        def mark_from_ev(ev):
            if ev is None: return "×"
            if ev >= 0.20: return "◎"
            if ev >= 0.05: return "○"
            if ev >= 0.00: return "△"
            return "×"

        df_ok = pd.DataFrame([
            {
                "印": mark_from_ev(ev),
                "買い目": f"{o[0]}-{o[1]}-{o[2]}",
                "根拠の組（3連複）": f"{min(S)}={sorted(list(S))[1]}={max(S)}",
                "当たりやすさ": p_est,
                "オッズ": odds,
                "これ以上なら買いたい倍率": needed_odds(p_est, margin),
                "見合う最低ライン": req,
                "割に合う度": ev,
                "購入": buy_map.get(o, 0)
            }
            for (o, p_est, S, odds, req, ev, ok) in ok_rows
        ])
        df_ok = df_ok[["印","買い目","根拠の組（3連複）","当たりやすさ","オッズ","これ以上なら買いたい倍率","見合う最低ライン","割に合う度","購入"]]

        # 背景グラデ（matplotlib 必要）。Cloudで導入済みなら有効。
        try:
            styled = (
                df_ok.style
                    .format({"当たりやすさ":"{:.2%}","見合う最低ライン":"{:.2%}","割に合う度":"{:+.1%}","これ以上なら買いたい倍率":"{:.2f}倍"})
                    .background_gradient(subset=["割に合う度"], cmap="Greens")
            )
        except Exception:
            # 依存が無い場合はグラデ無しで表示
            styled = df_ok.style.format({"当たりやすさ":"{:.2%}","見合う最低ライン":"{:.2%}","割に合う度":"{:+.1%}","これ以上なら買いたい倍率":"{:.2f}倍"})

        st.dataframe(styled, use_container_width=True)
    else:
        st.info("割に合う買い目が見つかりません（見送り推奨）。")

    # NG（参考）
    with st.expander("参考：今回は見送り（割に合わない）"):
        if ng_rows:
            df_ng = pd.DataFrame([
                {
                    "買い目": f"{o[0]}-{o[1]}-{o[2]}",
                    "根拠の組（3連複）": f"{min(S)}={sorted(list(S))[1]}={max(S)}",
                    "当たりやすさ": p_est,
                    "オッズ": odds,
                    "これ以上なら買いたい倍率": needed_odds(p_est, margin),
                    "見合う最低ライン": req,
                    "割に合う度": ev
                }
                for (o, p_est, S, odds, req, ev, ok) in ng_rows
            ])
            st.dataframe(
                df_ng.style.format({"当たりやすさ":"{:.2%}","見合う最低ライン":"{:.2%}","割に合う度":"{:+.1%}","これ以上なら買いたい倍率":"{:.2f}倍"}),
                use_container_width=True
            )
        else:
            st.write("（なし）")

    # サマリー
    hit_rate_est = sum(p for (o, p, *_ ) in [(o,p,S) for (o,p,S,odds,req,ev,ok) in ok_rows])
    total_bet = sum(buy_map.get(o,0) for (o, *_ ) in [(o,p,S,odds,req,ev,ok) for (o,p,S,odds,req,ev,ok) in ok_rows])
    cA, cB = st.columns(2)
    cA.metric("想定の当たりやすさ（最終・合算）", f"{hit_rate_est:.1%}")
    cB.metric("合計購入", f"{total_bet} 円")

    # ===== CSV保存 =====
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
    c1.download_button("このレースの記録を保存（CSV）", race_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"race_{d.strftime('%Y%m%d')}_{vname}_{rno}.csv", mime="text/csv")
    c2.download_button("買い目一覧を保存（CSV）", ticket_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"ticket_{d.strftime('%Y%m%d')}_{vname}_{rno}.csv", mime="text/csv")

else:
    st.info("左の条件を設定して「この条件で診断する」を押してください。")
