# -*- coding: utf-8 -*-
"""
app.py
- PC/スマホ両対応（layout="wide"）
- 進捗バー（8ステップ）＋大きな見出し（YYYY-MM-DD 会場 R）
- 点数絞り（6〜8点）・同一ペア最大2点・候補多すぎ時の余裕%引き締め
- “割に合うものだけ”で最終出力
"""

from datetime import date
import json
import streamlit as st
import altair as alt
import pandas as pd

from core_bridge import * 
    VENUES, VENUE_ID2NAME,
    get_trio_odds, get_trifecta_odds, get_just_before_info,
    normalize_probs_from_odds, top5_coverage, inclusion_mass_for_boat,
    pair_mass, estimate_head_rate, choose_R_by_coverage,
    coverage_targets, build_trifecta_candidates, add_pair_hedge_if_needed,
    head_market_rate, pair_overbet_ratio, value_ratios_for_pair,
    ev_of, allocate_budget_by_prob, trim_candidates_with_rules
)

# ---------- ページ設定（PC/スマホ両対応） ----------
st.set_page_config(
    page_title="レース診断（1レース）",
    page_icon="⛵",
    layout="wide"
)

# ---------- テーマ補強CSS ----------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] * {
  color: #111 !important;
}
[data-testid="stAppViewContainer"] { background: #fff !important; }
[data-testid="stHeader"] { background: #fff !important; }
a, a:visited { color: #0E7AFE !important; text-decoration: none; }
.small { font-size: 0.92rem; color: #444 !important; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; background:#eef; margin-right:6px; color:#111 !important; }
.badge-strong { background:#e8f5e9; }
.hint { background:#f7f7ff; padding:10px 12px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.title("⛵ レース診断（1レース）")
st.caption("このレースの“全体像”を、人気の偏りと当てやすさで見える化します。")

# ---------- サイドバー ----------
with st.sidebar:
    st.header("レース選択")
    today = date.today()
    d = st.date_input("開催日", value=today, format="YYYY-MM-DD")
    venue_display = [f"{vid:02d} - {name}" for vid, name in VENUES]
    vsel = st.selectbox("開催場", venue_display, index=len(VENUES)-1)
    vid = int(vsel.split(" - ")[0])
    vname = VENUE_ID2NAME.get(vid, f"場{vid}")
    rno = st.radio("レース番号", list(range(1,13)), index=7, horizontal=True)

    st.divider()
    st.header("予想の方針（お試し）")
    preset = st.radio(" ", ["🟢 当たり重視", "🟡 ほどよく", "🔴 高配当狙い"],
                      horizontal=True, label_visibility="collapsed")
    if preset == "🟢 当たり重視":
        target_cover = st.radio("的中目標", ["50% 目標", "75% 目標"], index=0, horizontal=True)
        st.caption("当てにいくときの目安。どこまで押さえるか。")
    else:
        target_cover = None

    st.divider()
    st.header("最終の買い方（割に合うものだけ）")
    race_cap = st.number_input("このレースに使う上限（円）", min_value=100, value=600, step=100)
    min_unit = st.number_input("最小購入単位（円）", min_value=100, value=100, step=100)
    margin_pct = st.slider("余裕（%）", 0, 30, 10, 1)
    margin = margin_pct / 100.0
    anti_str = st.radio("本命に偏った並びを避ける", ["使わない", "少し避ける", "だいぶ避ける"], horizontal=True)
    add_hedge = st.checkbox("保険を1点足す", value=True)
    max_candidates = st.slider("候補の最大点数", 4, 10, 8, 1)
    st.caption("多すぎる場合は6〜8点を目安に自動で絞ります。")

    do_run = st.button("この条件で診断する", type="primary", use_container_width=True)

# ---------- 実行 ----------
if do_run:
    # 大見出し（どのレースを診ているか）
    st.markdown(f"## {d.strftime('%Y-%m-%d')}　{vname}　**{rno}R**　<span class='badge'>診断開始</span>", unsafe_allow_html=True)

    progress = st.progress(0)
    step_total = 8
    s = st.status("準備中…", state="running")

    try:
        # 1) 3連複
        s.update(label="1/8 3連複オッズを取得中…")
        trio_odds, update_tag = get_trio_odds(d, vid, rno)
        progress.progress(1/step_total)

        # 2) 3連単
        s.update(label="2/8 3連単オッズを取得中…")
        trifecta_odds = get_trifecta_odds(d, vid, rno)
        progress.progress(2/step_total)

        # 3) 直前情報
        s.update(label="3/8 直前情報（展示など）を取得中…")
        just_before = get_just_before_info(d, vid, rno)  # 用途：表示
        progress.progress(3/step_total)

        # 4) 確率化（Top10）
        s.update(label="4/8 3連複Top10を確率化しています…")
        trio_sorted = sorted(trio_odds.items(), key=lambda x: x[1])
        pmap_top10, top10_items = normalize_probs_from_odds(trio_sorted, top_n=10, alpha=1.0)
        ssum = sum((1.0/x for _, x in top10_items)) or 1.0
        progress.progress(4/step_total)

        # 5) 指標計算
        s.update(label="5/8 指標を計算中…")
        cov5 = top5_coverage(pmap_top10)
        inc1 = inclusion_mass_for_boat(pmap_top10, 1)
        head1_est = estimate_head_rate(pmap_top10, head=1)
        head1_mkt = head_market_rate(trifecta_odds, head=1)
        mass_pairs = pair_mass(pmap_top10)
        top_pairs = sorted(mass_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
        R, label_cov = choose_R_by_coverage(pmap_top10)
        cov_targets = coverage_targets(pmap_top10, (0.25, 0.50, 0.75))
        progress.progress(5/step_total)

        # 6) 並び展開（プリビュー & フィルタ）
        s.update(label="6/8 並びを展開しています…")
        base_preview = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2)

        # 過熱判定
        primary_pair = top_pairs[0][0] if top_pairs else (1, 2)
        over_ratio = pair_overbet_ratio(primary_pair, pmap_top10, trifecta_odds, beta=1.0)

        # 予想の方針で軽いフィルタ（プレビューのみ）
        preview = base_preview[:]
        def is_pair_head2(o, pair):
            i, j = pair
            return (o[0], o[1]) in [(i,j),(j,i)]
        if preset == "🟢 当たり重視":
            target = 0.50 if (target_cover == "50% 目標") else 0.75
            Rt = R
            items = sorted(pmap_top10.items(), key=lambda x: x[1], reverse=True)
            acc = sum(v for _, v in items[:Rt])
            while acc < target and Rt < min(10, len(items)):
                Rt += 1
                acc = sum(v for _, v in items[:Rt])
            preview = build_trifecta_candidates(pmap_top10, R=Rt, avoid_top=True, max_per_set=2)[:12]
        elif preset == "🟡 ほどよく":
            if (over_ratio > 1.10 and cov5 <= 0.65):
                preview = [x for x in preview if not is_pair_head2(x[0], primary_pair)]
        else:  # 高配当狙い
            if (over_ratio > 1.15 and cov5 <= 0.55):
                preview = [x for x in preview if not is_pair_head2(x[0], primary_pair)]
            preview = preview[:6]
        progress.progress(6/step_total)

        # 7) EVチェック → OK/NG
        s.update(label="7/8 EV（割に合うか）をチェック中…")
        base_cands = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2)
        if add_hedge:
            base_cands = add_pair_hedge_if_needed(base_cands, pmap_top10, top_pairs, max_extra=1)

        # “本命に偏った並びを避ける”
        anti_mode = {"使わない":"M0","少し避ける":"M1","だいぶ避ける":"M2"}[anti_str]
        if anti_mode == "M1" and (over_ratio > 1.10 and cov5 <= 0.65):
            base_cands = [x for x in base_cands if not is_pair_head2(x[0], primary_pair)]
        elif anti_mode == "M2" and (over_ratio > 1.15 and cov5 <= 0.55):
            base_cands = [x for x in base_cands if not is_pair_head2(x[0], primary_pair)]

        # EV振り分け
        ok_rows, ng_rows = [], []
        for (o, p_est, S) in base_cands:
            odds, req, ev, ok = ev_of(o, p_est, trifecta_odds, margin=margin)
            (ok_rows if ok else ng_rows).append((o, p_est, S, odds, req, ev, ok))

        # 点数絞り（6〜8推奨）＆同一ペア最大2点
        trimmed = trim_candidates_with_rules(ok_rows, max_points=max_candidates, max_same_pair_points=2)

        # 多すぎる時：余裕%を+5ppして再ふるい（1回だけ）
        if len(trimmed) > max_candidates:
            margin2 = min(0.30, margin + 0.05)
            ok2 = []
            for (o, p_est, S, _, _, _, _) in ok_rows:
                odds, req, ev, ok = ev_of(o, p_est, trifecta_odds, margin=margin2)
                if ok:
                    ok2.append((o, p_est, S, odds, req, ev, True))
            trimmed = trim_candidates_with_rules(ok2, max_points=max_candidates, max_same_pair_points=2)

        progress.progress(7/step_total)

        # 8) 配分
        s.update(label="8/8 資金配分中…")
        buys_input = [(o, p, S) for (o,p,S,odds,req,ev,ok) in trimmed]
        bets, used = allocate_budget_by_prob(buys_input, race_cap, min_unit=min_unit)
        buy_map = {o: b for (o, p, S, b) in bets}
        progress.progress(1.0)
        s.update(label="完了", state="complete")

    except Exception as e:
        s.update(label="エラーが発生しました。", state="error")
        st.exception(e)
        st.stop()

    # ===== 上段：レースの傾向（2列×2行） =====
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

    # 短いヒント
    hints = []
    if cov5 >= 0.70:
        hints.append("本命寄りで固め。紐は広めに拾うのが無難。")
    elif cov5 >= 0.60:
        hints.append("やや本命寄り。的中目標50%が現実的。")
    elif cov5 >= 0.55:
        hints.append("中庸。妙味サイドも混ぜるとバランス良し。")
    else:
        hints.append("やや荒れ気配。点数は絞って妙味を優先。")
    if head1_est > 0 and (head1_mkt / head1_est) >= 1.15:
        hints.append("1号艇が買われ過ぎ。1-◯-◯の本命順序は控えめに。")
    elif head1_est > 0 and (head1_mkt / head1_est) <= 0.90:
        hints.append("1号艇の人気は控えめ。1軸の妙味が出やすい。")
    st.markdown(f"<div class='hint'>{'　'.join(hints)}</div>", unsafe_allow_html=True)

    # 直前情報（あれば）
    if just_before:
        jb_txt = []
        if "display_time" in just_before:
            jb_txt.append(f"展示タイム: {just_before['display_time']}")
        if "wind" in just_before:
            jb_txt.append(f"風: {just_before['wind']}")
        if "wave" in just_before:
            jb_txt.append(f"波高: {just_before['wave']}")
        if jb_txt:
            st.caption(" / ".join(jb_txt))

    st.caption(f"オッズ更新時刻: {update_tag or '不明'}")

    # ===== 同舟券コンビ TOP3 =====
    st.subheader("同舟券コンビ TOP3（2艇）")
    mass_pairs = pair_mass(pmap_top10)
    top_pairs = sorted(mass_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
    df_pairs = pd.DataFrame([{"コンビ": f"{i}号艇-{j}号艇", "一緒に来やすさ": m} for (i,j), m in top_pairs])
    chart_pairs = alt.Chart(df_pairs).mark_bar().encode(
        x=alt.X("コンビ:N", title="コンビ"),
        y=alt.Y("一緒に来やすさ:Q", title="一緒に来やすさ", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("一緒に来やすさ:Q", title="一緒に来やすさ", format=".1%")]
    ).properties(height=180)
    st.altair_chart(chart_pairs, use_container_width=True)
    st.caption("同じ舟券に絡みやすい2艇の組み合わせ。高いほど一緒に来やすい傾向。")

    # ===== カバー率 =====
    st.subheader("どこまで押さえれば 25/50/75%？")
    chips = []
    for tval in (0.25, 0.50, 0.75):
        k, items = cov_targets[tval]
        sets = ", ".join(f"{min(S)}={sorted(list(S))[1]}={max(S)}" for S,_ in items[:k])
        st.markdown(f"<span class='badge badge-strong'>{int(tval*100)}% 到達</span> 上位{k}組", unsafe_allow_html=True)
        st.write(sets if sets else "-")

    # ===== お試しプレビュー =====
    st.subheader("当てやすさの目安（お試し並べ）")
    if preview:
        df_prev = pd.DataFrame([{"買い目": f"{o[0]}-{o[1]}-{o[2]}", "当たりやすさ": p} for (o, p, _) in preview])
        st.dataframe(df_prev.style.format({"当たりやすさ":"{:.2%}"}), use_container_width=True)

    # ===== 3着ヒント =====
    with st.expander("3着どれが“おいしい”？（ヒント）", expanded=False):
        primary_pair = top_pairs[0][0] if top_pairs else (1,2)
        opt_order = st.radio("並びを選ぶ", [f"{primary_pair[0]}-{primary_pair[1]}", f"{primary_pair[1]}-{primary_pair[0]}"],
                             horizontal=True, index=0)
        head, second = [int(x) for x in opt_order.split("-")]
        vr = value_ratios_for_pair(head, second, pmap_top10, trifecta_odds)
        df_vr = pd.DataFrame([
            {"3着": k, "期待（モデル）": ex, "市場（人気）": mk, "市場/期待（低い=おいしい）": r, "オッズ": odds}
            for (k, ex, mk, r, odds) in vr
        ])
        cL, cR = st.columns(2)
        chart_vr = alt.Chart(df_vr).mark_bar().encode(
            x=alt.X("3着:N"), y=alt.Y("市場/期待（低い=おいしい）:Q"),
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

    # ===== 最終：買うならコレ =====
    st.subheader("買うならコレ（割に合うものだけ）")
    if trimmed:
        def mark_from_ev(ev):
            if ev is None: return "×"
            if ev >= 0.20: return "◎"
            if ev >= 0.05: return "○"
            if ev >= 0.00: return "△"
            return "×"
        df_ok = pd.DataFrame([{
            "印": mark_from_ev(ev),
            "買い目": f"{o[0]}-{o[1]}-{o[2]}",
            "根拠の組（3連複）": f"{min(S)}={sorted(list(S))[1]}={max(S)}",
            "当たりやすさ": p_est,
            "オッズ": odds,
            "これ以上なら買いたい倍率": (1.0+margin)/p_est if p_est>0 else None,
            "見合う最低ライン": req,
            "割に合う度": ev,
            "購入": buy_map.get(o, 0)
        } for (o,p_est,S,odds,req,ev,ok) in trimmed])
        try:
            styled = df_ok.style.format({
                "当たりやすさ":"{:.2%}","見合う最低ライン":"{:.2%}",
                "割に合う度":"{:+.1%}","これ以上なら買いたい倍率":"{:.2f}倍"
            }).background_gradient(subset=["割に合う度"], cmap="Greens")
        except Exception:
            styled = df_ok.style.format({
                "当たりやすさ":"{:.2%}","見合う最低ライン":"{:.2%}",
                "割に合う度":"{:+.1%}","これ以上なら買いたい倍率":"{:.2f}倍"
            })
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("割に合う買い目が見つかりません（見送り推奨）。")

    # サマリー
    hit_rate_est = sum(p for (o,p,S,odds,req,ev,ok) in trimmed)
    total_bet = sum(buy_map.get(o,0) for (o,p,S,odds,req,ev,ok) in trimmed)
    cA, cB = st.columns(2)
    cA.metric("想定の当たりやすさ（最終・合算）", f"{hit_rate_est:.1%}")
    cB.metric("合計購入", f"{total_bet} 円")

    # CSV保存
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
        "R": R, "max_candidates": len(trimmed), "race_cap": race_cap, "margin": margin
    }
    ticket_records = []
    for (o,p_est,S,odds,req,ev,ok) in trimmed:
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
