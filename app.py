# -*- coding: utf-8 -*-
"""
app.py
- PC/スマホ両対応（layout="wide"）
- 進捗バー（8ステップ）＋大きな見出し（YYYY-MM-DD 会場 R）
- 点数絞り（6〜8点）・同一ペア最大2点・候補多すぎ時の余裕%引き締め
- “割に合うものだけ”で最終出力（core.py の新ロジックに完全準拠）
"""

from datetime import date
import json
import streamlit as st
import altair as alt
import pandas as pd

# ここが最重要：core_bridge から”存在するAPI名”だけを import
from core_bridge import (
    VENUES, VENUE_ID2NAME,
    get_trio_odds, get_trifecta_odds, get_just_before_info,
    normalize_probs_from_odds, normalize_with_dynamic_alpha,
    top5_coverage, inclusion_mass_for_boat,
    pair_mass, estimate_head_rate, choose_R_by_coverage,
    build_trifecta_candidates, add_pair_hedge_if_needed,
    market_head_rate, pair_overbet_ratio, value_ratios_for_pair,
    evaluate_candidates_with_overbet, ev_of_band, adjust_for_slippage,
    trim_candidates_with_rules, allocate_budget_safely
)

# ============== app内ユーティリティ（coreに無い補助） ==============
def coverage_targets_local(pmap_topN, targets=(0.25, 0.50, 0.75)):
    """
    TopNの {S: p} を受け、累積で t を超えるまでに必要なセット数と内訳を返す。
    返り値: { t: (k, [(S,p), ...]) }
    """
    items = sorted(pmap_topN.items(), key=lambda x: x[1], reverse=True)
    out = {}
    for t in targets:
        acc, k, picked = 0.0, 0, []
        for S, p in items:
            if acc >= t: break
            picked.append((S, p))
            acc += p
            k += 1
        out[t] = (k, picked)
    return out

def format_set(S):
    a, b, c = sorted(list(S))
    return f"{a}={b}={c}"

def is_pair_head2(order, pair):
    i, j = pair
    return (order[0], order[1]) in ((i, j), (j, i))

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
    vsel = st.selectbox("開催場", venue_display, index=max(0, len(VENUES)-1))
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
    add_hedge = st.checkbox("保険を条件付きで足す（推奨）", value=True)
    max_candidates = st.slider("候補の最大点数", 4, 10, 8, 1)
    st.caption("多すぎる場合は6〜8点を目安に自動で絞ります。")

    do_run = st.button("この条件で診断する", type="primary", use_container_width=True)

# ---------- 実行 ----------
if do_run:
    st.markdown(
        f"## {d.strftime('%Y-%m-%d')}　{vname}　**{rno}R**　"
        f"<span class='badge'>診断開始</span>",
        unsafe_allow_html=True
    )

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

        # 4) 確率化（Top10）※αはレースの“固さ”で自動選択
        s.update(label="4/8 3連複Top10を確率化しています…")
        trio_sorted = sorted(trio_odds.items(), key=lambda x: x[1])  # オッズ昇順
        pmap_top10, top10_items, alpha_used, cov5_preview = normalize_with_dynamic_alpha(trio_sorted, top_n=10)
        ssum = sum((1.0/x for _, x in top10_items)) or 1.0
        progress.progress(4/step_total)

        # 5) 指標計算
        s.update(label="5/8 指標を計算中…")
        cov5 = top5_coverage(pmap_top10)
        inc1 = inclusion_mass_for_boat(pmap_top10, 1)
        head1_est = estimate_head_rate(pmap_top10, head=1)
        head1_mkt = market_head_rate(trifecta_odds, head=1)
        mass_pairs = pair_mass(pmap_top10)
        top_pairs = sorted(mass_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
        R, label_cov = choose_R_by_coverage(pmap_top10)
        cov_targets = coverage_targets_local(pmap_top10, (0.25, 0.50, 0.75))
        progress.progress(5/step_total)

        # 6) 並び展開（プレビュー & フィルタ）
        s.update(label="6/8 並びを展開しています…")
        base_preview = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2, cov5_hint=cov5)

        primary_pair = top_pairs[0][0] if top_pairs else (1, 2)
        over_ratio = pair_overbet_ratio(primary_pair, pmap_top10, trifecta_odds,)

        # 予想方針で軽いフィルタ（プレビューのみ）
        preview = base_preview[:]
        if preset == "🟢 当たり重視":
            target = 0.50 if (target_cover == "50% 目標") else 0.75
            Rt = R
            items = sorted(pmap_top10.items(), key=lambda x: x[1], reverse=True)
            acc = sum(v for _, v in items[:Rt])
            while acc < target and Rt < min(10, len(items)):
                Rt += 1
                acc = sum(v for _, v in items[:Rt])
            preview = build_trifecta_candidates(pmap_top10, R=Rt, avoid_top=True, max_per_set=2, cov5_hint=cov5)[:12]
        elif preset == "🟡 ほどよく":
            if (over_ratio > 1.10 and cov5 <= 0.65):
                preview = [x for x in preview if not is_pair_head2(x[0], primary_pair)]
        else:  # 🔴 高配当狙い
            if (over_ratio > 1.15 and cov5 <= 0.55):
                preview = [x for x in preview if not is_pair_head2(x[0], primary_pair)]
            preview = preview[:6]
        progress.progress(6/step_total)

        # 7) EVチェック（coreの新方式で一括判定）
        s.update(label="7/8 EV（割に合うか）をチェック中…")
        base_cands = build_trifecta_candidates(pmap_top10, R=R, avoid_top=True, max_per_set=2, cov5_hint=cov5)

        # 条件付きヘッジ：堅くて本命過熱なら1点だけ追加（推奨条件）
        if add_hedge and (cov5 >= 0.68 and over_ratio >= 1.12):
            base_cands = add_pair_hedge_if_needed(base_cands, pmap_top10, top_pairs, max_extra=1)

        # 本命回避強度
        anti_mode = {"使わない": "M0", "少し避ける": "M1", "だいぶ避ける": "M2"}[anti_str]
        if anti_mode == "M1" and (over_ratio > 1.10 and cov5 <= 0.65):
            base_cands = [x for x in base_cands if not is_pair_head2(x[0], primary_pair)]
        elif anti_mode == "M2" and (over_ratio > 1.15 and cov5 <= 0.55):
            base_cands = [x for x in base_cands if not is_pair_head2(x[0], primary_pair)]

        # 過熱課金・オッズ帯出し分け・スリッページを内包した判定
        judged = evaluate_candidates_with_overbet(
            base_cands, pmap_top10, trifecta_odds, base_margin=margin,
            overbet_thresh=1.30, overbet_cut=1.50, overbet_extra=0.03,
            long_odds_extra=0.10, short_odds_relax=0.00,
            long_odds_threshold=25.0, short_odds_threshold=12.0,
            max_odds=60.0, slippage=0.07
        )

        # 点数絞り（6〜8推奨）＆同一ペア最大2点
        trimmed = trim_candidates_with_rules(judged, max_points=max_candidates, max_same_pair_points=2)

        # 多すぎる時：余裕%を+5ppして再ふるい（1回だけ）
        if len(trimmed) > max_candidates:
            judged2 = evaluate_candidates_with_overbet(
                base_cands, pmap_top10, trifecta_odds, base_margin=min(0.30, margin + 0.05),
                overbet_thresh=1.30, overbet_cut=1.50, overbet_extra=0.03,
                long_odds_extra=0.10, short_odds_relax=0.00,
                long_odds_threshold=25.0, short_odds_threshold=12.0,
                max_odds=60.0, slippage=0.07
            )
            trimmed = trim_candidates_with_rules(judged2, max_points=max_candidates, max_same_pair_points=2)

        progress.progress(7/step_total)

        # 8) 配分（半ケリー＋上限）
        s.update(label="8/8 資金配分中…")
        bets, used = allocate_budget_safely(trimmed, race_cap, min_unit=min_unit, per_bet_cap_ratio=0.40)
        buy_map = {o: b for (o, p, S, b, od) in bets}
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
        st.metric("1号艇が1着になりやすさ（推定）", f"{head1_est:.1%}")
        st.caption("3複→3単の順序分解による推定。")
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
        if "display_time" in just_before: jb_txt.append(f"展示タイム: {just_before['display_time']}")
        if "wind" in just_before: jb_txt.append(f"風: {just_before['wind']}")
        if "wave" in just_before: jb_txt.append(f"波高: {just_before['wave']}")
        if jb_txt: st.caption(" / ".join(jb_txt))

    st.caption(f"オッズ更新時刻: {update_tag or '不明'} / α={alpha_used:.2f}（自動）")

    # ===== 同舟券コンビ TOP3 =====
    st.subheader("同舟券コンビ TOP3（2艇）")
    mass_pairs = pair_mass(pmap_top10)
    top_pairs = sorted(mass_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
    df_pairs = pd.DataFrame([{"コンビ": f"{i}号艇-{j}号艇", "一緒に来やすさ": m} for (i, j), m in top_pairs])
    chart_pairs = alt.Chart(df_pairs).mark_bar().encode(
        x=alt.X("コンビ:N", title="コンビ"),
        y=alt.Y("一緒に来やすさ:Q", title="一緒に来やすさ", axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("一緒に来やすさ:Q", title="一緒に来やすさ", format=".1%")]
    ).properties(height=180)
    st.altair_chart(chart_pairs, use_container_width=True)
    st.caption("同じ舟券に絡みやすい2艇の組み合わせ。高いほど一緒に来やすい傾向。")

    # ===== カバー率 =====
    st.subheader("どこまで押さえれば 25/50/75%？")
    for tval in (0.25, 0.50, 0.75):
        k, items = cov_targets[tval]
        sets = ", ".join(format_set(S) for S,_ in items[:k])
        st.markdown(f"<span class='badge badge-strong'>{int(tval*100)}% 到達</span> 上位{k}組", unsafe_allow_html=True)
        st.write(sets if sets else "-")

    # ===== お試しプレビュー =====
    st.subheader("当てやすさの目安（お試し並べ）")
    if 'preview' in locals() and preview:
        df_prev = pd.DataFrame([{"買い目": f"{o[0]}-{o[1]}-{o[2]}", "当たりやすさ": p} for (o, p, _) in preview])
        try:
            st.dataframe(df_prev.style.format({"当たりやすさ": "{:.2%}"}), use_container_width=True)
        except Exception:
            st.dataframe(df_prev, use_container_width=True)

    # ===== 3着ヒント =====
    with st.expander("3着どれが“おいしい”？（ヒント）", expanded=False):
        primary_pair = top_pairs[0][0] if top_pairs else (1, 2)
        opt_order = st.radio(
            "並びを選ぶ",
            [f"{primary_pair[0]}-{primary_pair[1]}", f"{primary_pair[1]}-{primary_pair[0]}"],
            horizontal=True, index=0
        )
        head, second = [int(x) for x in opt_order.split("-")]
        vr = value_ratios_for_pair(head, second, pmap_top10, trifecta_odds)
        df_vr = pd.DataFrame([
            {"3着": k, "期待（モデル）": ex, "市場（人気）": mk, "市場/期待（低い=おいしい）": r, "オッズ": odds}
            for (k, ex, mk, r, odds) in vr
        ])
        cL, cR = st.columns(2)
        chart_vr = alt.Chart(df_vr).mark_bar().encode(
            x=alt.X("3着:N"),
            y=alt.Y("市場/期待（低い=おいしい）:Q"),
            tooltip=[
                "3着",
                alt.Tooltip("期待（モデル）:Q", format=".1%"),
                alt.Tooltip("市場（人気）:Q", format=".1%"),
                alt.Tooltip("市場/期待（低い=おいしい）:Q", format=".2f"),
                "オッズ"
            ]
        ).properties(height=220)
        cL.altair_chart(chart_vr, use_container_width=True)

        chart_vr2 = alt.Chart(
            df_vr.melt(id_vars=["3着"], value_vars=["期待（モデル）", "市場（人気）"], var_name="種別", value_name="p")
        ).mark_bar().encode(
            x=alt.X("3着:N"),
            y=alt.Y("p:Q", axis=alt.Axis(format=".0%")),
            color=alt.Color("種別:N"),
            column=alt.Column("種別:N", header=alt.Header(title=None))
        ).properties(height=220)
        cR.altair_chart(chart_vr2, use_container_width=True)
        st.caption("棒が低いほど“まだ買われすぎていない＝おいしい3着”の目安です。")

    st.divider()

    # ===== 最終：買うならコレ =====
    st.subheader("買うならコレ（割に合うものだけ）")
    if 'trimmed' in locals() and trimmed:
        def mark_from_ev(ev):
            if ev is None: return "×"
            if ev >= 0.20: return "◎"
            if ev >= 0.05: return "○"
            if ev >= 0.00: return "△"
            return "×"
        df_ok = pd.DataFrame([{
            "印": mark_from_ev(ev),
            "買い目": f"{o[0]}-{o[1]}-{o[2]}",
            "根拠の組（3連複）": format_set(S),
            "当たりやすさ": p_est,
            "評価用オッズ（スリッページ込）": odds_eval,
            "見合う最低ライン": req,
            "割に合う度": ev,
            "過熱度（市場/期待）": over,
            "購入": buy_map.get(o, 0)
        } for (o, p_est, S, odds_eval, req, ev, ok, over) in trimmed])
        try:
            styled = df_ok.style.format({
                "当たりやすさ": "{:.2%}",
                "見合う最低ライン": "{:.2%}",
                "割に合う度": "{:+.1%}",
            }).background_gradient(subset=["割に合う度"], cmap="Greens")
        except Exception:
            styled = df_ok.style.format({
                "当たりやすさ": "{:.2%}",
                "見合う最低ライン": "{:.2%}",
                "割に合う度": "{:+.1%}",
            })
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("割に合う買い目が見つかりません（見送り推奨）。")

    # サマリー
    hit_rate_est = sum(p for (o, p, S, odds_eval, req, ev, ok, over) in trimmed) if 'trimmed' in locals() else 0.0
    total_bet = sum(buy_map.get(o, 0) for (o, p, S, odds_eval, req, ev, ok, over) in trimmed) if 'trimmed' in locals() else 0
    cA, cB = st.columns(2)
    cA.metric("想定の当たりやすさ（最終・合算）", f"{hit_rate_est:.1%}")
    cB.metric("合計購入", f"{total_bet} 円")

    # CSV保存
    top10_list = []
    acc = 0.0
    for (S, odd) in top10_items:
        p = (1.0 / odd) / ssum; acc += p
        top10_list.append({"set": format_set(S), "odds": float(odd), "p": float(p), "cum": float(acc)})
    race_record = {
        "date": d.strftime("%Y%m%d"), "venue": vname, "rno": rno, "odds_update": update_tag or "",
        "top10_list": json.dumps(top10_list, ensure_ascii=False),
        "top5_coverage": round(cov5, 6), "inc_mass_1": round(inclusion_mass_for_boat(pmap_top10, 1), 6), "head1_est": round(head1_est, 6),
        "R": R, "max_candidates": len(trimmed) if 'trimmed' in locals() else 0, "race_cap": race_cap, "margin": margin,
        "alpha_used": alpha_used
    }
    ticket_records = []
    if 'trimmed' in locals():
        for (o, p_est, S, odds_eval, req, ev, ok, over) in trimmed:
            ticket_records.append({
                "date": d.strftime("%Y%m%d"), "venue": vname, "rno": rno,
                "selection": f"{o[0]}-{o[1]}-{o[2]}",
                "from_set": format_set(S),
                "p_est": round(p_est, 6), "odds_eval": odds_eval,
                "req_p": round(req, 6) if req is not None else None, "ev_est": round(ev, 6) if ev is not None else None,
                "over_ratio": round(over, 6) if over is not None else None,
                "needed_odds_now": round(((1.0 + margin) / p_est), 6) if p_est > 0 else None,
                "bet_amount": buy_map.get(o, 0)
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
