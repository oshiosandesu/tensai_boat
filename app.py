# app.py
# 天才ボートくん：単レース画面（PC/スマホ対応ダークUI）
# - LIVE/SIMデータソース可視化・フォールバック切替（デバッグ）対応版
# - 🔄 更新ボタンでキャッシュを確実に無効化する refresh_token を追加

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict

from core import (
    VENUE_ID2NAME, VENUE_NAME2ID,
    Snapshot, ModelParams,
    fetch_snapshot, build_probabilities,
    build_ev_candidates, allocate_budget,
    candidates_to_frame, allocations_to_frame,
    marginal_probs_head, marginal_probs_contend, marginal_probs_include,
    COMBS_3F, PERMS_3T
)

st.set_page_config(page_title="天才ボートくん | 単レース", layout="wide", initial_sidebar_state="collapsed")

# ====== ダークUI（boaters-boatrace風：濃紺＋青アクセント） ======
CUSTOM_CSS = """
<style>
:root {
  --bg:#0E1117; --panel:#121826; --border:#1E2638;
  --text:#EAEFF6; --sub:#A7B1C2; --accent:#1E90FF; --warn:#F6C64E; --bad:#F05B5B;
}
body, .stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 0.8rem; padding-bottom: 2rem; }
.stTabs [data-baseweb="tab"] { font-weight: 600; color: var(--sub); }
.stTabs [data-baseweb="tab"]:hover { color: var(--text); }
.stTabs [aria-selected="true"] { color: var(--text); border-bottom: 2px solid var(--accent); }
.card {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 14px; padding: 14px; margin-bottom: 12px;
}
.hstack{ display:flex; gap:12px; align-items:center; }
.vstack{ display:flex; flex-direction:column; gap:8px; }
.kpi-title { font-size: 12px; color: var(--sub); }
.kpi-value { font-size: 20px; font-weight: 700; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid var(--border); background:#0f1422; color:var(--sub); }
.badge.good { color:#7BE389; border-color:#2a4e34; background:#0d1a12;}
.badge.warn { color:var(--warn); border-color:#5a4b1e; background:#1b160b;}
.btn-row { display:flex; gap:8px; align-items:center; }
hr.sep { border:none; border-top:1px solid var(--border); margin:10px 0; }
.table-tight table { font-size: 13px; }
.header-bar {
  position: sticky; top: 0; z-index: 100;
  background: linear-gradient(180deg, rgba(14,17,23,0.95), rgba(14,17,23,0.65));
  backdrop-filter: blur(6px);
  border-bottom: 1px solid var(--border);
  padding: 10px 8px 6px 8px; margin-bottom: 10px;
}
.select-row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.select-row > * { flex: none; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ====== 内部ユーティリティ（当タブ専用：呼び出し前に定義） ======
def _pick_by_k(pmap: Dict, K: int):
    items = sorted(pmap.items(), key=lambda kv: kv[1], reverse=True)
    chosen = [k for k, _ in items[:K]]
    hitrate = float(sum(v for _, v in items[:K]))
    return chosen, hitrate

def _pick_by_target(pmap: Dict, target: float):
    items = sorted(pmap.items(), key=lambda kv: kv[1], reverse=True)
    s = 0.0
    chosen = []
    for k, v in items:
        chosen.append(k)
        s += v
        if s >= target:
            break
    return chosen, float(s)

# ====== サイドバー（解析パラメータ & デバッグ） ======
with st.sidebar:
    st.header("⚙️ 解析パラメータ")
    alpha_3f = st.slider("3複の厚み α", 0.5, 2.0, 1.0, 0.05)
    w_blend = st.slider("ブレンド重み w（p寄り）", 0.0, 1.0, 0.6, 0.05)
    slip = st.slider("スリッページ%", 0.0, 0.05, 0.01, 0.005)
    lam = st.slider("不確実性λ", 0.0, 0.10, 0.03, 0.005)
    ev_th = st.slider("EV' 閾値", -0.2, 0.2, 0.03, 0.01)
    gap_th = st.slider("Edge(p-q) 閾値", -0.05, 0.05, 0.005, 0.001)
    max_pts = st.slider("最大点数(勝ちモード)", 1, 30, 10, 1)
    max_pair_head = st.slider("同一(頭-2着)上限", 1, 6, 3, 1)
    st.caption("※ 変更すると次回更新時に反映")

    st.markdown("---")
    st.header("💰 資金配分（EVモード）")
    race_budget = st.number_input("1R上限（円）", min_value=0, max_value=1_000_000, value=5000, step=500)
    min_unit = st.number_input("最小単位（円）", min_value=100, max_value=10000, value=500, step=100)
    st.caption("※ 当てにいくモードでは配分は行いません（提案のみ）")

    st.markdown("---")
    st.header("🛠 デバッグ")
    allow_sim = st.toggle("シミュレーションフォールバックを許可", value=True,
                          help="OFFにすると実オッズ取得に失敗した際は空データになります（問題の切り分け用）")

# ====== ヘッダーバー ======
with st.container():
    st.markdown('<div class="header-bar">', unsafe_allow_html=True)
    cols = st.columns([2,3,3,3,3,2])
    with cols[0]:
        st.markdown("### 🛶 天才ボートくん")
        st.caption("オッズの“歪み”で戦う ｜ EVモード & 当てにいくモード")
    today = datetime.now()
    with cols[1]:
        date = st.date_input("開催日", value=today.date(), format="YYYY-MM-DD")
    with cols[2]:
        venue_name = st.selectbox("会場", options=list(VENUE_NAME2ID.keys()), index=15-1)  # 丸亀を初期値
        venue_id = VENUE_NAME2ID[venue_name]
    with cols[3]:
        race_no = st.number_input("レース", min_value=1, max_value=12, value=9, step=1)
    with cols[4]:
        mode = st.segmented_control("モード", options=["EVで勝つ", "当てにいく"], default="EVで勝つ")
    reload_click = cols[5].button("🔄 更新", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ====== 更新ボタンでキャッシュを無効化するトークン ======
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0
if reload_click:
    st.session_state.refresh_token += 1  # ボタン押下のたびにトークンを変える

# ====== スナップショット取得（30秒キャッシュ） ======
@st.cache_data(show_spinner=False, ttl=30)
def _load_snapshot(date_str: str, vid: int, rno: int, allow_sim_flag: bool, refresh_token: int) -> Snapshot:
    # refresh_token はキャッシュキーにだけ使う（中では未使用）
    return fetch_snapshot(date_str, vid, rno, allow_sim_fallback=allow_sim_flag)

date_str = date.strftime("%Y%m%d")
snapshot: Snapshot = _load_snapshot(
    date_str, venue_id, race_no, allow_sim_flag=allow_sim, refresh_token=st.session_state.refresh_token
)

# ====== モデル確率構築 ======
params = ModelParams(
    alpha_3f=float(alpha_3f),
    blend_w=float(w_blend),
    slippage_pct=float(slip),
    lambda_uncertainty=float(lam),
    ev_threshold=float(ev_th),
    gap_threshold=float(gap_th),
    max_points=int(max_pts),
    max_pair_per_head=int(max_pair_head),
)
probs = build_probabilities(snapshot, params)

# ====== 相場サマリー（トップカード3枚 + 取得ソース） ======
k1, k2, k3, k4 = st.columns([2,2,2,3])
with k1:
    st.markdown('<div class="card vstack">', unsafe_allow_html=True)
    st.caption("レース概要")
    st.markdown(f"**{VENUE_ID2NAME[snapshot.venue_id]} {snapshot.race_no}R**")
    st.caption(f"取得: {snapshot.taken_at.strftime('%H:%M:%S')}")
    st.markdown('</div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="card vstack">', unsafe_allow_html=True)
    st.caption("相場指標")
    st.markdown(f"Top5カバレッジ(3複): **{snapshot.meta.get('coverage_top5_3f',0):.2f}**")
    st.markdown(f"エントロピー(3単): **{snapshot.meta.get('entropy_3t',0):.2f}**")
    st.markdown('</div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="card vstack">', unsafe_allow_html=True)
    q_sum = snapshot.meta.get("book_sum", None)
    st.caption("ブック合算（参考）")
    if q_sum is not None:
        st.markdown(f"∑1/odds(3単): **{q_sum:.2f}**")
    else:
        st.markdown("—")
    st.markdown('</div>', unsafe_allow_html=True)
with k4:
    st.markdown('<div class="card hstack">', unsafe_allow_html=True)
    # データソース表示（LIVE/SIM）
    src3t = snapshot.meta.get("source_3t", "?")
    src3f = snapshot.meta.get("source_3f", "?")
    cnt3t = snapshot.meta.get("count_3t", 0)
    cnt3f = snapshot.meta.get("count_3f", 0)
    cls_live3t = "badge good" if str(src3t).startswith("live") else ("badge warn" if src3t == "sim" else "badge")
    cls_live3f = "badge good" if str(src3f).startswith("live") else ("badge warn" if src3f == "sim" else "badge")
    st.markdown(f'<span class="{cls_live3t}">3単: {src3t} / {cnt3t}件</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="{cls_live3f}">3複: {src3f} / {cnt3f}件</span>', unsafe_allow_html=True)

    st.markdown('<div style="flex:1"></div>', unsafe_allow_html=True)
    # 相場判定バッジ
    tag = "標準"
    cov = snapshot.meta.get("coverage_top5_3f", 0.0)
    ent = snapshot.meta.get("entropy_3t", 0.0)
    if cov >= 0.75 and ent <= 3.8:
        tag = "固め"
    elif cov <= 0.60 or ent >= 4.2:
        tag = "荒れやすい"
    cls = "badge"
    if tag == "固め": cls += " good"
    if tag == "荒れやすい": cls += " warn"
    st.markdown(f'<span class="{cls}">相場判定: {tag}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ====== タブ群 ======
tab1, tab2, tab3, tab4 = st.tabs(["選手比較", "オッズ可視化", "EV（勝ち）", "当てにいく（遊び）"])

# ---- タブ1：選手比較（6カード） ----
with tab1:
    head_p = marginal_probs_head(probs["p3t"])
    head_q = marginal_probs_head(probs["q3t"])
    inc_p = marginal_probs_include(probs["p3t"])
    inc_q = marginal_probs_include(probs["q3t"])

    rows = []
    for lane in range(1, 7):
        r = snapshot.entries.get(lane)
        nm = r.name if r and r.name else f"Lane{lane}"
        klass = r.klass or "-"
        motor = f"{r.motor_no or '-'} / {r.motor_2r or '-'}"
        boat = f"{r.boat_no or '-'} / {r.boat_2r or '-'}"
        ex = f"{(r.exhibit_time or 0):.2f}" if (r and r.exhibit_time) else "-"
        rows.append({
            "枠": lane, "選手": nm, "級": klass,
            "展示": ex,
            "モーターNo/2連率": motor,
            "ボートNo/2連率": boat,
            "頭(モデル/市場)": f"{head_p.get(lane,0):.2f} / {head_q.get(lane,0):.2f}",
            "含有(モデル/市場)": f"{inc_p.get(lane,0):.2f} / {inc_q.get(lane,0):.2f}",
        })
    st.markdown('<div class="card table-tight">', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- タブ2：オッズ可視化 ----
with tab2:
    trio_probs = []
    for comb in COMBS_3F:
        o = next((x.odds for x in snapshot.odds_trio if x.comb == comb), None)
        if o:
            trio_probs.append((comb, 1.0 / o))
    trio_probs.sort(key=lambda x: x[1], reverse=True)
    top10 = trio_probs[:10]
    df_top10 = pd.DataFrame({
        "組(3複)": [f"{a}-{b}-{c}" for (a,b,c),_ in top10],
        "相対厚み(1/odds)": [v for _, v in top10],
    })
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("3複 Top10（相対厚み）")
    st.bar_chart(df_top10.set_index("組(3複)"))
    st.markdown('</div>', unsafe_allow_html=True)

    q_items = sorted(probs["q3t"].items(), key=lambda kv: kv[1], reverse=True)
    ranks = [f"{a}-{b}-{c}" for (a,b,c),_ in q_items[:20]]
    vals = [v for _, v in q_items[:20]]
    df_q = pd.DataFrame({"組(3単)": ranks, "q(市場)": vals})
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("3単 人気上位（市場確率q）")
    st.bar_chart(df_q.set_index("組(3単)"))
    st.markdown('</div>', unsafe_allow_html=True)

# ---- タブ3：EV（勝ち） ----
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("買い目候補（EVモード）")
    ev_cands = build_ev_candidates(snapshot, probs, params)
    if ev_cands:
        df_c = candidates_to_frame(ev_cands)
        st.dataframe(df_c, use_container_width=True, hide_index=True)
        st.markdown("—")
        st.markdown("#### 資金配分（半ケリー基準・丸めあり）")
        allocs = allocate_budget(ev_cands, race_budget=race_budget, min_unit=min_unit, kelly_fraction_scale=0.5)
        df_a = allocations_to_frame(allocs)
        st.dataframe(df_a, use_container_width=True, hide_index=True)
        tot = int(df_a["ベット額"].sum()) if not df_a.empty else 0
        st.markdown(f"**合計ベット額:** {tot:,} 円")
    else:
        st.info("条件に合致する候補がありません。EV'閾値・Edge閾値・点数上限などを緩めてください。")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- タブ4：当てにいく（遊び） ----
with tab4:
    st.markdown('<div class="card vstack">', unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)
    with colA:
        hit_mode = st.selectbox("対象", options=["3連単", "3連複"], index=0)
    with colB:
        select_type = st.selectbox("選び方", options=["点数で指定", "目標的中率で指定"], index=0)
    with colC:
        if select_type == "点数で指定":
            K = st.number_input("点数K", min_value=1, max_value=60, value=8, step=1)
            target = None
        else:
            target = st.slider("目標的中率(%)", 10, 95, 40, 1) / 100.0
            K = None

    if hit_mode == "3連単":
        pmap = probs["pstar3t"]
    else:
        pmap = probs["pstar3f"]

    if select_type == "点数で指定":
        chosen, hitrate = _pick_by_k(pmap=pmap, K=int(K))
    else:
        chosen, hitrate = _pick_by_target(pmap=pmap, target=float(target))

    if hit_mode == "3連単":
        label = [f"{a}-{b}-{c}" for (a,b,c) in chosen]
        odds_lookup = {x.comb: x.odds for x in snapshot.odds_trifecta}
        odds_vals = [odds_lookup.get(t, None) for t in chosen]
        df_hit = pd.DataFrame({"出目(3単)": label, "p*（想定的中率の構成要素）": [pmap[t] for t in chosen], "odds(参考)": odds_vals})
        st.dataframe(df_hit, use_container_width=True, hide_index=True)
    else:
        label = [f"{a}-{b}-{c}" for (a,b,c) in chosen]
        odds_lookup = {x.comb: x.odds for x in snapshot.odds_trio}
        odds_vals = [odds_lookup.get(tuple(sorted(t)), None) for t in chosen]
        df_hit = pd.DataFrame({"出目(3複)": label, "p*（想定的中率の構成要素）": [pmap[t] for t in chosen], "odds(参考)": odds_vals})
        st.dataframe(df_hit, use_container_width=True, hide_index=True)

    st.markdown(f"**想定的中率（合算）:** {hitrate:.2%}")
    st.caption("※ 遊びモードは“的中体験”重視（ROIは低下し得ます）。")
    st.markdown('</div>', unsafe_allow_html=True)
