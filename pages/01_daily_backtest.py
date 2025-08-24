# pages/01_daily_backtest.py
# 天才ボートくん：日次バックテスト（EVモード / 当てにいくモード）
# - 既存の app.py / core.py に合わせた純依存構成
# - レースごとに snapshot->確率化->候補抽出->配分->結果照合 を実施
# - 注意：fetch_snapshot は「実行時点のオッズ」を取得します（厳密な時点固定は今後の拡張で）

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date as date_cls
from typing import Dict, Tuple, List, Optional

from core import (
    VENUE_ID2NAME, VENUE_NAME2ID,
    Snapshot, ModelParams,
    fetch_snapshot, build_probabilities,
    build_ev_candidates, allocate_budget,
    candidates_to_frame, PERMS_3T
)

# ---- 可能なら確定結果を pyjpboatrace から取得（失敗時は None） ----
try:
    import pyjpboatrace as pjb
except Exception:
    pjb = None


st.set_page_config(page_title="天才ボートくん | 日次バックテスト", layout="wide", initial_sidebar_state="expanded")

# ====== スタイル（app.py のダークUIに合わせる） ======
CUSTOM_CSS = """
<style>
:root {
  --bg:#0E1117; --panel:#121826; --border:#1E2638;
  --text:#EAEFF6; --sub:#A7B1C2; --accent:#1E90FF; --warn:#F6C64E; --bad:#F05B5B;
}
body, .stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 0.8rem; padding-bottom: 2rem; }
.card {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 14px; padding: 14px; margin-bottom: 12px;
}
.table-tight table { font-size: 13px; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid var(--border); background:#0f1422; color:var(--sub); }
.badge.good { color:#7BE389; border-color:#2a4e34; background:#0d1a12;}
.badge.warn { color:var(--warn); border-color:#5a4b1e; background:#1b160b;}
.header-bar {
  position: sticky; top: 0; z-index: 100;
  background: linear-gradient(180deg, rgba(14,17,23,0.95), rgba(14,17,23,0.65));
  backdrop-filter: blur(6px);
  border-bottom: 1px solid var(--border);
  padding: 10px 8px 6px 8px; margin-bottom: 10px;
}
.sep { border:none; border-top:1px solid var(--border); margin:10px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ====== 確定結果取得ユーティリティ ======
def get_trifecta_result(date_str: str, venue_id: int, race_no: int) -> Optional[Tuple[Tuple[int, int, int], float]]:
    """
    確定結果（3連単の着順タプル、払戻オッズ）を返す。
    - 返り値例：((1,2,3), 12.3)  # 12.3倍（= 払戻 / 100円）
    - 取得できない場合は None
    """
    # pyjpboatrace のAPI差異に幅を持たせてトライ
    if pjb is None:
        return None
    try:
        data = None
        if hasattr(pjb, "get_trifecta_result"):
            data = pjb.get_trifecta_result(date=date_str, jcd=venue_id, rno=race_no)
        elif hasattr(pjb, "OfficialAPI"):
            api = pjb.OfficialAPI()
            data = api.result_trifecta(date=date_str, jcd=venue_id, rno=race_no)
        # 代表的な形に合わせてパース
        if isinstance(data, dict):
            a = int(data.get("first") or data.get("a") or data.get("i") or 0)
            b = int(data.get("second") or data.get("b") or data.get("j") or 0)
            c = int(data.get("third") or data.get("c") or data.get("k") or 0)
            odds = data.get("odds") or data.get("payout_odds") or data.get("value") or None
            if a and b and c and odds is not None:
                return (a, b, c), float(odds)
        elif isinstance(data, (list, tuple)) and len(data) >= 1:
            # list[dict]や[(comb, odds)]形式への緩和
            row = data[0]
            if isinstance(row, dict):
                a = int(row.get("first") or row.get("a") or row.get("i") or 0)
                b = int(row.get("second") or row.get("b") or row.get("j") or 0)
                c = int(row.get("third") or row.get("c") or row.get("k") or 0)
                odds = row.get("odds") or row.get("payout_odds") or row.get("value") or None
                if a and b and c and odds is not None:
                    return (a, b, c), float(odds)
            elif isinstance(row, (list, tuple)) and len(row) == 2:
                comb, odds = row
                comb = tuple(int(x) for x in comb)
                return comb, float(odds)
    except Exception:
        pass
    return None


# ====== キャッシュ層（スナップショット） ======
@st.cache_data(show_spinner=False, ttl=30)
def _load_snapshot(date_str: str, vid: int, rno: int) -> Snapshot:
    return fetch_snapshot(date_str, vid, rno)


# ====== ページUI（入力） ======
st.markdown('<div class="header-bar">', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns([2, 3, 3, 3, 3, 2])
with c1:
    st.markdown("### 📊 日次バックテスト")
with c2:
    target_date: date_cls = st.date_input("開催日", value=datetime.now().date(), format="YYYY-MM-DD")
with c3:
    venue_opts = list(VENUE_NAME2ID.keys())
    pick_all = st.checkbox("全会場", value=True)
    if pick_all:
        venues_selected = venue_opts
    else:
        venues_selected = st.multiselect("対象会場", options=venue_opts, default=["丸亀"] if "丸亀" in venue_opts else venue_opts[:1])
with c4:
    r_start = st.number_input("開始R", 1, 12, 1, 1)
with c5:
    r_end = st.number_input("終了R", 1, 12, 12, 1)
with c6:
    run_btn = st.button("▶ 実行", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# 解析パラメータ
with st.sidebar:
    st.header("⚙️ 解析パラメータ（EVモード）")
    alpha_3f = st.slider("3複の厚み α", 0.5, 2.0, 1.0, 0.05)
    w_blend = st.slider("ブレンド重み w（p寄り）", 0.0, 1.0, 0.6, 0.05)
    slip = st.slider("スリッページ%", 0.0, 0.05, 0.01, 0.005)
    lam = st.slider("不確実性λ", 0.0, 0.10, 0.03, 0.005)
    ev_th = st.slider("EV' 閾値", -0.2, 0.2, 0.03, 0.01)
    gap_th = st.slider("Edge(p-q) 閾値", -0.05, 0.05, 0.005, 0.001)
    max_pts = st.slider("最大点数(勝ちモード)", 1, 30, 10, 1)
    max_pair_head = st.slider("同一(頭-2着)上限", 1, 6, 3, 1)
    st.caption("※ 候補抽出/配分はこれらに従います")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.header("💰 資金（EVモード）")
    race_budget = st.number_input("1R上限（円）", min_value=0, max_value=1_000_000, value=5000, step=500)
    min_unit = st.number_input("最小単位（円）", min_value=100, max_value=10000, value=500, step=100)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.header("🎯 当てにいく（遊び）")
    hit_target_type = st.selectbox("選び方", ["点数で指定", "目標的中率で指定"], index=0)
    if hit_target_type == "点数で指定":
        hit_K_3t = st.number_input("3連単 K点", 1, 60, 8, 1)
        hit_K_3f = st.number_input("3連複 K点", 1, 60, 5, 1)
        hit_T = None
    else:
        hit_T = st.slider("目標的中率(%)", 10, 95, 40, 1) / 100.0
        hit_K_3t, hit_K_3f = None, None
    hit_unit = st.number_input("遊びの1点あたりベット（円）", min_value=0, max_value=10000, value=500, step=100)
    st.caption("※ 遊びモードは均等ベットで評価します（ROIより“当てる体験”重視）")


# ====== 内部：遊びモードの選抜 ======
def pick_topK(prob_map: Dict[Tuple[int, int, int], float], K: int) -> Tuple[List[Tuple[int, int, int]], float]:
    items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    chosen = [k for k, _ in items[:K]]
    hitrate = float(sum(v for _, v in items[:K]))
    return chosen, hitrate


def pick_by_target(prob_map: Dict[Tuple[int, int, int], float], target: float) -> Tuple[List[Tuple[int, int, int]], float]:
    items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    s = 0.0
    chosen = []
    for k, v in items:
        chosen.append(k)
        s += v
        if s >= target:
            break
    return chosen, float(s)


# ====== 実行 ======
if run_btn:
    tgt_date_str = target_date.strftime("%Y%m%d")
    venues = venues_selected if not isinstance(venues_selected, str) else [venues_selected]
    v_ids = [VENUE_NAME2ID[v] for v in venues]

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

    # 集計用
    rows_detail = []
    total_bet_ev = 0
    total_ret_ev = 0
    hit_cnt_ev = 0
    race_cnt_ev = 0

    total_bet_hit_3t = 0
    total_ret_hit_3t = 0
    hit_cnt_hit_3t = 0
    race_cnt_hit_3t = 0

    total_bet_hit_3f = 0
    total_ret_hit_3f = 0
    hit_cnt_hit_3f = 0
    race_cnt_hit_3f = 0

    progress = st.progress(0.0)
    tasks = len(v_ids) * max(0, (int(r_end) - int(r_start) + 1))
    done = 0

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("レース明細")
    detail_container = st.container()
    st.markdown('</div>', unsafe_allow_html=True)

    for vid in v_ids:
        for rno in range(int(r_start), int(r_end) + 1):
            try:
                snap: Snapshot = _load_snapshot(tgt_date_str, vid, rno)
                probs = build_probabilities(snap, params)
            except Exception as e:
                rows_detail.append({
                    "会場": VENUE_ID2NAME.get(vid, str(vid)),
                    "R": rno,
                    "状態": f"取得失敗: {e}",
                    "EV候補点数": 0,
                    "EVベット": 0,
                    "EV払戻": 0,
                    "遊び3単点数": 0,
                    "遊び3単ベット": 0,
                    "遊び3単払戻": 0,
                    "遊び3複点数": 0,
                    "遊び3複ベット": 0,
                    "遊び3複払戻": 0,
                })
                done += 1
                progress.progress(min(1.0, done / max(1, tasks)))
                continue

            # 結果（可能なら）
            result = get_trifecta_result(tgt_date_str, vid, rno)
            result_comb: Optional[Tuple[int, int, int]] = result[0] if result else None
            result_odds: Optional[float] = result[1] if result else None

            # ---- EVモード ----
            ev_cands = build_ev_candidates(snap, probs, params)
            allocs = allocate_budget(ev_cands, race_budget=race_budget, min_unit=min_unit, kelly_fraction_scale=0.5)
            bet_ev = int(sum(a.stake for a in allocs)) if allocs else 0
            ret_ev = 0
            hit_flag_ev = False
            if result_comb is not None and allocs:
                # 当たり組の合計ベット×オッズで払戻（複数同一組が存在し得ない設計だが念のため合算）
                match_sum = sum(a.stake for a in allocs if a.comb == result_comb)
                if match_sum > 0 and result_odds is not None:
                    ret_ev = int(round(match_sum * float(result_odds) / (min_unit if min_unit > 0 else 1) * min_unit))
                    hit_flag_ev = True
            total_bet_ev += bet_ev
            total_ret_ev += ret_ev
            if bet_ev > 0:
                race_cnt_ev += 1
                if hit_flag_ev:
                    hit_cnt_ev += 1

            # ---- 当てにいく（3連単） ----
            pstar3t = probs["pstar3t"]
            if hit_target_type == "点数で指定":
                chosen_3t, hitrate_3t = pick_topK(pstar3t, int(hit_K_3t))
            else:
                chosen_3t, hitrate_3t = pick_by_target(pstar3t, float(hit_T))
            bet_hit_3t = int(hit_unit) * len(chosen_3t) if hit_unit and chosen_3t else 0
            ret_hit_3t = 0
            hit_flag_hit_3t = False
            if result_comb is not None and chosen_3t:
                if result_comb in chosen_3t and result_odds is not None:
                    ret_hit_3t = int(round(hit_unit * float(result_odds)))
                    hit_flag_hit_3t = True
            total_bet_hit_3t += bet_hit_3t
            total_ret_hit_3t += ret_hit_3t
            if bet_hit_3t > 0:
                race_cnt_hit_3t += 1
                if hit_flag_hit_3t:
                    hit_cnt_hit_3t += 1

            # ---- 当てにいく（3連複） ----
            pstar3f = probs["pstar3f"]
            if hit_target_type == "点数で指定":
                chosen_3f, hitrate_3f = pick_topK(pstar3f, int(hit_K_3f))
            else:
                chosen_3f, hitrate_3f = pick_by_target(pstar3f, float(hit_T))
            bet_hit_3f = int(hit_unit) * len(chosen_3f) if hit_unit and chosen_3f else 0
            # 3連複の結果と払戻を pyjpboatrace から取れる場合は別途実装が必要だが、
            # 本ページでは 3連単の的中評価のみ厳密計算。3連複の払戻は評価省略（=0）とする。
            # ※ 将来的に OfficialAPI の trio payout を追加してください。
            ret_hit_3f = 0
            hit_flag_hit_3f = False
            # 的中判定のみ（3複は昇順セット一致）
            if result_comb is not None and chosen_3f:
                res_set = tuple(sorted(result_comb))
                if res_set in [tuple(sorted(x)) for x in chosen_3f]:
                    hit_flag_hit_3f = True
                    # 払戻はAPI実装次第で加算（今は0のまま）
            total_bet_hit_3f += bet_hit_3f
            total_ret_hit_3f += ret_hit_3f
            if bet_hit_3f > 0:
                race_cnt_hit_3f += 1
                if hit_flag_hit_3f:
                    hit_cnt_hit_3f += 1

            # ---- 明細行（画面表示用） ----
            rows_detail.append({
                "会場": VENUE_ID2NAME.get(vid, str(vid)),
                "R": rno,
                "状態": "OK" if result_comb is not None else "結果不明",
                "結果(3単)": f"{result_comb[0]}-{result_comb[1]}-{result_comb[2]}" if result_comb else "-",
                "結果オッズ": round(float(result_odds), 2) if result_odds is not None else None,
                "EV候補点数": len(ev_cands),
                "EVベット": bet_ev,
                "EV払戻": ret_ev,
                "遊び3単点数": len(chosen_3t),
                "想定的中率3単(%)": round(hitrate_3t * 100.0, 1) if 'hitrate_3t' in locals() else None,
                "遊び3単ベット": bet_hit_3t,
                "遊び3単払戻": ret_hit_3t,
                "遊び3複点数": len(chosen_3f),
                "想定的中率3複(%)": round(hitrate_3f * 100.0, 1) if 'hitrate_3f' in locals() else None,
                "遊び3複ベット": bet_hit_3f,
                "遊び3複払戻": ret_hit_3f,
            })

            done += 1
            progress.progress(min(1.0, done / max(1, tasks)))

    # ====== 集計と表示 ======
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("集計（EVモード）")
    roi_ev = (total_ret_ev / total_bet_ev) if total_bet_ev > 0 else 0.0
    hit_rate_ev = (hit_cnt_ev / race_cnt_ev) if race_cnt_ev > 0 else 0.0
    colA, colB, colC, colD = st.columns(4)
    with colA: st.metric("総投下", f"{total_bet_ev:,} 円")
    with colB: st.metric("総払戻", f"{total_ret_ev:,} 円")
    with colC: st.metric("ROI", f"{roi_ev:.2f} 倍")
    with colD: st.metric("的中率", f"{hit_rate_ev:.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("集計（当てにいく：3連単）")
    roi_hit_3t = (total_ret_hit_3t / total_bet_hit_3t) if total_bet_hit_3t > 0 else 0.0
    hit_rate_hit_3t = (hit_cnt_hit_3t / race_cnt_hit_3t) if race_cnt_hit_3t > 0 else 0.0
    colE, colF, colG, colH = st.columns(4)
    with colE: st.metric("総投下(3単)", f"{total_bet_hit_3t:,} 円")
    with colF: st.metric("総払戻(3単)", f"{total_ret_hit_3t:,} 円")
    with colG: st.metric("ROI(3単)", f"{roi_hit_3t:.2f} 倍")
    with colH: st.metric("的中率(3単)", f"{hit_rate_hit_3t:.1%}")
    st.caption("※ 遊びモードは均等ベットで算出。ROIは参考値です。")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("集計（当てにいく：3連複）")
    roi_hit_3f = (total_ret_hit_3f / total_bet_hit_3f) if total_bet_hit_3f > 0 else 0.0
    hit_rate_hit_3f = (hit_cnt_hit_3f / race_cnt_hit_3f) if race_cnt_hit_3f > 0 else 0.0
    colI, colJ, colK, colL = st.columns(4)
    with colI: st.metric("総投下(3複)", f"{total_bet_hit_3f:,} 円")
    with colJ: st.metric("総払戻(3複)", f"{total_ret_hit_3f:,} 円")
    with colK: st.metric("ROI(3複)", f"{roi_hit_3f:.2f} 倍")
    with colL: st.metric("的中率(3複)", f"{hit_rate_hit_3f:.1%}")
    st.caption("※ 現状、3連複の払戻取得は未実装のため 0 扱いです（API対応後に加算可能）。")
    st.markdown('</div>', unsafe_allow_html=True)

    # 明細テーブル
    st.markdown('<div class="card table-tight">', unsafe_allow_html=True)
    st.subheader("明細テーブル")
    df_detail = pd.DataFrame(rows_detail)
    st.dataframe(df_detail, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("左上の開催日・会場・R範囲を選んで「▶ 実行」を押してください。")
