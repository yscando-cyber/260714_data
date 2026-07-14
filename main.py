import glob
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------
# 페이지 설정
# ----------------------------------------------------------------
st.set_page_config(page_title="🏘️ 인구구조 쌍둥이 지역 찾기", page_icon="👯", layout="wide")
st.title("👯 우리 동네와 인구구조가 가장 비슷한 '쌍둥이 지역' 찾기")
st.caption("행정안전부 연령별 인구현황(월간) 데이터 기반 · Plotly 인터랙티브 시각화")

CSV_FILENAME = "202606_202606_연령별인구현황_월간.csv"

# ----------------------------------------------------------------
# 데이터 로딩 & 전처리
# ----------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    # 정확한 파일명이 없으면 같은 폴더에서 "연령별인구현황"이 포함된 csv를 자동으로 찾음
    path = CSV_FILENAME
    if not os.path.exists(path):
        candidates = glob.glob("*연령별인구현황*.csv")
        if candidates:
            path = candidates[0]
        else:
            return None, None, None

    raw = pd.read_csv(path, encoding="cp949", low_memory=False)

    # 컬럼명에서 "YYYY년MM월" 접두어를 자동 탐지 (다른 월 데이터를 넣어도 동작하도록)
    sample_col = next(c for c in raw.columns if "_계_총인구수" in c)
    prefix = sample_col.split("_계_총인구수")[0]

    age_pattern = re.compile(rf"^{re.escape(prefix)}_(계|남|여)_(\d+)세$")
    age100_pattern = re.compile(rf"^{re.escape(prefix)}_(계|남|여)_100세 이상$")

    def clean_num(series):
        return pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce"
        ).fillna(0)

    # 행정구역명 파싱: "서울특별시 종로구 청운효자동(1111051500)" -> 이름 / 코드 / 레벨
    def parse_region(raw_name):
        m = re.match(r"^(.*?)\s*\((\d+)\)\s*$", raw_name.strip())
        if not m:
            return raw_name.strip(), "", "기타"
        name, code = m.group(1).strip(), m.group(2)
        if code[2:] == "0" * 8:
            level = "시도"
        elif code[5:] == "0" * 5:
            level = "시군구"
        else:
            level = "읍면동"
        return name, code, level

    parsed = raw["행정구역"].apply(parse_region)
    out = pd.DataFrame({
        "코드": [p[1] for p in parsed],
        "지역명": [p[0] for p in parsed],
        "레벨": [p[2] for p in parsed],
    })
    out["총인구수"] = clean_num(raw[f"{prefix}_계_총인구수"])

    # 0~100세+ 나이별 인구수를 5세 단위로 묶기 (성별: 계/남/여)
    bin_labels = [f"{i}-{i+4}세" for i in range(0, 100, 5)] + ["100세+"]
    for gender in ["계", "남", "여"]:
        age_cols = [f"{prefix}_{gender}_{i}세" for i in range(100)] + [f"{prefix}_{gender}_100세 이상"]
        age_cols = [c for c in age_cols if c in raw.columns]
        age_vals = raw[age_cols].apply(clean_num)
        # 5세 단위로 합산
        n_full_bins = 20  # 0~99세 -> 20개 구간
        for b in range(n_full_bins):
            cols_in_bin = age_vals.columns[b * 5:(b + 1) * 5]
            out[f"{gender}_{bin_labels[b]}"] = age_vals[cols_in_bin].sum(axis=1).values
        out[f"{gender}_{bin_labels[-1]}"] = age_vals.iloc[:, -1].values  # 100세 이상

    out = out[out["총인구수"] > 0].reset_index(drop=True)
    return out, bin_labels, prefix


df, bin_labels, prefix = load_data()

if df is None:
    st.error(
        f"'{CSV_FILENAME}' 파일을 찾을 수 없습니다. "
        "main.py와 같은 폴더(리포지토리 루트)에 CSV 파일을 함께 올려주세요."
    )
    st.stop()

# ----------------------------------------------------------------
# 인구 구조(비율) 행렬 계산 - 유사도 비교용
# ----------------------------------------------------------------
total_cols = [f"계_{b}" for b in bin_labels]
matrix = df[total_cols].values.astype(float)
row_sums = matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
proportions = matrix / row_sums  # 각 지역의 연령대별 인구 비율 (합=1)

# 코사인 유사도 계산을 위한 정규화
norms = np.linalg.norm(proportions, axis=1, keepdims=True)
norms[norms == 0] = 1
normalized = proportions / norms

# ----------------------------------------------------------------
# 사이드바 - 지역 검색
# ----------------------------------------------------------------
st.sidebar.header("⚙️ 설정")

level_options = ["시도", "시군구", "읍면동"]
level_filter = st.sidebar.multiselect("검색할 행정 단위", level_options, default=level_options)

search_text = st.sidebar.text_input("지역 이름 검색 (예: 강남, 해운대, 수원)", "")

candidates = df[df["레벨"].isin(level_filter)]
if search_text:
    candidates = candidates[candidates["지역명"].str.contains(search_text, na=False)]

if candidates.empty:
    st.sidebar.warning("검색 결과가 없습니다.")
    st.stop()

region_display = candidates["지역명"] + " · " + candidates["레벨"] + " (인구 " + candidates["총인구수"].map("{:,.0f}".format) + "명)"
region_map = dict(zip(region_display, candidates.index))

selected_display = st.sidebar.selectbox("기준 지역 선택", region_display.tolist())
selected_idx = region_map[selected_display]

same_level_only = st.sidebar.checkbox("같은 행정 단위끼리만 비교 (권장)", value=True)
top_n = st.sidebar.slider("쌍둥이 지역 후보 개수", min_value=3, max_value=20, value=5)

st.sidebar.markdown("---")
st.sidebar.caption(f"데이터 기준월: {prefix} · 총 {len(df):,}개 행정구역")

# ----------------------------------------------------------------
# 선택 지역 정보
# ----------------------------------------------------------------
base = df.loc[selected_idx]
st.subheader(f"📍 {base['지역명']} ({base['레벨']}) 인구 구조")

age_mid = np.arange(2, 101, 5)  # 각 구간의 중간 나이 근사치 (100세+ 구간은 102로 보정)
age_mid[-1] = 102
weighted_avg_age = float((proportions[selected_idx] * age_mid).sum())

youth_ratio = proportions[selected_idx][:3].sum() * 100      # 0-14세
working_ratio = proportions[selected_idx][3:13].sum() * 100  # 15-64세
senior_ratio = proportions[selected_idx][13:].sum() * 100    # 65세 이상
aging_index = (senior_ratio / youth_ratio * 100) if youth_ratio > 0 else float("nan")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("총 인구수", f"{base['총인구수']:,.0f}명")
m2.metric("근사 평균연령", f"{weighted_avg_age:.1f}세")
m3.metric("유소년 비율(0-14)", f"{youth_ratio:.1f}%")
m4.metric("고령 비율(65+)", f"{senior_ratio:.1f}%")
m5.metric("고령화지수", f"{aging_index:.0f}" if not np.isnan(aging_index) else "N/A")

# ----------------------------------------------------------------
# 인구 피라미드 (남/여)
# ----------------------------------------------------------------
male_vals = df.loc[selected_idx, [f"남_{b}" for b in bin_labels]].values.astype(float)
female_vals = df.loc[selected_idx, [f"여_{b}" for b in bin_labels]].values.astype(float)

pyramid_fig = go.Figure()
pyramid_fig.add_trace(go.Bar(
    y=bin_labels, x=-male_vals, name="남성", orientation="h",
    marker_color="#4c78a8",
    hovertemplate="%{y} 남성: %{customdata:,.0f}명<extra></extra>",
    customdata=male_vals,
))
pyramid_fig.add_trace(go.Bar(
    y=bin_labels, x=female_vals, name="여성", orientation="h",
    marker_color="#e45756",
    hovertemplate="%{y} 여성: %{x:,.0f}명<extra></extra>",
))
max_val = max(male_vals.max(), female_vals.max()) * 1.1 if len(male_vals) else 1
pyramid_fig.update_layout(
    title=f"{base['지역명']} 인구 피라미드",
    barmode="overlay",
    height=600,
    template="plotly_white",
    xaxis=dict(
        title="인구수",
        range=[-max_val, max_val],
        tickvals=np.linspace(-max_val, max_val, 7),
        ticktext=[f"{abs(v):,.0f}" for v in np.linspace(-max_val, max_val, 7)],
    ),
    yaxis=dict(title="연령대"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(pyramid_fig, use_container_width=True)

# ----------------------------------------------------------------
# 쌍둥이 지역 찾기 (코사인 유사도 기반)
# ----------------------------------------------------------------
st.divider()
st.subheader(f"👯 '{base['지역명']}'와(과) 인구 구조가 가장 비슷한 지역")

pool_mask = np.ones(len(df), dtype=bool)
pool_mask[selected_idx] = False
if same_level_only:
    pool_mask &= (df["레벨"] == base["레벨"]).values

sims = normalized @ normalized[selected_idx]
sims_masked = np.where(pool_mask, sims, -np.inf)
top_idx = np.argsort(sims_masked)[::-1][:top_n]

twin_df = pd.DataFrame({
    "지역명": df.loc[top_idx, "지역명"].values,
    "행정단위": df.loc[top_idx, "레벨"].values,
    "총인구수": df.loc[top_idx, "총인구수"].values,
    "유사도(%)": (sims[top_idx] * 100).round(2),
})

col_table, col_bar = st.columns([1.1, 1])
with col_table:
    st.dataframe(
        twin_df.style.format({"총인구수": "{:,.0f}", "유사도(%)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

with col_bar:
    bar_fig = go.Figure(go.Bar(
        x=twin_df["유사도(%)"][::-1],
        y=twin_df["지역명"][::-1],
        orientation="h",
        marker_color=twin_df["유사도(%)"][::-1],
        marker_colorscale="Viridis",
    ))
    bar_fig.update_layout(
        title="유사도 순위",
        height=380,
        template="plotly_white",
        xaxis_title="유사도(%)",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(bar_fig, use_container_width=True)

# ----------------------------------------------------------------
# 1위 쌍둥이 지역과 연령 구조 곡선 비교
# ----------------------------------------------------------------
twin_pick = st.selectbox("비교할 쌍둥이 지역 선택", twin_df["지역명"].tolist())
twin_idx = top_idx[twin_df["지역명"].tolist().index(twin_pick)]

compare_fig = go.Figure()
compare_fig.add_trace(go.Scatter(
    x=bin_labels, y=proportions[selected_idx] * 100, mode="lines+markers",
    name=base["지역명"], line=dict(color="#4c78a8", width=3),
))
compare_fig.add_trace(go.Scatter(
    x=bin_labels, y=proportions[twin_idx] * 100, mode="lines+markers",
    name=df.loc[twin_idx, "지역명"], line=dict(color="#e45756", width=3, dash="dash"),
))
compare_fig.update_layout(
    title=f"연령대별 인구 비율 비교: {base['지역명']} vs {df.loc[twin_idx, '지역명']}",
    xaxis_title="연령대",
    yaxis_title="비율(%)",
    height=450,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(compare_fig, use_container_width=True)

st.caption(
    "※ '인구 구조 유사도'는 연령대별 인구 비율(전체 대비 %) 분포의 코사인 유사도로 계산되며, "
    "지역의 절대 인구 규모와는 무관하게 '나이 구성 패턴'이 얼마나 비슷한지를 나타냅니다."
)

with st.expander("📋 전체 데이터 보기"):
    st.dataframe(df, use_container_width=True)
