import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------
# 페이지 설정
# ----------------------------------------------------------------
st.set_page_config(page_title="🇰🇷 연령별 인구현황 분석", page_icon="👥", layout="wide")
st.title("👥 행정구역별 연령별 인구현황 분석")
st.caption("행정안전부 주민등록 연령별 인구현황(월간) CSV 분석 대시보드")

GENDER_LABEL = {"계": "전체", "남": "남성", "여": "여성"}

# ----------------------------------------------------------------
# 파일 업로드
# ----------------------------------------------------------------
uploaded = st.file_uploader(
    "연령별인구현황 CSV 파일을 업로드하세요 (행정안전부 주민등록인구 통계)",
    type=["csv"],
)

if uploaded is None:
    st.info("👆 CSV 파일을 업로드하면 분석이 시작됩니다. (예: 202606_202606_연령별인구현황_월간.csv)")
    st.stop()


# ----------------------------------------------------------------
# 데이터 로딩 & 파싱
# ----------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data(file):
    # 이 통계 CSV는 EUC-KR/CP949 인코딩이며, 숫자에 천단위 콤마가 포함되어 있어
    # thousands="," 옵션으로 처음부터 숫자로 파싱한다. (콤마 때문에 str로 읽히던 문제 수정)
    last_err = None
    for enc in ["cp949", "euc-kr", "utf-8-sig", "utf-8"]:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc, thousands=",")
            return df, enc
        except Exception as e:
            last_err = e
    raise last_err


@st.cache_data(show_spinner=False)
def parse_age_columns(columns):
    """
    컬럼명 패턴: 'YYYY년MM월_성별_항목'
    항목: 총인구수 / 연령구간인구수 / 0세~99세 / 100세 이상
    -> 성별별로 '나이(정수) -> 컬럼명' 매핑과, 총인구수 컬럼명을 분리해서 반환.
    나이 컬럼과 age_mid 배열의 길이가 반드시 일치하도록 함께 구성한다.
    """
    pattern = re.compile(r"^(\d{4}년\d{2}월)_(계|남|여)_(.+)$")
    prefix = None
    age_map = {"계": {}, "남": {}, "여": {}}
    total_col = {"계": None, "남": None, "여": None}

    for col in columns:
        m = pattern.match(col)
        if not m:
            continue
        prefix = m.group(1)
        gender = m.group(2)
        item = m.group(3).strip()

        if item == "총인구수":
            total_col[gender] = col
        elif item == "연령구간인구수":
            continue  # 나이별 데이터가 아니므로 제외 (기존 에러의 원인 중 하나)
        elif item == "100세 이상":
            age_map[gender][100] = col
        else:
            am = re.match(r"^(\d+)세$", item)
            if am:
                age_map[gender][int(am.group(1))] = col

    # 나이 오름차순으로 정렬된 (age, colname) 리스트로 변환
    sorted_age_cols = {
        g: sorted(age_map[g].items(), key=lambda x: x[0]) for g in age_map
    }
    return prefix, sorted_age_cols, total_col


def weighted_avg_age(row, age_cols):
    """
    row: 데이터프레임의 한 행 (Series)
    age_cols: [(age:int, colname:str), ...] - 나이와 컬럼명이 1:1로 매칭된 리스트
    나이 컬럼 개수와 age_mid 배열 길이를 항상 동일하게 만들어서 broadcast 에러를 방지.
    """
    ages = np.array([a for a, _ in age_cols], dtype=float)
    pops = np.array([float(row[c]) for _, c in age_cols], dtype=float)
    total = pops.sum()
    if total <= 0:
        return np.nan
    proportions = pops / total
    return float((proportions * ages).sum())


def age_group_sum(row, age_cols, bin_size=5):
    """5세 단위로 인구를 묶어서 (구간 라벨, 인구수) 리스트 반환. 100세는 '100+'로 별도 처리."""
    buckets = {}
    for age, col in age_cols:
        if age >= 100:
            label = "100+"
        else:
            start = (age // bin_size) * bin_size
            label = f"{start}-{start + bin_size - 1}"
        buckets[label] = buckets.get(label, 0) + float(row[col])

    def sort_key(label):
        return 999 if label == "100+" else int(label.split("-")[0])

    ordered_labels = sorted(buckets.keys(), key=sort_key)
    return ordered_labels, [buckets[l] for l in ordered_labels]


# ----------------------------------------------------------------
# 데이터 로드 실행
# ----------------------------------------------------------------
try:
    df, used_encoding = load_data(uploaded)
except Exception as e:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

if "행정구역" not in df.columns:
    st.error("'행정구역' 컬럼을 찾을 수 없습니다. 올바른 인구현황 CSV 파일인지 확인해주세요.")
    st.stop()

prefix, age_cols_by_gender, total_col_by_gender = parse_age_columns(df.columns)

if prefix is None or not age_cols_by_gender["계"]:
    st.error("연령별 인구 컬럼을 파싱하지 못했습니다. 파일 형식을 확인해주세요.")
    st.stop()

# 컬럼-배열 길이 검증 (핵심 수정 지점)
n_age_cols = len(age_cols_by_gender["계"])
st.sidebar.caption(f"✅ 인식된 나이 구간 수: {n_age_cols}개 (0세~100세 이상)")

df["행정구역_표시"] = df["행정구역"].str.replace(r"\s+", " ", regex=True).str.strip()

# ----------------------------------------------------------------
# 사이드바 - 지역 선택
# ----------------------------------------------------------------
st.sidebar.header("⚙️ 설정")
st.sidebar.caption(f"기준 연월: {prefix} · 인코딩: {used_encoding}")

search = st.sidebar.text_input("지역명 검색", placeholder="예: 종로구, 해운대구 ...")
region_options = df["행정구역_표시"].tolist()
if search:
    region_options = [r for r in region_options if search in r]

if not region_options:
    st.sidebar.warning("검색 결과가 없습니다.")
    st.stop()

selected_regions = st.sidebar.multiselect(
    "분석할 지역 선택 (여러 개 선택 시 비교)",
    options=region_options,
    default=region_options[:1],
)

gender_choice = st.sidebar.radio("성별 기준", ["계", "남", "여"], format_func=lambda g: GENDER_LABEL[g], index=0)
bin_size = st.sidebar.slider("인구 피라미드 연령 구간(세)", min_value=1, max_value=10, value=5)

if not selected_regions:
    st.warning("사이드바에서 최소 하나의 지역을 선택해주세요.")
    st.stop()

sub_df = df[df["행정구역_표시"].isin(selected_regions)].copy()

# ----------------------------------------------------------------
# 지역별 핵심 지표 계산
# ----------------------------------------------------------------
age_cols = age_cols_by_gender[gender_choice]
total_col = total_col_by_gender[gender_choice]

results = []
for _, row in sub_df.iterrows():
    total_pop = float(row[total_col]) if total_col else np.nan
    avg_age = weighted_avg_age(row, age_cols)

    youth = sum(float(row[c]) for a, c in age_cols if a <= 14)
    elderly = sum(float(row[c]) for a, c in age_cols if a >= 65)
    working = sum(float(row[c]) for a, c in age_cols if 15 <= a <= 64)

    results.append({
        "지역": row["행정구역_표시"],
        "총인구": total_pop,
        "평균연령": round(avg_age, 1) if not np.isnan(avg_age) else None,
        "유소년비율(0-14세,%)": round(youth / total_pop * 100, 1) if total_pop else None,
        "생산연령비율(15-64세,%)": round(working / total_pop * 100, 1) if total_pop else None,
        "고령인구비율(65세+,%)": round(elderly / total_pop * 100, 1) if total_pop else None,
    })

result_df = pd.DataFrame(results)

# ----------------------------------------------------------------
# 첫 번째 지역 상세 지표 카드
# ----------------------------------------------------------------
main_row = result_df.iloc[0]
st.subheader(f"📍 {main_row['지역']} ({GENDER_LABEL[gender_choice]} 기준)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("총인구", f"{main_row['총인구']:,.0f}명" if main_row["총인구"] else "N/A")
c2.metric("평균연령", f"{main_row['평균연령']}세" if main_row["평균연령"] is not None else "N/A")
c3.metric("유소년비율(0-14세)", f"{main_row['유소년비율(0-14세,%)']}%")
c4.metric("고령인구비율(65세+)", f"{main_row['고령인구비율(65세+,%)']}%")

st.divider()

# ----------------------------------------------------------------
# 인구 피라미드 (첫 번째 지역, 남/여 비교)
# ----------------------------------------------------------------
st.subheader("🔺 인구 피라미드 (남 vs 여)")

pyramid_row = sub_df.iloc[0]
male_labels, male_values = age_group_sum(pyramid_row, age_cols_by_gender["남"], bin_size)
female_labels, female_values = age_group_sum(pyramid_row, age_cols_by_gender["여"], bin_size)

pyramid_fig = go.Figure()
pyramid_fig.add_trace(go.Bar(
    y=male_labels, x=[-v for v in male_values], name="남성",
    orientation="h", marker_color="#4c78a8",
    hovertemplate="%{y}세: %{customdata:,.0f}명<extra>남성</extra>",
    customdata=male_values,
))
pyramid_fig.add_trace(go.Bar(
    y=female_labels, x=female_values, name="여성",
    orientation="h", marker_color="#e45756",
    hovertemplate="%{y}세: %{x:,.0f}명<extra>여성</extra>",
))
pyramid_fig.update_layout(
    barmode="relative",
    height=700,
    template="plotly_white",
    xaxis_title="인구수",
    yaxis_title="연령대",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=30, b=10),
)
pyramid_fig.update_xaxes(tickvals=None)
st.plotly_chart(pyramid_fig, use_container_width=True)

# ----------------------------------------------------------------
# 지역 간 비교
# ----------------------------------------------------------------
if len(selected_regions) > 1:
    st.subheader("🔀 선택 지역 비교")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    comp_fig = go.Figure()
    for _, row in sub_df.iterrows():
        total = float(row[total_col])
        ages = [a for a, _ in age_cols]
        props = [float(row[c]) / total * 100 if total else 0 for _, c in age_cols]
        comp_fig.add_trace(go.Scatter(x=ages, y=props, mode="lines", name=row["행정구역_표시"]))

    comp_fig.update_layout(
        height=450,
        template="plotly_white",
        xaxis_title="나이",
        yaxis_title="비율 (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(comp_fig, use_container_width=True)
else:
    st.subheader("📈 연령별 인구 분포")
    ages = [a for a, _ in age_cols]
    values = [float(pyramid_row[c]) for _, c in age_cols]
    dist_fig = px.area(x=ages, y=values, labels={"x": "나이", "y": "인구수"})
    dist_fig.update_layout(height=400, template="plotly_white", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(dist_fig, use_container_width=True)

# ----------------------------------------------------------------
# 원본 데이터 & 다운로드
# ----------------------------------------------------------------
with st.expander("📋 계산 결과 데이터 보기"):
    st.dataframe(result_df, use_container_width=True, hide_index=True)
    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("결과 CSV 다운로드", data=csv, file_name="population_analysis_result.csv", mime="text/csv")

st.caption("데이터 출처: 행정안전부 주민등록 연령별 인구현황(월간)")
