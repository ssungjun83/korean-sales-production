from io import BytesIO
from pathlib import Path
import re
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="국내영업 생산요청 vs 출고 관리", layout="wide")
st.title("국내영업 생산요청 대비 출고 관리")
st.caption("기준: 품목코드 앞 4자리(예: S129) 동일 시 1 EA = 1 PACK, 분기 누적 출고 집계")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
    :root {
        --font: "Noto Sans KR", sans-serif;
    }
    html, body, .stApp, [class*="st-"], [class*="css"] {
        font-family: "Noto Sans KR", sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_FONT_FAMILY = "Noto Sans KR"


def normalize_name(name: object) -> str:
    return str(name).replace("\n", "").replace(" ", "").replace("\t", "").strip()


def build_colmap(df: pd.DataFrame) -> dict[str, str]:
    return {normalize_name(c): c for c in df.columns}


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cmap = build_colmap(df)
    for c in candidates:
        key = normalize_name(c)
        if key in cmap:
            return cmap[key]
    return None


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def normalize_code(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


PACK_COUNT_PATTERNS = [
    re.compile(r"(?<!\d)(\d{1,3})\s*팩"),
    re.compile(r"(?<!\d)(\d{1,3})\s*개입"),
    re.compile(r"(?<!\d)(\d{1,3})\s*P\b", re.IGNORECASE),
    re.compile(r"_(\d{1,3})(?:\b|$)"),
]


def extract_pack_count(name: object) -> float:
    if pd.isna(name):
        return np.nan
    text = str(name).strip()
    if not text:
        return np.nan
    for pattern in PACK_COUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            value = int(match.group(1))
            if value > 0:
                return float(value)
    return np.nan


def summarize_pack_counts(values: pd.Series, max_units: int = 4) -> str:
    nums = pd.to_numeric(values, errors="coerce").dropna()
    units = sorted({int(v) for v in nums if int(v) > 0})
    if not units:
        return ""
    if len(units) <= max_units:
        return ", ".join(str(v) for v in units)
    return ", ".join(str(v) for v in units[:max_units]) + f" 외 {len(units) - max_units}"


def summarize_names(values: pd.Series, max_names: int = 3) -> str:
    names = [str(v).strip() for v in values if pd.notna(v) and str(v).strip() and str(v).strip() != "0"]
    if not names:
        return ""
    seen = []
    for n in names:
        if n not in seen:
            seen.append(n)
    if len(seen) <= max_names:
        return ", ".join(seen)
    return ", ".join(seen[:max_names]) + f" 외 {len(seen) - max_names}"


def summarize_codes(values: pd.Series, max_codes: int = 5) -> str:
    codes = [str(v).strip() for v in values if pd.notna(v) and str(v).strip() and str(v).strip() != "0"]
    if not codes:
        return ""
    seen = []
    for c in codes:
        if c not in seen:
            seen.append(c)
    if len(seen) <= max_codes:
        return ", ".join(seen)
    return ", ".join(seen[:max_codes]) + f" 외 {len(seen) - max_codes}"


@st.cache_data
def load_data(base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    files = list(Path(base_dir).glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("현재 폴더에 xlsx 파일이 없습니다.")

    request_df = None
    inbound_df = None
    request_file = ""
    inbound_file = ""

    for file in files:
        df = pd.read_excel(file)
        cmap = build_colmap(df)
        has_item = normalize_name("품목코드") in cmap
        has_qty = normalize_name("수량") in cmap
        has_request_qty = any("요청수량" in normalize_name(c) for c in df.columns)

        if has_item and has_request_qty and not has_qty:
            request_df = df
            request_file = file.name
        elif has_item and has_qty:
            inbound_df = df
            inbound_file = file.name

    if request_df is None or inbound_df is None:
        raise ValueError("요청 파일/입고 파일을 자동 식별하지 못했습니다. xlsx 파일 구조를 확인해주세요.")

    return request_df, inbound_df, request_file, inbound_file


def prepare_request(df: pd.DataFrame) -> pd.DataFrame:
    year_col = find_col(df, ["년"])
    quarter_col = find_col(df, ["분기"])
    item_col = find_col(df, ["품목코드"])
    name_col = find_col(df, ["품명"])
    pcode_col = find_col(df, ["P 코드", "P코드"])

    if not (year_col and quarter_col and item_col):
        raise ValueError("요청 파일에 필수 컬럼(년/분기/품목코드)이 없습니다.")

    qty_cols = [c for c in df.columns if "요청수량" in normalize_name(c)]
    if not qty_cols:
        raise ValueError("요청 파일에서 요청수량 컬럼을 찾지 못했습니다.")

    quarter_total_cols = [c for c in qty_cols if "분기" in normalize_name(c)]

    req = df.copy()
    req["년"] = pd.to_numeric(req[year_col], errors="coerce").astype("Int64")
    req["분기"] = pd.to_numeric(req[quarter_col], errors="coerce").astype("Int64")
    req["품목코드"] = normalize_code(req[item_col])
    req["제품코드"] = req["품목코드"].str[:4]
    req["P코드"] = req[pcode_col] if pcode_col else ""
    req["품명"] = req[name_col] if name_col else ""
    req["PACK당낱개수"] = req["품명"].apply(extract_pack_count)

    if quarter_total_cols:
        req["요청수량_PACK"] = to_numeric(req[quarter_total_cols[0]])
    else:
        qty_numeric = req[qty_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        req["요청수량_PACK"] = qty_numeric.sum(axis=1)

    return req[["년", "분기", "품목코드", "제품코드", "P코드", "품명", "PACK당낱개수", "요청수량_PACK"]]


def prepare_inbound(df: pd.DataFrame) -> pd.DataFrame:
    year_col = find_col(df, ["년"])
    quarter_col = find_col(df, ["분기"])
    item_col = find_col(df, ["품목코드"])
    qty_col = find_col(df, ["수량"])
    date_col = find_col(df, ["이동일자"])
    name_col = find_col(df, ["품명"])

    if not (year_col and quarter_col and item_col and qty_col):
        raise ValueError("입고 파일에 필수 컬럼(년/분기/품목코드/수량)이 없습니다.")

    inbound = df.copy()
    inbound["년"] = pd.to_numeric(inbound[year_col], errors="coerce").astype("Int64")
    inbound["분기"] = pd.to_numeric(inbound[quarter_col], errors="coerce").astype("Int64")
    inbound["품목코드"] = normalize_code(inbound[item_col])
    inbound["제품코드"] = inbound["품목코드"].str[:4]
    inbound["품명"] = inbound[name_col] if name_col else ""
    inbound["PACK당낱개수"] = inbound["품명"].apply(extract_pack_count)
    inbound["출고수량_EA"] = to_numeric(inbound[qty_col])
    inbound["이동일자"] = pd.to_datetime(inbound[date_col], errors="coerce") if date_col else pd.NaT

    return inbound[["년", "분기", "품목코드", "제품코드", "품명", "PACK당낱개수", "이동일자", "출고수량_EA"]]


def status_label(request_qty: pd.Series, shipped_qty: pd.Series) -> pd.Series:
    return np.select(
        [
            (request_qty == 0) & (shipped_qty > 0),
            (request_qty > 0) & (shipped_qty == 0),
            (request_qty > 0) & (shipped_qty < request_qty),
            (request_qty > 0) & (shipped_qty == request_qty),
            (request_qty > 0) & (shipped_qty > request_qty),
        ],
        [
            "요청없음(출고발생)",
            "미출고",
            "출고중",
            "출고완료",
            "요청초과출고",
        ],
        default="확인필요",
    )


def add_progress_columns(df: pd.DataFrame, req_col: str, ship_col: str) -> pd.DataFrame:
    out = df.copy()
    req = pd.to_numeric(out[req_col], errors="coerce").fillna(0)
    ship_total = pd.to_numeric(out[ship_col], errors="coerce").fillna(0)
    ship_matched = np.minimum(req, ship_total)
    ship_excess = np.maximum(ship_total - req, 0)

    out["총출고수량_EA"] = ship_total
    out["매칭출고수량_EA"] = ship_matched
    out["초과출고수량_EA"] = ship_excess
    out["잔량"] = np.maximum(req - ship_matched, 0)
    out["진행률(%)"] = np.where(req > 0, (ship_matched / req) * 100, np.nan)
    out["상태"] = status_label(req, ship_total)
    return out


def format_table(
    df: pd.DataFrame,
    int_cols: list[str],
    pct_cols: list[str] | None = None,
    progress_bar_cols: list[str] | None = None,
):
    pct_cols = pct_cols or []
    progress_bar_cols = progress_bar_cols or []
    fmt = {c: "{:,.0f}" for c in int_cols if c in df.columns}
    fmt.update({c: "{:,.1f}" for c in pct_cols if c in df.columns})
    styler = df.style.format(fmt, na_rep="")
    for c in progress_bar_cols:
        if c in df.columns:
            # 100%를 가득 찬 기준으로 시각화(초과값은 막대가 가득 찬 상태로 표시)
            styler = styler.bar(subset=[c], vmin=0, vmax=100, color="#93c5fd")
    return styler


def parse_search_terms(query: str) -> list[str]:
    # 공백은 제품명 내부 문자로 취급하고, 구분자는 쉼표/|/ 로만 사용한다.
    terms = [t.strip() for t in re.split(r"[,|/]+", str(query)) if t.strip()]
    # keep input order while removing duplicates
    return list(dict.fromkeys(terms))


def apply_or_search(df: pd.DataFrame, query: str, columns: list[str]) -> pd.DataFrame:
    terms = parse_search_terms(query)
    use_cols = [c for c in columns if c in df.columns]
    if not terms or not use_cols:
        return df

    mask = pd.Series(False, index=df.index)
    for term in terms:
        term_mask = pd.Series(False, index=df.index)
        for col in use_cols:
            term_mask = term_mask | df[col].astype(str).str.contains(term, case=False, na=False, regex=False)
        mask = mask | term_mask
    return df[mask]


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()


def apply_chart_style(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            labelFont=APP_FONT_FAMILY,
            titleFont=APP_FONT_FAMILY,
            labelFontSize=13,
            titleFontSize=15,
        )
        .configure_legend(
            labelFont=APP_FONT_FAMILY,
            titleFont=APP_FONT_FAMILY,
            labelFontSize=13,
            titleFontSize=14,
        )
        .configure_title(
            font=APP_FONT_FAMILY,
            fontSize=20,
        )
    )


try:
    req_raw, inbound_raw, req_file, inbound_file = load_data(".")
    req = prepare_request(req_raw)
    inbound = prepare_inbound(inbound_raw)
except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()

# 사이드바 필터
all_years = sorted(pd.concat([req["년"].dropna(), inbound["년"].dropna()]).astype(int).unique().tolist())
all_quarters = sorted(pd.concat([req["분기"].dropna(), inbound["분기"].dropna()]).astype(int).unique().tolist())

year_options = ["전체"] + all_years
quarter_options = ["전체"] + all_quarters

st.sidebar.header("조회 조건")
selected_year = st.sidebar.selectbox("년", year_options, index=1 if len(year_options) > 1 else 0)
selected_quarter = st.sidebar.selectbox("분기", quarter_options, index=1 if len(quarter_options) > 1 else 0)

st.sidebar.markdown("---")
st.sidebar.write(f"요청 파일: `{req_file}`")
st.sidebar.write(f"입고 파일: `{inbound_file}`")

req_f = req.copy()
in_f = inbound.copy()
if selected_year != "전체":
    req_f = req_f[req_f["년"] == selected_year]
    in_f = in_f[in_f["년"] == selected_year]
if selected_quarter != "전체":
    req_f = req_f[req_f["분기"] == selected_quarter]
    in_f = in_f[in_f["분기"] == selected_quarter]

# PACK 기준 수량을 낱개 기준으로 환산
req_f["요청수량_낱개환산"] = req_f["요청수량_PACK"] * req_f["PACK당낱개수"]
in_f["출고수량_낱개환산"] = in_f["출고수량_EA"] * in_f["PACK당낱개수"]

# 제품코드(앞4자리) 기준 요약
req_prod = req_f.groupby(["년", "분기", "제품코드"], as_index=False)["요청수량_PACK"].sum()
in_prod = in_f.groupby(["년", "분기", "제품코드"], as_index=False)["출고수량_EA"].sum()
prod = req_prod.merge(in_prod, on=["년", "분기", "제품코드"], how="outer").fillna(0)

pcode_prod = (
    req_f.groupby(["년", "분기", "제품코드"], as_index=False)["P코드"]
    .apply(summarize_codes)
    .rename(columns={"P코드": "P코드"})
)
prod = prod.merge(pcode_prod, on=["년", "분기", "제품코드"], how="left")
prod["P코드"] = prod["P코드"].fillna("")

name_src = pd.concat(
    [
        req_f[["년", "분기", "제품코드", "품명"]],
        in_f[["년", "분기", "제품코드", "품명"]],
    ],
    ignore_index=True,
)
prod_names = (
    name_src.groupby(["년", "분기", "제품코드"], as_index=False)["품명"]
    .apply(summarize_names)
    .rename(columns={"품명": "품명"})
)
prod = prod.merge(prod_names, on=["년", "분기", "제품코드"], how="left")
prod["품명"] = prod["품명"].fillna("")

pack_src_prod = pd.concat(
    [
        req_f[["년", "분기", "제품코드", "PACK당낱개수"]],
        in_f[["년", "분기", "제품코드", "PACK당낱개수"]],
    ],
    ignore_index=True,
)
prod_pack = (
    pack_src_prod.groupby(["년", "분기", "제품코드"], as_index=False)["PACK당낱개수"]
    .apply(summarize_pack_counts)
    .rename(columns={"PACK당낱개수": "PACK당낱개수"})
)
prod = prod.merge(prod_pack, on=["년", "분기", "제품코드"], how="left")
prod["PACK당낱개수"] = prod["PACK당낱개수"].fillna("")

req_piece_prod = (
    req_f.groupby(["년", "분기", "제품코드"], as_index=False)["요청수량_낱개환산"]
    .sum(min_count=1)
    .rename(columns={"요청수량_낱개환산": "요청수량_낱개"})
)
in_piece_prod = (
    in_f.groupby(["년", "분기", "제품코드"], as_index=False)["출고수량_낱개환산"]
    .sum(min_count=1)
    .rename(columns={"출고수량_낱개환산": "출고수량_낱개"})
)
prod = prod.merge(req_piece_prod, on=["년", "분기", "제품코드"], how="left")
prod = prod.merge(in_piece_prod, on=["년", "분기", "제품코드"], how="left")
prod_req_piece = pd.to_numeric(prod["요청수량_낱개"], errors="coerce").fillna(0)
prod_ship_piece = pd.to_numeric(prod["출고수량_낱개"], errors="coerce").fillna(0)
prod["매칭출고수량_낱개"] = np.minimum(prod_req_piece, prod_ship_piece)
prod["초과출고수량_낱개"] = np.maximum(prod_ship_piece - prod_req_piece, 0)
prod["잔량_낱개"] = np.maximum(prod_req_piece - prod_ship_piece, 0)

prod = add_progress_columns(prod, "요청수량_PACK", "출고수량_EA")

# 품목코드 상세 요약
req_item = req_f.groupby(["년", "분기", "품목코드"], as_index=False).agg(
    품명=("품명", "first"), 요청수량_PACK=("요청수량_PACK", "sum")
)
in_item = in_f.groupby(["년", "분기", "품목코드"], as_index=False).agg(
    품명=("품명", "first"), 출고수량_EA=("출고수량_EA", "sum")
)
item = req_item.merge(in_item, on=["년", "분기", "품목코드"], how="outer", suffixes=("_요청", "_출고")).fillna(0)
item["품명"] = np.where(item["품명_요청"].astype(str) != "0", item["품명_요청"], item["품명_출고"])
item["제품코드"] = item["품목코드"].astype(str).str[:4]
pcode_item = (
    req_f.groupby(["년", "분기", "품목코드"], as_index=False)["P코드"]
    .apply(summarize_codes)
    .rename(columns={"P코드": "P코드"})
)
item = item.merge(pcode_item, on=["년", "분기", "품목코드"], how="left")
item["P코드"] = item["P코드"].fillna("")

pack_src_item = pd.concat(
    [
        req_f[["년", "분기", "품목코드", "PACK당낱개수"]],
        in_f[["년", "분기", "품목코드", "PACK당낱개수"]],
    ],
    ignore_index=True,
)
item_pack = (
    pack_src_item.groupby(["년", "분기", "품목코드"], as_index=False)["PACK당낱개수"]
    .apply(summarize_pack_counts)
    .rename(columns={"PACK당낱개수": "PACK당낱개수"})
)
item = item.merge(item_pack, on=["년", "분기", "품목코드"], how="left")
item["PACK당낱개수"] = item["PACK당낱개수"].fillna("")

req_piece_item = (
    req_f.groupby(["년", "분기", "품목코드"], as_index=False)["요청수량_낱개환산"]
    .sum(min_count=1)
    .rename(columns={"요청수량_낱개환산": "요청수량_낱개"})
)
in_piece_item = (
    in_f.groupby(["년", "분기", "품목코드"], as_index=False)["출고수량_낱개환산"]
    .sum(min_count=1)
    .rename(columns={"출고수량_낱개환산": "출고수량_낱개"})
)
item = item.merge(req_piece_item, on=["년", "분기", "품목코드"], how="left")
item = item.merge(in_piece_item, on=["년", "분기", "품목코드"], how="left")
item_req_piece = pd.to_numeric(item["요청수량_낱개"], errors="coerce").fillna(0)
item_ship_piece = pd.to_numeric(item["출고수량_낱개"], errors="coerce").fillna(0)
item["매칭출고수량_낱개"] = np.minimum(item_req_piece, item_ship_piece)
item["초과출고수량_낱개"] = np.maximum(item_ship_piece - item_req_piece, 0)
item["잔량_낱개"] = np.maximum(item_req_piece - item_ship_piece, 0)

item = add_progress_columns(item, "요청수량_PACK", "출고수량_EA")

global_search = st.text_input(
    "통합 검색 (OR)",
    "",
    placeholder="예: S036, Bandage, 미출고",
)
st.caption("쉼표(,) 또는 | 또는 / 로 키워드를 구분하면 OR 조건으로 검색합니다.")

# KPI (통합 검색 입력 시 검색결과 기준 요약)
kpi_source = item.copy()
has_global_terms = bool(parse_search_terms(global_search))
if has_global_terms:
    kpi_source = apply_or_search(
        kpi_source,
        global_search,
        ["제품코드", "품목코드", "P코드", "품명", "상태", "PACK당낱개수", "년", "분기"],
    )

total_req = float(pd.to_numeric(kpi_source["요청수량_PACK"], errors="coerce").fillna(0).sum())
total_ship_total = float(pd.to_numeric(kpi_source["총출고수량_EA"], errors="coerce").fillna(0).sum())
total_ship_matched = float(pd.to_numeric(kpi_source["매칭출고수량_EA"], errors="coerce").fillna(0).sum())
total_ship_excess = float(pd.to_numeric(kpi_source["초과출고수량_EA"], errors="coerce").fillna(0).sum())
total_remaining = total_req - total_ship_matched
progress_pct = (total_ship_matched / total_req * 100) if total_req > 0 else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("요청수량 (PACK)", f"{total_req:,.0f}")
c2.metric("매칭출고수량 (EA)", f"{total_ship_matched:,.0f}")
c3.metric("초과출고수량 (EA)", f"{total_ship_excess:,.0f}")
c4.metric("총출고수량 (EA)", f"{total_ship_total:,.0f}")
c5.metric("잔량 (요청-매칭)", f"{total_remaining:,.0f}")
c6.metric("진행률(매칭기준)", f"{progress_pct:,.1f}%")

item_count = len(kpi_source)
product_count = kpi_source["제품코드"].astype(str).replace("nan", "").replace("", np.nan).nunique(dropna=True)
status_counts = kpi_source["상태"].value_counts()
status_text = ", ".join([f"{k} {v}건" for k, v in status_counts.items()][:4]) if not status_counts.empty else "없음"
scope_text = "통합 검색 결과 요약" if has_global_terms else "전체 요약"
st.caption(f"{scope_text} | 품목 {item_count:,}건 | 제품코드 {product_count:,}개 | 상태분포: {status_text}")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["제품코드 요약", "품목코드 상세", "분기 누적 추이"])

with tab1:
    show_short_only = st.checkbox("미달(출고중/미출고)만 보기", value=False)
    prod_view = prod.copy()
    prod_view = apply_or_search(prod_view, global_search, ["제품코드", "P코드", "품명", "상태", "PACK당낱개수", "년", "분기"])
    if show_short_only:
        prod_view = prod_view[prod_view["상태"].isin(["미출고", "출고중"])]

    prod_view = prod_view.sort_values(["상태", "잔량"], ascending=[True, False])
    prod_cols = [
        "년",
        "분기",
        "제품코드",
        "P코드",
        "품명",
        "PACK당낱개수",
        "요청수량_PACK",
        "총출고수량_EA",
        "매칭출고수량_EA",
        "초과출고수량_EA",
        "잔량",
        "요청수량_낱개",
        "출고수량_낱개",
        "매칭출고수량_낱개",
        "초과출고수량_낱개",
        "잔량_낱개",
        "진행률(%)",
        "상태",
    ]
    st.dataframe(
        format_table(
            prod_view[prod_cols],
            int_cols=[
                "년",
                "분기",
                "요청수량_PACK",
                "총출고수량_EA",
                "매칭출고수량_EA",
                "초과출고수량_EA",
                "잔량",
                "요청수량_낱개",
                "출고수량_낱개",
                "매칭출고수량_낱개",
                "초과출고수량_낱개",
                "잔량_낱개",
            ],
            pct_cols=["진행률(%)"],
            progress_bar_cols=["진행률(%)"],
        ),
        use_container_width=True,
        hide_index=True,
    )
    excel_data_prod = to_excel_bytes(prod_view[prod_cols], sheet_name="제품코드요약")
    st.download_button(
        "제품코드 요약 엑셀 다운로드",
        data=excel_data_prod,
        file_name="제품코드_요약_요청대비출고.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    chart_all = prod_view.sort_values("잔량", ascending=False).copy()
    shortage_all = chart_all[chart_all["잔량"] > 0].copy()
    default_top_n = min(20, max(1, len(shortage_all))) if len(shortage_all) > 0 else 1

    c_left, c_right = st.columns([2, 1])
    with c_left:
        chart_mode = st.radio(
            "그래프 표시 범위",
            ["전체 제품", "상위 부족 제품"],
            horizontal=True,
            key="chart_mode",
        )
    with c_right:
        top_n = st.number_input(
            "상위 부족 제품 개수",
            min_value=1,
            max_value=max(1, len(shortage_all)),
            value=default_top_n,
            step=1,
            key="chart_top_n",
        )

    if chart_mode == "전체 제품":
        chart_source = chart_all
        chart_title = f"전체 제품 잔량 그래프 ({len(chart_source):,}개)"
    else:
        chart_source = shortage_all.head(int(top_n))
        chart_title = f"상위 부족 {int(top_n):,}개 제품"

    st.subheader(chart_title)
    if chart_source.empty:
        st.info("표시할 제품이 없습니다.")
    else:
        chart_df = chart_source[["제품코드", "품명", "잔량"]].copy()
        chart_df["대표품명"] = chart_df["품명"].astype(str).str.split(",").str[0].str.strip()
        chart_df["대표품명"] = chart_df["대표품명"].replace("", "(품명없음)")
        chart_df["그래프표시명"] = chart_df.apply(
            lambda r: f"{r['제품코드']} | {r['대표품명'][:24]}{'...' if len(r['대표품명']) > 24 else ''}",
            axis=1,
        )
        y_max = float(pd.to_numeric(chart_df["잔량"], errors="coerce").fillna(0).max())
        y_domain_max = max(1.0, y_max * 1.12)
        x_enc = alt.X(
            "그래프표시명:N",
            sort="-y",
            title="제품코드 | 제품명",
            axis=alt.Axis(labelAngle=-90, labelPadding=8, labelLimit=220),
        )
        y_enc = alt.Y("잔량:Q", title="잔량", scale=alt.Scale(domain=[0, y_domain_max]))
        bar = alt.Chart(chart_df).mark_bar().encode(
            x=x_enc,
            y=y_enc,
            tooltip=[
                alt.Tooltip("제품코드:N", title="제품코드"),
                alt.Tooltip("품명:N", title="품명"),
                alt.Tooltip("잔량:Q", title="잔량", format=","),
            ],
        )
        label = alt.Chart(chart_df).mark_text(
            dy=-10,
            font=APP_FONT_FAMILY,
            fontSize=14,
            fontWeight="bold",
            color="#111827",
        ).encode(
            x=x_enc,
            y=alt.Y("잔량:Q", scale=alt.Scale(domain=[0, y_domain_max])),
            text=alt.Text("잔량:Q", format=","),
        )
        chart_height = max(520, min(1400, 26 * len(chart_df)))
        bar_chart = apply_chart_style(
            (bar + label).properties(
                height=chart_height,
                padding={"top": 50, "bottom": 170, "left": 20, "right": 20},
            )
        )
        st.altair_chart(bar_chart, use_container_width=True)

with tab2:
    item_view = item.copy()
    item_view = apply_or_search(
        item_view,
        global_search,
        ["제품코드", "품목코드", "P코드", "품명", "상태", "PACK당낱개수", "년", "분기"],
    )

    item_view = item_view.sort_values(["상태", "잔량"], ascending=[True, False])
    item_cols = [
        "년",
        "분기",
        "제품코드",
        "P코드",
        "품목코드",
        "품명",
        "PACK당낱개수",
        "요청수량_PACK",
        "총출고수량_EA",
        "매칭출고수량_EA",
        "초과출고수량_EA",
        "잔량",
        "요청수량_낱개",
        "출고수량_낱개",
        "매칭출고수량_낱개",
        "초과출고수량_낱개",
        "잔량_낱개",
        "진행률(%)",
        "상태",
    ]
    st.dataframe(
        format_table(
            item_view[item_cols],
            int_cols=[
                "년",
                "분기",
                "요청수량_PACK",
                "총출고수량_EA",
                "매칭출고수량_EA",
                "초과출고수량_EA",
                "잔량",
                "요청수량_낱개",
                "출고수량_낱개",
                "매칭출고수량_낱개",
                "초과출고수량_낱개",
                "잔량_낱개",
            ],
            pct_cols=["진행률(%)"],
            progress_bar_cols=["진행률(%)"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    excel_data = to_excel_bytes(item_view[item_cols], sheet_name="품목코드상세")
    st.download_button(
        "품목코드 상세 엑셀 다운로드",
        data=excel_data,
        file_name="품목코드_요청대비출고_상세.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with tab3:
    st.write("분기 내 일자별 누적 출고량 추이")
    daily = in_f.dropna(subset=["이동일자"]).groupby("이동일자", as_index=True)["출고수량_EA"].sum().sort_index().cumsum()
    trend_export_df = pd.DataFrame(columns=["일자", "누적출고(EA)", "요청수량(PACK)"])
    if daily.empty:
        st.info("표시할 이동일자 데이터가 없습니다.")
    else:
        trend_df = pd.DataFrame({"누적출고(EA)": daily})
        if total_req > 0:
            trend_df["요청수량(PACK)"] = total_req
        trend_reset = trend_df.reset_index()
        date_col = trend_reset.columns[0]
        trend_reset = trend_reset.rename(columns={date_col: "일자"})
        trend_export_df = trend_reset.copy()
        trend_long = trend_reset.melt(id_vars="일자", var_name="구분", value_name="수량")

        line = alt.Chart(trend_long).mark_line(point=True).encode(
            x=alt.X("일자:T", title="이동일자"),
            y=alt.Y("수량:Q", title="수량"),
            color=alt.Color("구분:N", title="지표"),
            tooltip=[
                alt.Tooltip("일자:T", title="일자"),
                alt.Tooltip("구분:N", title="지표"),
                alt.Tooltip("수량:Q", title="수량", format=","),
            ],
        )
        last_points = trend_long.sort_values("일자").groupby("구분", as_index=False).tail(1)
        labels = alt.Chart(last_points).mark_text(
            dx=6,
            align="left",
            font=APP_FONT_FAMILY,
            fontSize=14,
            fontWeight="bold",
        ).encode(
            x=alt.X("일자:T"),
            y=alt.Y("수량:Q"),
            color=alt.Color("구분:N", legend=None),
            text=alt.Text("수량:Q", format=","),
        )
        line_chart = apply_chart_style((line + labels).properties(height=360))
        st.altair_chart(line_chart, use_container_width=True)

    excel_data_trend = to_excel_bytes(trend_export_df, sheet_name="분기누적추이")
    st.download_button(
        "분기 누적 추이 엑셀 다운로드",
        data=excel_data_trend,
        file_name="분기_누적출고_추이.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.caption("입고 원본은 사용자 요청에 따라 중복 제거 없이 그대로 집계합니다.")
