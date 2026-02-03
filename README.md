# 국내영업 생산요청 대비 출고 관리 (Streamlit)

국내영업의 생산요청 수량 대비 실제 출고 현황을 확인하는 Streamlit 대시보드입니다.

## 주요 기능
- 제품명코드(품목코드 앞 4자리 S코드) 기준 요약
- 품목코드 상세(브랜드, P코드, 제품코드(마스터) 포함)
- 동일제품 통합(낱개 기준)
  - 집계 우선순위: 제품코드(마스터) -> P코드 -> 제품군명
- 통합 검색(OR)
- 상태 분류(미출고/출고중/출고완료/요청초과출고/요청없음)
- 모든 탭 엑셀 다운로드

## 실행 방법(로컬)
```bash
pip install -r requirements.txt
streamlit run app.py
```
또는 `run_streamlit.bat` 실행

## Streamlit Cloud 배포
1. Streamlit Cloud에서 `New app` 클릭
2. Repository: `ssungjun83/korean-sales-production`
3. Branch: `master`
4. Main file path: `app.py`
5. Deploy

## 데이터 파일
아래 파일이 프로젝트 루트에 있어야 합니다.
- `2026년 국내 생산 요청 수량 리스트.xlsx`
- `2026년 국내제품 입고수량.xlsx`
- `판매코드-제품코드 매칭 마스터 데이터.xlsx`

## 참고
- 엑셀 임시파일(`~$*.xlsx`)은 자동 제외됩니다.
- 캐시 이슈가 있으면 앱 우측 상단 메뉴에서 `Clear cache` 후 재실행하세요.
