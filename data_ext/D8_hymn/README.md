# D8 — HYMN (Hybrid Indoor and Indoor-Outdoor Positioning Multi-Technology Dataset), UWB 부분

- 출처: TU Dresden, Chair of Transport Systems Information Technology (TUD-ITVS). Zenodo DOI 10.5281/zenodo.17979434 (v1.3, 2026-04-14; 파일 레코드 19566178). GitHub `TUD-ITVS/HYMN-dataset`.
- 기술 논문: "Descriptor: A Hybrid Indoor and Indoor-Outdoor Positioning Multi-Technology Dataset (HYMN)", arXiv 2604.20349. 벤치마크: arXiv 2609.25835 (IPIN 2026).
- 라이선스: 데이터 CC BY 4.0 (`LICENSE-DATA`), 코드 MIT (`LICENSE-CODE`).
- 파일: `HYMN-dataset-1.3.zip` (63,031,322 B, MD5 `4874f6c931561eaaa17bb0972a052e42`, Zenodo 기재값과 일치). **읽기 전용. 압축을 풀 때는 `data_ext/D8_hymn/x/` 아래에.** 내려받은 날 2026-10-14.
- 쓰는 부분: `data/raw/uwb/*.sqlite` (지점당 1개, LOG.MESSAGE = JSON 라운드, 앵커 10개 항목, `value` m, 실패 = −70, `accuracy`), `data/reference/csv/{anchor_coordinates,point_coordinates,time_reference}.csv`. 전처리 참고: `preprocessing/src/uwb/loadData.py`, 앵커 ID 매핑 `preprocessing/src/preprocess_uwb.py` (`anchor_mapping`).
- 주 태그 `8121069331292423818` ("Zigpos links"). 보조 태그 `8121069331292422716` 은 결측 81 %.
- 구조 점검과 판정: `doc/공개데이터_조사_2026-10-14.md` §3. 실험 계획(사전 등록 예정): `doc/실험12_계획.md`.
