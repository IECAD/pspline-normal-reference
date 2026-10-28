# 실험 1 보고서 (비교 모델 2차 + 문헌 대조 수정판)

작성: 2026-10-12 · 계획(사전 등록): `doc/실험1_비교모델_계획.md` "2차", "2차 수정", "추가 수정" · 문헌 대조: `doc/문헌대조_결과.md` · 코드: `src/baselines/models.py`, `experiments/exp1_baselines.py` (단계 2·3·4), `exp1_baselines_summarize.py` · 결과: 이 폴더(2차), `../batch2_fix/`(GDN·LapRPCA·RBC^φ·USADFIX), `../batch2_fix2/`(USADFIX2) · 1·2차 합친 표: `summary.md` · 위치화: `results/exp2/with_batch2/summary.md`

## 0. 요약 (테스트, 제안 모델 PS 대비, 허용오차 D1 0.0165 · D2 0.0171)
- **1·2차와 수정판을 합쳐도 PS를 허용오차 이상으로 이긴 모델은 없다.**
- **D1 (주입):** 2차 모델 대부분이 대등이다.
  - RRR −0.004, Transformer −0.003, RBC^φ −0.002, GDN −0.002, OmniAnomaly −0.001, Laplacian-RPCA +0.013, 2D Conv-AE +0.016
  - D1은 노드별 기준(nc 입력)만 주면 많은 모델이 비슷해지는 쉬운 과제다.
- **D2 (다지표 주입):** 대등은 노드별 ST-SSD형, 셀 평균, OLS뿐이다. 2차 모델은 모두 PS 우위다.
  - RRR +0.048, GDN +0.062, OmniAnomaly +0.065, USAD 원형 수정판 +0.31, 2D Conv-AE +0.37, Transformer +0.37, Laplacian-RPCA +0.44, RBC^φ +0.55
- **USAD 정정:** 1차의 "USAD 대등(D1)"은 학습 오류판(적대 항 상쇄)의 결과였다. 원형대로 고친 USADFIX2는 D1 +0.072, D2 +0.31로 PS 우위다.
- **위치화 (실험 2 재집계):**
  - HitRate에서 PS가 지는 경우는 없다 (모든 시나리오).
  - Smearing에서는 짧은 스파이크(SC3)에서 일부 모델이 약간 낮다: OmniAnomaly D2 SC3m 0.013, USADFIX2 D1 SC3 0.027, LSTM-AE.
  - 그 밖의 열세는 oracle뿐이다.
  - 센서 간 모델(Transformer 0.22, RBC^φ 0.37, GDN 0.15, RRR 0.14)은 D1 Smearing이 PS(0.074)보다 크다.
- **추론 비용:** PS 0.0034 ms/윈도우 (CPU). Transformer 0.37 ms, OmniAnomaly 23–33 ms (GPU, 표본 8개 평균)로 수천 배다.

## 1. 사전 예상과 비교
| 예상 (실행 전) | 결과 |
|---|---|
| 어떤 모델도 PS를 이기지 못한다 | 맞음 |
| 거의 모든 모델이 검증에서 nc를 고른다 | 맞음 (USADFIX의 D2만 raw) |
| GDN·Transformer(nc)는 D1에서 대등, D2에서 Smearing이 PS보다 클 것 | 맞음 (D2 Smearing GDN 0.104, Transformer 0.147 vs PS 0.089). D2 AUC-PR도 PS 우위 |
| OmniAnomaly·2D Conv-AE는 PS보다 낮을 것 | D2에서만 맞음. D1에서는 대등 |
| Laplacian-RPCA는 1차 RPCA와 비슷 | 틀림. D1은 높고(0.922 vs 0.785), D2는 낮다(0.466 vs 0.554) |
| USADFIX는 1차 USAD와 비슷하거나 약간 낮을 것 | 틀림. 검증·테스트에서 무너짐 (출력 무제한 → 발산 추정). Sigmoid 출력판(USADFIX2)으로 판정 |
| GDN 수정판은 누수판보다 낮을 것 | 검증에서 거의 같음 (D1 0.965 vs 0.964, D2 0.893 vs 0.905) |
| RBC^φ는 PCA-SPE와 비슷하거나 약간 높을 것 | 틀림. D1은 크게 높고(0.937 vs 0.703), D2는 크게 낮다(0.354 vs 0.865). D2 nc 입력에서 T² 항이 정상 변동에 민감한 것으로 보임 (추정) |

## 2. 해석
1. **노드별 정상 기준이 성능의 핵심이라는 1차 결론이 유지된다.** 같은 nc 입력을 쓰면, 센서 간 구조를 쓰는 모델(선형·딥러닝·그래프)이 D1에서는 PS와 같아지고 D2에서는 PS보다 낮아진다.
2. **위치화(번짐)에서는 행 독립인 PS가 유리하다.** AUC-PR이 대등한 센서 간 모델도 Smearing이 2–5배다.
3. 짧은 스파이크(SC3)에서는 평활하지 않거나 시간 문맥을 쓰는 모델이 셀 단위 지표에서 약간 낫다 (셀 평균, OmniAnomaly). 실험 2의 기존 관찰과 같다.

## 3. 한계
- GDN·OmniAnomaly는 재구현이다 (원 코드 아님). InterFusion은 빠졌다 (`doc/실험1_비교모델_계획.md` "2차" 1절).
- Laplacian-RPCA는 센서 그래프 변형이다 (원 논문은 표본 그래프).
- 딥러닝은 시드 1개, 구성 2–4개다. 데이터가 작아(노드당 윈도우 5–19개) 원 논문 설정보다 크게 줄였다.
- 수정판(GDN, LapRPCA, USADFIX, USADFIX2, RBC^φ)은 2차 검증 탐색 중·후의 문헌 대조로 추가했다. 일부 검증 결과를 본 뒤의 수정이고, 그 경위는 계획서 "2차 수정", "추가 수정"에 적혀 있다.
