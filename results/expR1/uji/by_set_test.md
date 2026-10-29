# 실험 R1-6 사후 분석: 테스트 세트별 (탐색, 판정에 쓰지 않음)

재현 확인: 네 세트를 합친 값과 등록 실행의 차이 — AUC-PR 최대 8.33e-17, 오경보율 최대 9.71e-17 (짝 1080개)

## 조건 N (자연 결측)

평균 셀 AUC-PR

ref                                       CF     CG     CI     GM     NM     PS    PS1   PS1H    PSH
group                                                                                               
all four (registered)                  0.606  0.603  0.606  0.265  0.613  0.609  0.613  0.605  0.601
same directions (trn14, trn15, tst01)  0.843  0.841  0.843  0.277  0.851  0.848  0.850  0.847  0.846
tst05 (other directions)               0.420  0.415  0.419  0.273  0.428  0.421  0.427  0.420  0.415

실현 셀 오경보율

ref                                       CF     CG     CI     GM     NM     PS    PS1   PS1H    PSH
group                                                                                               
all four (registered)                  0.062  0.055  0.062  0.035  0.064  0.062  0.064  0.063  0.062
same directions (trn14, trn15, tst01)  0.029  0.024  0.028  0.035  0.029  0.028  0.029  0.028  0.028
tst05 (other directions)               0.167  0.153  0.167  0.035  0.174  0.169  0.173  0.173  0.171

- same directions (trn14, trn15, tst01): PS − GM +0.5717 (양 1.00, n 30)
- same directions (trn14, trn15, tst01): PS − CG +0.0078 (양 0.93, n 30)
- same directions (trn14, trn15, tst01): PS − CI +0.0058 (양 0.90, n 30)
- same directions (trn14, trn15, tst01): PS − NM -0.0028 (양 0.13, n 30)
- tst05 (other directions): PS − GM +0.1476 (양 1.00, n 30)
- tst05 (other directions): PS − CG +0.0063 (양 0.80, n 30)
- tst05 (other directions): PS − CI +0.0017 (양 0.73, n 30)
- tst05 (other directions): PS − NM -0.0072 (양 0.03, n 30)
- all four (registered): PS − GM +0.3439 (양 1.00, n 30)
- all four (registered): PS − CG +0.0059 (양 0.87, n 30)
- all four (registered): PS − CI +0.0025 (양 0.80, n 30)
- all four (registered): PS − NM -0.0044 (양 0.00, n 30)

## 조건 M (+70 % 학습 칸 손실)

평균 셀 AUC-PR

ref                                       CF     CG     CI     GM     NM     PS    PS1   PS1H    PSH
group                                                                                               
all four (registered)                  0.522  0.454  0.522  0.244  0.574  0.557  0.573  0.561  0.549
same directions (trn14, trn15, tst01)  0.699  0.583  0.697  0.254  0.804  0.780  0.803  0.792  0.775
tst05 (other directions)               0.368  0.326  0.369  0.252  0.397  0.383  0.396  0.383  0.376

실현 셀 오경보율

ref                                       CF     CG     CI     GM     NM     PS    PS1   PS1H    PSH
group                                                                                               
all four (registered)                  0.052  0.039  0.052  0.034  0.059  0.054  0.059  0.056  0.054
same directions (trn14, trn15, tst01)  0.028  0.022  0.028  0.034  0.026  0.025  0.027  0.025  0.024
tst05 (other directions)               0.130  0.093  0.129  0.034  0.161  0.147  0.161  0.156  0.148

- same directions (trn14, trn15, tst01): PS − GM +0.5262 (양 1.00, n 90)
- same directions (trn14, trn15, tst01): PS − CG +0.1971 (양 1.00, n 90)
- same directions (trn14, trn15, tst01): PS − CI +0.0831 (양 1.00, n 90)
- same directions (trn14, trn15, tst01): PS − NM -0.0239 (양 0.00, n 90)
- tst05 (other directions): PS − GM +0.1308 (양 1.00, n 90)
- tst05 (other directions): PS − CG +0.0569 (양 0.97, n 90)
- tst05 (other directions): PS − CI +0.0136 (양 0.79, n 90)
- tst05 (other directions): PS − NM -0.0137 (양 0.11, n 90)
- all four (registered): PS − GM +0.3132 (양 1.00, n 90)
- all four (registered): PS − CG +0.1029 (양 1.00, n 90)
- all four (registered): PS − CI +0.0355 (양 0.98, n 90)
- all four (registered): PS − NM -0.0170 (양 0.02, n 90)
