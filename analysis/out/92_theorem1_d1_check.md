# 정리 1 회고 검증 — D1 (네 참조, 실험 6 보간 조건)

x = σ_R 비율² − 1 ∝ V_e − V_e0 (명제 4′; 방마다 상수배). AUC-PR 은 실험 6 보간 실험의 저장값(장면 SC1–SC3, 주입 시드 평균).

## (a) (방, 조건, 시드) 안에서 AUC-PR 순서 대 x 순서 (참조 cell·cellfill·cellinterp·PS)

| 방 | 조건 | 시드 | Spearman val | Spearman test | AUC(test) 순서 | x 순서(작은 것부터) | x: cell / fill / interp / PS |
|---|---|---|---|---|---|---|---|
| classroom | r=0.3 | 0 | +1.00 | +0.80 | PS > cellinterp > cellfill > cell | PS < cellfill < cellinterp < cell | 0.10 / 0.08 / 0.08 / 0.01 |
| classroom | r=0.3 | 1 | +0.80 | +0.80 | PS > cellfill > cellinterp > cell | PS < cellinterp < cellfill < cell | 0.36 / 0.10 / 0.10 / 0.04 |
| classroom | r=0.3 | 2 | +1.00 | +0.80 | PS > cellinterp > cellfill > cell | PS < cellfill < cellinterp < cell | 0.11 / 0.08 / 0.08 / 0.02 |
| classroom | r=0.5 | 0 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 2.38 / 0.17 / 0.18 / -0.00 |
| classroom | r=0.5 | 1 | +1.00 | +0.80 | PS > cellfill > cellinterp > cell | PS < cellinterp < cellfill < cell | 2.08 / 0.17 / 0.17 / 0.07 |
| classroom | r=0.5 | 2 | +0.80 | +0.80 | PS > cellfill > cellinterp > cell | PS < cellinterp < cellfill < cell | 1.14 / 0.16 / 0.16 / 0.03 |
| classroom | r=0.7 | 0 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 12.35 / 0.32 / 0.37 / 0.07 |
| classroom | r=0.7 | 1 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 11.30 / 0.34 / 0.38 / 0.11 |
| classroom | r=0.7 | 2 | +0.80 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 8.70 / 0.32 / 0.35 / 0.08 |
| classroom | pi=0.3,L=4 | 0 | +1.00 | +0.80 | PS > cellinterp > cellfill > cell | PS < cellfill < cellinterp < cell | 0.24 / 0.06 / 0.06 / 0.00 |
| classroom | pi=0.3,L=4 | 1 | +1.00 | +0.40 | cellfill > cellinterp > PS > cell | PS < cellfill < cellinterp < cell | 0.13 / 0.08 / 0.08 / 0.00 |
| classroom | pi=0.3,L=4 | 2 | +0.80 | +0.80 | PS > cellfill > cellinterp > cell | PS < cellinterp < cellfill < cell | 0.12 / 0.10 / 0.10 / -0.01 |
| classroom | pi=0.5,L=4 | 0 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 2.34 / 0.24 / 0.25 / 0.06 |
| classroom | pi=0.5,L=4 | 1 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 1.67 / 0.15 / 0.15 / 0.00 |
| classroom | pi=0.5,L=4 | 2 | +0.80 | +0.80 | PS > cellinterp > cellfill > cell | PS < cellfill < cellinterp < cell | 1.61 / 0.21 / 0.21 / 0.00 |
| classroom | pi=0.5,L=8 | 0 | +1.00 | +0.80 | cellfill > PS > cellinterp > cell | PS < cellfill < cellinterp < cell | 1.39 / 0.24 / 0.25 / 0.03 |
| classroom | pi=0.5,L=8 | 1 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 2.67 / 0.16 / 0.18 / 0.06 |
| classroom | pi=0.5,L=8 | 2 | +1.00 | +0.40 | cellfill > cellinterp > PS > cell | PS < cellfill < cellinterp < cell | 1.98 / 0.24 / 0.25 / 0.05 |
| classroom | pi=0.7,L=4 | 0 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 9.60 / 0.28 / 0.33 / 0.07 |
| classroom | pi=0.7,L=4 | 1 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 10.74 / 0.30 / 0.33 / 0.06 |
| classroom | pi=0.7,L=4 | 2 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 9.85 / 0.36 / 0.40 / 0.10 |
| hallway | r=0.3 | 0 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.14 / 0.14 / 0.14 / 0.04 |
| hallway | r=0.3 | 1 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.15 / 0.15 / 0.15 / 0.04 |
| hallway | r=0.3 | 2 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.23 / 0.23 / 0.23 / 0.03 |
| hallway | r=0.5 | 0 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.31 / 0.31 / 0.31 / 0.11 |
| hallway | r=0.5 | 1 | +nan | -1.00 | PS > cell > cellfill > cellinterp | cell < cellfill < cellinterp < PS | -0.09 / -0.09 / -0.09 / -0.02 |
| hallway | r=0.5 | 2 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.38 / 0.38 / 0.38 / 0.11 |
| hallway | r=0.7 | 0 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.55 / 0.55 / 0.55 / 0.18 |
| hallway | r=0.7 | 1 | +0.77 | +0.95 | cellfill > cellinterp > PS > cell | cellinterp < cellfill < PS < cell | 0.40 / 0.00 / 0.00 / 0.01 |
| hallway | r=0.7 | 2 | +0.80 | +0.80 | PS > cellfill > cellinterp > cell | PS < cellinterp < cellfill < cell | 0.81 / 0.68 / 0.68 / 0.32 |
| hallway | pi=0.3,L=4 | 0 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.27 / 0.27 / 0.27 / 0.07 |
| hallway | pi=0.3,L=4 | 1 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.11 / 0.11 / 0.11 / 0.02 |
| hallway | pi=0.3,L=4 | 2 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.06 / 0.06 / 0.06 / 0.03 |
| hallway | pi=0.5,L=4 | 0 | +nan | +1.00 | cell > cellfill > cellinterp > PS | cell < cellfill < cellinterp < PS | -0.10 / -0.10 / -0.10 / -0.02 |
| hallway | pi=0.5,L=4 | 1 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.24 / 0.24 / 0.24 / 0.06 |
| hallway | pi=0.5,L=4 | 2 | +nan | -1.00 | PS > cell > cellfill > cellinterp | cell < cellfill < cellinterp < PS | -0.11 / -0.11 / -0.11 / -0.03 |
| hallway | pi=0.5,L=8 | 0 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.48 / 0.48 / 0.48 / 0.19 |
| hallway | pi=0.5,L=8 | 1 | +nan | -1.00 | PS > cell > cellfill > cellinterp | cell < cellfill < cellinterp < PS | -0.07 / -0.07 / -0.07 / 0.00 |
| hallway | pi=0.5,L=8 | 2 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.34 / 0.34 / 0.34 / 0.15 |
| hallway | pi=0.7,L=4 | 0 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 5.95 / 0.79 / 0.80 / 0.41 |
| hallway | pi=0.7,L=4 | 1 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.39 / 0.39 / 0.39 / 0.15 |
| hallway | pi=0.7,L=4 | 2 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.24 / 0.24 / 0.24 / 0.05 |
| lab | r=0.3 | 0 | +0.95 | +0.95 | PS > cellfill > cellinterp > cell | PS < cellinterp < cellfill < cell | 1.09 / 0.29 / 0.29 / 0.02 |
| lab | r=0.3 | 1 | +0.95 | +0.40 | cellfill > cellinterp > PS > cell | PS < cellfill < cellinterp < cell | 0.61 / 0.15 / 0.15 / 0.02 |
| lab | r=0.3 | 2 | +1.00 | +1.00 | PS > cell > cellfill > cellinterp | PS < cell < cellfill < cellinterp | 0.13 / 0.13 / 0.13 / 0.05 |
| lab | r=0.5 | 0 | +1.00 | +0.80 | PS > cellfill > cellinterp > cell | PS < cellinterp < cellfill < cell | 22.11 / 0.43 / 0.43 / 0.09 |
| lab | r=0.5 | 1 | +0.80 | +0.80 | PS > cellinterp > cellfill > cell | PS < cellfill < cellinterp < cell | 18.13 / 0.44 / 0.44 / 0.11 |
| lab | r=0.5 | 2 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 22.05 / 0.24 / 0.24 / 0.03 |
| lab | r=0.7 | 0 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 146.43 / 0.67 / 0.70 / 0.20 |
| lab | r=0.7 | 1 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 174.62 / 0.72 / 0.74 / 0.26 |
| lab | r=0.7 | 2 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 146.79 / 0.53 / 0.57 / 0.05 |
| lab | pi=0.3,L=4 | 0 | +1.00 | +0.95 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 0.67 / 0.11 / 0.11 / 0.02 |
| lab | pi=0.3,L=4 | 1 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 0.69 / 0.09 / 0.09 / 0.03 |
| lab | pi=0.3,L=4 | 2 | +0.95 | +0.80 | PS > cellinterp > cellfill > cell | PS < cellfill < cellinterp < cell | 1.48 / 0.18 / 0.18 / 0.04 |
| lab | pi=0.5,L=4 | 0 | +1.00 | +0.80 | PS > cellinterp > cellfill > cell | PS < cellfill < cellinterp < cell | 6.74 / 0.18 / 0.18 / 0.07 |
| lab | pi=0.5,L=4 | 1 | +0.80 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 20.65 / 0.27 / 0.28 / 0.08 |
| lab | pi=0.5,L=4 | 2 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 22.42 / 0.17 / 0.17 / 0.03 |
| lab | pi=0.5,L=8 | 0 | +0.80 | +0.80 | PS > cellinterp > cellfill > cell | PS < cellfill < cellinterp < cell | 3.85 / 0.21 / 0.22 / 0.06 |
| lab | pi=0.5,L=8 | 1 | +0.80 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 27.40 / 0.34 / 0.36 / 0.15 |
| lab | pi=0.5,L=8 | 2 | +0.80 | +0.80 | PS > cellinterp > cellfill > cell | PS < cellfill < cellinterp < cell | 14.03 / 0.26 / 0.26 / 0.11 |
| lab | pi=0.7,L=4 | 0 | +0.80 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 141.98 / 0.43 / 0.43 / 0.16 |
| lab | pi=0.7,L=4 | 1 | +1.00 | +0.80 | PS > cellinterp > cellfill > cell | PS < cellfill < cellinterp < cell | 111.06 / 0.34 / 0.36 / 0.00 |
| lab | pi=0.7,L=4 | 2 | +1.00 | +1.00 | PS > cellfill > cellinterp > cell | PS < cellfill < cellinterp < cell | 224.66 / 0.42 / 0.46 / 0.16 |

조건 63개. Spearman 중앙값 val +nan / test +1.00; ρ ≥ 0.8 인 조건 val 46, test 40.

## (b) 방마다 (조건, 시드, 참조) 점: AUC-PR(test) = Φ0 + κ x, 참조 더미 F 검정

| 방 | 점 | x 범위 | Φ0 | κ | R² | 잔차 SD | 더미 F | p | 더미 계수 (fill, interp, PS; cell 기준) |
|---|---|---|---|---|---|---|---|---|---|
| classroom | 78 | -0.01–1.98 | 0.881 | -0.178 | 0.944 | 0.0164 | 6.48 | 0.000597 | +0.0243, +0.0254, +0.0201 |
| hallway | 87 | -0.11–0.81 | 0.957 | -0.013 | 0.776 | 0.0016 | 13.54 | 2.93e-07 | +0.0003, +0.0003, +0.0024 |
| lab | 73 | 0.00–1.48 | 0.974 | -0.034 | 0.697 | 0.0061 | 14.75 | 1.69e-07 | +0.0141, +0.0142, +0.0119 |

## 측정값

| 방 | 조건 | 시드 | 참조 | σ_R 비율 | x | AUC val | AUC test |
|---|---|---|---|---|---|---|---|
| classroom | r=0.3 | 0 | PS | 1.007 | +0.013 | 0.919 | 0.874 |
| classroom | r=0.3 | 0 | cell | 1.048 | +0.099 | 0.884 | 0.849 |
| classroom | r=0.3 | 0 | cellfill | 1.042 | +0.085 | 0.889 | 0.866 |
| classroom | r=0.3 | 0 | cellinterp | 1.042 | +0.085 | 0.889 | 0.866 |
| classroom | r=0.3 | 1 | PS | 1.020 | +0.040 | 0.911 | 0.875 |
| classroom | r=0.3 | 1 | cell | 1.168 | +0.365 | 0.817 | 0.772 |
| classroom | r=0.3 | 1 | cellfill | 1.049 | +0.101 | 0.887 | 0.856 |
| classroom | r=0.3 | 1 | cellinterp | 1.049 | +0.100 | 0.887 | 0.855 |
| classroom | r=0.3 | 2 | PS | 1.012 | +0.023 | 0.917 | 0.878 |
| classroom | r=0.3 | 2 | cell | 1.055 | +0.112 | 0.888 | 0.843 |
| classroom | r=0.3 | 2 | cellfill | 1.040 | +0.082 | 0.902 | 0.859 |
| classroom | r=0.3 | 2 | cellinterp | 1.040 | +0.082 | 0.902 | 0.859 |
| classroom | r=0.5 | 0 | PS | 0.998 | -0.005 | 0.922 | 0.873 |
| classroom | r=0.5 | 0 | cell | 1.838 | +2.380 | 0.600 | 0.550 |
| classroom | r=0.5 | 0 | cellfill | 1.080 | +0.167 | 0.875 | 0.836 |
| classroom | r=0.5 | 0 | cellinterp | 1.086 | +0.180 | 0.874 | 0.835 |
| classroom | r=0.5 | 1 | PS | 1.034 | +0.069 | 0.913 | 0.875 |
| classroom | r=0.5 | 1 | cell | 1.755 | +2.081 | 0.617 | 0.560 |
| classroom | r=0.5 | 1 | cellfill | 1.084 | +0.175 | 0.899 | 0.822 |
| classroom | r=0.5 | 1 | cellinterp | 1.083 | +0.174 | 0.900 | 0.821 |
| classroom | r=0.5 | 2 | PS | 1.014 | +0.028 | 0.915 | 0.874 |
| classroom | r=0.5 | 2 | cell | 1.464 | +1.145 | 0.676 | 0.615 |
| classroom | r=0.5 | 2 | cellfill | 1.078 | +0.163 | 0.899 | 0.847 |
| classroom | r=0.5 | 2 | cellinterp | 1.078 | +0.161 | 0.899 | 0.847 |
| classroom | r=0.7 | 0 | PS | 1.033 | +0.068 | 0.919 | 0.868 |
| classroom | r=0.7 | 0 | cell | 3.654 | +12.354 | 0.425 | 0.376 |
| classroom | r=0.7 | 0 | cellfill | 1.150 | +0.322 | 0.871 | 0.825 |
| classroom | r=0.7 | 0 | cellinterp | 1.170 | +0.370 | 0.861 | 0.825 |
| classroom | r=0.7 | 1 | PS | 1.055 | +0.112 | 0.906 | 0.863 |
| classroom | r=0.7 | 1 | cell | 3.507 | +11.296 | 0.405 | 0.359 |
| classroom | r=0.7 | 1 | cellfill | 1.160 | +0.345 | 0.870 | 0.829 |
| classroom | r=0.7 | 1 | cellinterp | 1.173 | +0.375 | 0.869 | 0.825 |
| classroom | r=0.7 | 2 | PS | 1.038 | +0.078 | 0.907 | 0.865 |
| classroom | r=0.7 | 2 | cell | 3.114 | +8.696 | 0.426 | 0.395 |
| classroom | r=0.7 | 2 | cellfill | 1.150 | +0.321 | 0.885 | 0.827 |
| classroom | r=0.7 | 2 | cellinterp | 1.162 | +0.349 | 0.887 | 0.823 |
| classroom | pi=0.3,L=4 | 0 | PS | 1.001 | +0.002 | 0.921 | 0.876 |
| classroom | pi=0.3,L=4 | 0 | cell | 1.112 | +0.237 | 0.837 | 0.802 |
| classroom | pi=0.3,L=4 | 0 | cellfill | 1.031 | +0.063 | 0.896 | 0.873 |
| classroom | pi=0.3,L=4 | 0 | cellinterp | 1.031 | +0.064 | 0.896 | 0.873 |
| classroom | pi=0.3,L=4 | 1 | PS | 1.001 | +0.003 | 0.918 | 0.876 |
| classroom | pi=0.3,L=4 | 1 | cell | 1.062 | +0.127 | 0.858 | 0.836 |
| classroom | pi=0.3,L=4 | 1 | cellfill | 1.038 | +0.078 | 0.891 | 0.877 |
| classroom | pi=0.3,L=4 | 1 | cellinterp | 1.038 | +0.078 | 0.891 | 0.876 |
| classroom | pi=0.3,L=4 | 2 | PS | 0.995 | -0.010 | 0.911 | 0.882 |
| classroom | pi=0.3,L=4 | 2 | cell | 1.056 | +0.116 | 0.861 | 0.868 |
| classroom | pi=0.3,L=4 | 2 | cellfill | 1.047 | +0.097 | 0.869 | 0.876 |
| classroom | pi=0.3,L=4 | 2 | cellinterp | 1.047 | +0.097 | 0.869 | 0.876 |
| classroom | pi=0.5,L=4 | 0 | PS | 1.028 | +0.057 | 0.917 | 0.871 |
| classroom | pi=0.5,L=4 | 0 | cell | 1.828 | +2.341 | 0.640 | 0.575 |
| classroom | pi=0.5,L=4 | 0 | cellfill | 1.113 | +0.240 | 0.887 | 0.843 |
| classroom | pi=0.5,L=4 | 0 | cellinterp | 1.116 | +0.246 | 0.887 | 0.842 |
| classroom | pi=0.5,L=4 | 1 | PS | 1.002 | +0.005 | 0.918 | 0.882 |
| classroom | pi=0.5,L=4 | 1 | cell | 1.633 | +1.667 | 0.625 | 0.576 |
| classroom | pi=0.5,L=4 | 1 | cellfill | 1.072 | +0.149 | 0.892 | 0.867 |
| classroom | pi=0.5,L=4 | 1 | cellinterp | 1.074 | +0.154 | 0.892 | 0.865 |
| classroom | pi=0.5,L=4 | 2 | PS | 1.001 | +0.002 | 0.912 | 0.876 |
| classroom | pi=0.5,L=4 | 2 | cell | 1.615 | +1.609 | 0.645 | 0.596 |
| classroom | pi=0.5,L=4 | 2 | cellfill | 1.099 | +0.207 | 0.854 | 0.856 |
| classroom | pi=0.5,L=4 | 2 | cellinterp | 1.100 | +0.211 | 0.854 | 0.857 |
| classroom | pi=0.5,L=8 | 0 | PS | 1.017 | +0.034 | 0.916 | 0.873 |
| classroom | pi=0.5,L=8 | 0 | cell | 1.546 | +1.391 | 0.666 | 0.632 |
| classroom | pi=0.5,L=8 | 0 | cellfill | 1.112 | +0.237 | 0.872 | 0.874 |
| classroom | pi=0.5,L=8 | 0 | cellinterp | 1.116 | +0.245 | 0.872 | 0.873 |
| classroom | pi=0.5,L=8 | 1 | PS | 1.028 | +0.058 | 0.919 | 0.871 |
| classroom | pi=0.5,L=8 | 1 | cell | 1.916 | +2.671 | 0.583 | 0.539 |
| classroom | pi=0.5,L=8 | 1 | cellfill | 1.079 | +0.163 | 0.900 | 0.857 |
| classroom | pi=0.5,L=8 | 1 | cellinterp | 1.087 | +0.182 | 0.899 | 0.854 |
| classroom | pi=0.5,L=8 | 2 | PS | 1.023 | +0.047 | 0.901 | 0.869 |
| classroom | pi=0.5,L=8 | 2 | cell | 1.727 | +1.982 | 0.590 | 0.555 |
| classroom | pi=0.5,L=8 | 2 | cellfill | 1.112 | +0.236 | 0.849 | 0.880 |
| classroom | pi=0.5,L=8 | 2 | cellinterp | 1.117 | +0.248 | 0.847 | 0.879 |
| classroom | pi=0.7,L=4 | 0 | PS | 1.034 | +0.069 | 0.892 | 0.853 |
| classroom | pi=0.7,L=4 | 0 | cell | 3.256 | +9.603 | 0.419 | 0.373 |
| classroom | pi=0.7,L=4 | 0 | cellfill | 1.132 | +0.282 | 0.850 | 0.820 |
| classroom | pi=0.7,L=4 | 0 | cellinterp | 1.153 | +0.330 | 0.845 | 0.817 |
| classroom | pi=0.7,L=4 | 1 | PS | 1.028 | +0.058 | 0.911 | 0.869 |
| classroom | pi=0.7,L=4 | 1 | cell | 3.426 | +10.737 | 0.411 | 0.392 |
| classroom | pi=0.7,L=4 | 1 | cellfill | 1.139 | +0.297 | 0.868 | 0.820 |
| classroom | pi=0.7,L=4 | 1 | cellinterp | 1.155 | +0.333 | 0.860 | 0.815 |
| classroom | pi=0.7,L=4 | 2 | PS | 1.050 | +0.102 | 0.901 | 0.863 |
| classroom | pi=0.7,L=4 | 2 | cell | 3.294 | +9.851 | 0.419 | 0.386 |
| classroom | pi=0.7,L=4 | 2 | cellfill | 1.166 | +0.360 | 0.856 | 0.833 |
| classroom | pi=0.7,L=4 | 2 | cellinterp | 1.184 | +0.403 | 0.850 | 0.832 |
| classroom | 기준 | -1 | PS | 1.000 | +0.000 | 0.919 | 0.877 |
| classroom | 기준 | -1 | cell | 1.000 | +0.000 | 0.901 | 0.884 |
| classroom | 기준 | -1 | cellfill | 1.000 | +0.000 | 0.901 | 0.884 |
| classroom | 기준 | -1 | cellinterp | 1.000 | +0.000 | 0.901 | 0.884 |
| hallway | r=0.3 | 0 | PS | 1.019 | +0.038 | 1.000 | 0.958 |
| hallway | r=0.3 | 0 | cell | 1.068 | +0.141 | 0.999 | 0.955 |
| hallway | r=0.3 | 0 | cellfill | 1.068 | +0.141 | 0.999 | 0.955 |
| hallway | r=0.3 | 0 | cellinterp | 1.068 | +0.141 | 0.999 | 0.955 |
| hallway | r=0.3 | 1 | PS | 1.022 | +0.045 | 1.000 | 0.958 |
| hallway | r=0.3 | 1 | cell | 1.074 | +0.154 | 0.998 | 0.955 |
| hallway | r=0.3 | 1 | cellfill | 1.074 | +0.154 | 0.998 | 0.955 |
| hallway | r=0.3 | 1 | cellinterp | 1.074 | +0.154 | 0.998 | 0.955 |
| hallway | r=0.3 | 2 | PS | 1.017 | +0.033 | 1.000 | 0.958 |
| hallway | r=0.3 | 2 | cell | 1.107 | +0.226 | 0.997 | 0.953 |
| hallway | r=0.3 | 2 | cellfill | 1.107 | +0.226 | 0.997 | 0.953 |
| hallway | r=0.3 | 2 | cellinterp | 1.107 | +0.226 | 0.997 | 0.953 |
| hallway | r=0.5 | 0 | PS | 1.052 | +0.106 | 1.000 | 0.957 |
| hallway | r=0.5 | 0 | cell | 1.145 | +0.310 | 0.996 | 0.952 |
| hallway | r=0.5 | 0 | cellfill | 1.145 | +0.310 | 0.996 | 0.952 |
| hallway | r=0.5 | 0 | cellinterp | 1.145 | +0.310 | 0.996 | 0.952 |
| hallway | r=0.5 | 1 | PS | 0.991 | -0.019 | 1.000 | 0.958 |
| hallway | r=0.5 | 1 | cell | 0.954 | -0.090 | 1.000 | 0.958 |
| hallway | r=0.5 | 1 | cellfill | 0.954 | -0.090 | 1.000 | 0.958 |
| hallway | r=0.5 | 1 | cellinterp | 0.954 | -0.090 | 1.000 | 0.958 |
| hallway | r=0.5 | 2 | PS | 1.053 | +0.108 | 1.000 | 0.958 |
| hallway | r=0.5 | 2 | cell | 1.177 | +0.385 | 0.995 | 0.951 |
| hallway | r=0.5 | 2 | cellfill | 1.177 | +0.385 | 0.995 | 0.951 |
| hallway | r=0.5 | 2 | cellinterp | 1.177 | +0.385 | 0.995 | 0.951 |
| hallway | r=0.7 | 0 | PS | 1.088 | +0.183 | 1.000 | 0.957 |
| hallway | r=0.7 | 0 | cell | 1.243 | +0.545 | 0.992 | 0.948 |
| hallway | r=0.7 | 0 | cellfill | 1.243 | +0.545 | 0.992 | 0.948 |
| hallway | r=0.7 | 0 | cellinterp | 1.243 | +0.545 | 0.992 | 0.948 |
| hallway | r=0.7 | 1 | PS | 1.006 | +0.011 | 1.000 | 0.958 |
| hallway | r=0.7 | 1 | cell | 1.184 | +0.401 | 0.997 | 0.952 |
| hallway | r=0.7 | 1 | cellfill | 1.001 | +0.002 | 1.000 | 0.958 |
| hallway | r=0.7 | 1 | cellinterp | 1.001 | +0.002 | 1.000 | 0.958 |
| hallway | r=0.7 | 2 | PS | 1.149 | +0.321 | 1.000 | 0.957 |
| hallway | r=0.7 | 2 | cell | 1.346 | +0.812 | 0.990 | 0.945 |
| hallway | r=0.7 | 2 | cellfill | 1.298 | +0.684 | 0.992 | 0.946 |
| hallway | r=0.7 | 2 | cellinterp | 1.297 | +0.683 | 0.992 | 0.946 |
| hallway | pi=0.3,L=4 | 0 | PS | 1.034 | +0.070 | 1.000 | 0.957 |
| hallway | pi=0.3,L=4 | 0 | cell | 1.126 | +0.268 | 0.996 | 0.952 |
| hallway | pi=0.3,L=4 | 0 | cellfill | 1.126 | +0.268 | 0.996 | 0.952 |
| hallway | pi=0.3,L=4 | 0 | cellinterp | 1.126 | +0.268 | 0.996 | 0.952 |
| hallway | pi=0.3,L=4 | 1 | PS | 1.009 | +0.018 | 1.000 | 0.958 |
| hallway | pi=0.3,L=4 | 1 | cell | 1.055 | +0.113 | 0.999 | 0.956 |
| hallway | pi=0.3,L=4 | 1 | cellfill | 1.055 | +0.113 | 0.999 | 0.956 |
| hallway | pi=0.3,L=4 | 1 | cellinterp | 1.055 | +0.113 | 0.999 | 0.956 |
| hallway | pi=0.3,L=4 | 2 | PS | 1.017 | +0.034 | 1.000 | 0.958 |
| hallway | pi=0.3,L=4 | 2 | cell | 1.029 | +0.059 | 0.999 | 0.957 |
| hallway | pi=0.3,L=4 | 2 | cellfill | 1.029 | +0.059 | 0.999 | 0.957 |
| hallway | pi=0.3,L=4 | 2 | cellinterp | 1.029 | +0.059 | 0.999 | 0.957 |
| hallway | pi=0.5,L=4 | 0 | PS | 0.989 | -0.022 | 1.000 | 0.958 |
| hallway | pi=0.5,L=4 | 0 | cell | 0.950 | -0.098 | 1.000 | 0.958 |
| hallway | pi=0.5,L=4 | 0 | cellfill | 0.950 | -0.098 | 1.000 | 0.958 |
| hallway | pi=0.5,L=4 | 0 | cellinterp | 0.950 | -0.098 | 1.000 | 0.958 |
| hallway | pi=0.5,L=4 | 1 | PS | 1.029 | +0.060 | 1.000 | 0.957 |
| hallway | pi=0.5,L=4 | 1 | cell | 1.115 | +0.242 | 0.997 | 0.954 |
| hallway | pi=0.5,L=4 | 1 | cellfill | 1.115 | +0.242 | 0.997 | 0.954 |
| hallway | pi=0.5,L=4 | 1 | cellinterp | 1.115 | +0.242 | 0.997 | 0.954 |
| hallway | pi=0.5,L=4 | 2 | PS | 0.985 | -0.031 | 1.000 | 0.958 |
| hallway | pi=0.5,L=4 | 2 | cell | 0.946 | -0.105 | 1.000 | 0.958 |
| hallway | pi=0.5,L=4 | 2 | cellfill | 0.946 | -0.105 | 1.000 | 0.958 |
| hallway | pi=0.5,L=4 | 2 | cellinterp | 0.946 | -0.105 | 1.000 | 0.958 |
| hallway | pi=0.5,L=8 | 0 | PS | 1.091 | +0.191 | 1.000 | 0.957 |
| hallway | pi=0.5,L=8 | 0 | cell | 1.218 | +0.485 | 0.993 | 0.949 |
| hallway | pi=0.5,L=8 | 0 | cellfill | 1.218 | +0.485 | 0.993 | 0.949 |
| hallway | pi=0.5,L=8 | 0 | cellinterp | 1.218 | +0.485 | 0.993 | 0.949 |
| hallway | pi=0.5,L=8 | 1 | PS | 1.002 | +0.005 | 1.000 | 0.958 |
| hallway | pi=0.5,L=8 | 1 | cell | 0.965 | -0.068 | 1.000 | 0.958 |
| hallway | pi=0.5,L=8 | 1 | cellfill | 0.965 | -0.068 | 1.000 | 0.958 |
| hallway | pi=0.5,L=8 | 1 | cellinterp | 0.965 | -0.068 | 1.000 | 0.958 |
| hallway | pi=0.5,L=8 | 2 | PS | 1.073 | +0.151 | 1.000 | 0.957 |
| hallway | pi=0.5,L=8 | 2 | cell | 1.159 | +0.343 | 0.994 | 0.951 |
| hallway | pi=0.5,L=8 | 2 | cellfill | 1.159 | +0.343 | 0.994 | 0.951 |
| hallway | pi=0.5,L=8 | 2 | cellinterp | 1.159 | +0.343 | 0.994 | 0.951 |
| hallway | pi=0.7,L=4 | 0 | PS | 1.186 | +0.406 | 1.000 | 0.956 |
| hallway | pi=0.7,L=4 | 0 | cell | 2.636 | +5.947 | 0.896 | 0.874 |
| hallway | pi=0.7,L=4 | 0 | cellfill | 1.338 | +0.791 | 0.988 | 0.953 |
| hallway | pi=0.7,L=4 | 0 | cellinterp | 1.340 | +0.796 | 0.988 | 0.953 |
| hallway | pi=0.7,L=4 | 1 | PS | 1.072 | +0.150 | 1.000 | 0.957 |
| hallway | pi=0.7,L=4 | 1 | cell | 1.180 | +0.393 | 0.994 | 0.951 |
| hallway | pi=0.7,L=4 | 1 | cellfill | 1.180 | +0.393 | 0.994 | 0.951 |
| hallway | pi=0.7,L=4 | 1 | cellinterp | 1.180 | +0.393 | 0.994 | 0.951 |
| hallway | pi=0.7,L=4 | 2 | PS | 1.025 | +0.051 | 1.000 | 0.958 |
| hallway | pi=0.7,L=4 | 2 | cell | 1.114 | +0.241 | 0.999 | 0.954 |
| hallway | pi=0.7,L=4 | 2 | cellfill | 1.114 | +0.241 | 0.999 | 0.954 |
| hallway | pi=0.7,L=4 | 2 | cellinterp | 1.114 | +0.241 | 0.999 | 0.954 |
| hallway | 기준 | -1 | PS | 1.000 | +0.000 | 1.000 | 0.958 |
| hallway | 기준 | -1 | cell | 1.000 | +0.000 | 1.000 | 0.957 |
| hallway | 기준 | -1 | cellfill | 1.000 | +0.000 | 1.000 | 0.957 |
| hallway | 기준 | -1 | cellinterp | 1.000 | +0.000 | 1.000 | 0.957 |
| lab | r=0.3 | 0 | PS | 1.008 | +0.017 | 0.992 | 0.970 |
| lab | r=0.3 | 0 | cell | 1.445 | +1.089 | 0.950 | 0.925 |
| lab | r=0.3 | 0 | cellfill | 1.135 | +0.287 | 0.982 | 0.967 |
| lab | r=0.3 | 0 | cellinterp | 1.135 | +0.287 | 0.982 | 0.967 |
| lab | r=0.3 | 1 | PS | 1.011 | +0.023 | 0.992 | 0.969 |
| lab | r=0.3 | 1 | cell | 1.269 | +0.611 | 0.975 | 0.953 |
| lab | r=0.3 | 1 | cellfill | 1.071 | +0.147 | 0.989 | 0.970 |
| lab | r=0.3 | 1 | cellinterp | 1.071 | +0.147 | 0.989 | 0.970 |
| lab | r=0.3 | 2 | PS | 1.022 | +0.045 | 0.993 | 0.970 |
| lab | r=0.3 | 2 | cell | 1.065 | +0.133 | 0.990 | 0.969 |
| lab | r=0.3 | 2 | cellfill | 1.065 | +0.133 | 0.990 | 0.969 |
| lab | r=0.3 | 2 | cellinterp | 1.065 | +0.133 | 0.990 | 0.969 |
| lab | r=0.5 | 0 | PS | 1.043 | +0.087 | 0.991 | 0.970 |
| lab | r=0.5 | 0 | cell | 4.808 | +22.113 | 0.731 | 0.664 |
| lab | r=0.5 | 0 | cellfill | 1.195 | +0.428 | 0.981 | 0.966 |
| lab | r=0.5 | 0 | cellinterp | 1.195 | +0.427 | 0.981 | 0.966 |
| lab | r=0.5 | 1 | PS | 1.056 | +0.114 | 0.991 | 0.971 |
| lab | r=0.5 | 1 | cell | 4.373 | +18.126 | 0.763 | 0.696 |
| lab | r=0.5 | 1 | cellfill | 1.201 | +0.442 | 0.980 | 0.967 |
| lab | r=0.5 | 1 | cellinterp | 1.201 | +0.443 | 0.980 | 0.967 |
| lab | r=0.5 | 2 | PS | 1.012 | +0.025 | 0.993 | 0.972 |
| lab | r=0.5 | 2 | cell | 4.801 | +22.051 | 0.717 | 0.642 |
| lab | r=0.5 | 2 | cellfill | 1.113 | +0.239 | 0.988 | 0.964 |
| lab | r=0.5 | 2 | cellinterp | 1.114 | +0.240 | 0.988 | 0.964 |
| lab | r=0.7 | 0 | PS | 1.097 | +0.203 | 0.990 | 0.970 |
| lab | r=0.7 | 0 | cell | 12.142 | +146.432 | 0.462 | 0.386 |
| lab | r=0.7 | 0 | cellfill | 1.294 | +0.674 | 0.978 | 0.962 |
| lab | r=0.7 | 0 | cellinterp | 1.302 | +0.696 | 0.977 | 0.962 |
| lab | r=0.7 | 1 | PS | 1.124 | +0.263 | 0.989 | 0.970 |
| lab | r=0.7 | 1 | cell | 13.252 | +174.618 | 0.444 | 0.391 |
| lab | r=0.7 | 1 | cellfill | 1.312 | +0.720 | 0.973 | 0.965 |
| lab | r=0.7 | 1 | cellinterp | 1.318 | +0.737 | 0.973 | 0.965 |
| lab | r=0.7 | 2 | PS | 1.023 | +0.047 | 0.993 | 0.969 |
| lab | r=0.7 | 2 | cell | 12.157 | +146.786 | 0.474 | 0.410 |
| lab | r=0.7 | 2 | cellfill | 1.235 | +0.526 | 0.983 | 0.961 |
| lab | r=0.7 | 2 | cellinterp | 1.255 | +0.574 | 0.983 | 0.960 |
| lab | pi=0.3,L=4 | 0 | PS | 1.008 | +0.016 | 0.992 | 0.972 |
| lab | pi=0.3,L=4 | 0 | cell | 1.291 | +0.667 | 0.968 | 0.936 |
| lab | pi=0.3,L=4 | 0 | cellfill | 1.053 | +0.108 | 0.989 | 0.969 |
| lab | pi=0.3,L=4 | 0 | cellinterp | 1.053 | +0.108 | 0.989 | 0.969 |
| lab | pi=0.3,L=4 | 1 | PS | 1.015 | +0.030 | 0.992 | 0.971 |
| lab | pi=0.3,L=4 | 1 | cell | 1.299 | +0.688 | 0.960 | 0.941 |
| lab | pi=0.3,L=4 | 1 | cellfill | 1.043 | +0.088 | 0.990 | 0.967 |
| lab | pi=0.3,L=4 | 1 | cellinterp | 1.043 | +0.088 | 0.990 | 0.967 |
| lab | pi=0.3,L=4 | 2 | PS | 1.017 | +0.035 | 0.991 | 0.970 |
| lab | pi=0.3,L=4 | 2 | cell | 1.574 | +1.476 | 0.950 | 0.898 |
| lab | pi=0.3,L=4 | 2 | cellfill | 1.088 | +0.184 | 0.987 | 0.967 |
| lab | pi=0.3,L=4 | 2 | cellinterp | 1.088 | +0.184 | 0.987 | 0.967 |
| lab | pi=0.5,L=4 | 0 | PS | 1.035 | +0.072 | 0.992 | 0.971 |
| lab | pi=0.5,L=4 | 0 | cell | 2.783 | +6.744 | 0.863 | 0.811 |
| lab | pi=0.5,L=4 | 0 | cellfill | 1.085 | +0.177 | 0.989 | 0.969 |
| lab | pi=0.5,L=4 | 0 | cellinterp | 1.086 | +0.179 | 0.989 | 0.969 |
| lab | pi=0.5,L=4 | 1 | PS | 1.040 | +0.082 | 0.991 | 0.971 |
| lab | pi=0.5,L=4 | 1 | cell | 4.653 | +20.654 | 0.727 | 0.673 |
| lab | pi=0.5,L=4 | 1 | cellfill | 1.129 | +0.274 | 0.985 | 0.967 |
| lab | pi=0.5,L=4 | 1 | cellinterp | 1.129 | +0.275 | 0.985 | 0.967 |
| lab | pi=0.5,L=4 | 2 | PS | 1.017 | +0.034 | 0.991 | 0.968 |
| lab | pi=0.5,L=4 | 2 | cell | 4.839 | +22.419 | 0.763 | 0.715 |
| lab | pi=0.5,L=4 | 2 | cellfill | 1.082 | +0.171 | 0.990 | 0.965 |
| lab | pi=0.5,L=4 | 2 | cellinterp | 1.083 | +0.174 | 0.990 | 0.965 |
| lab | pi=0.5,L=8 | 0 | PS | 1.028 | +0.057 | 0.993 | 0.972 |
| lab | pi=0.5,L=8 | 0 | cell | 2.203 | +3.852 | 0.898 | 0.862 |
| lab | pi=0.5,L=8 | 0 | cellfill | 1.102 | +0.214 | 0.989 | 0.969 |
| lab | pi=0.5,L=8 | 0 | cellinterp | 1.103 | +0.217 | 0.989 | 0.969 |
| lab | pi=0.5,L=8 | 1 | PS | 1.073 | +0.151 | 0.991 | 0.971 |
| lab | pi=0.5,L=8 | 1 | cell | 5.329 | +27.401 | 0.724 | 0.703 |
| lab | pi=0.5,L=8 | 1 | cellfill | 1.157 | +0.338 | 0.983 | 0.962 |
| lab | pi=0.5,L=8 | 1 | cellinterp | 1.167 | +0.362 | 0.983 | 0.962 |
| lab | pi=0.5,L=8 | 2 | PS | 1.052 | +0.106 | 0.992 | 0.969 |
| lab | pi=0.5,L=8 | 2 | cell | 3.877 | +14.032 | 0.789 | 0.747 |
| lab | pi=0.5,L=8 | 2 | cellfill | 1.121 | +0.257 | 0.988 | 0.965 |
| lab | pi=0.5,L=8 | 2 | cellinterp | 1.125 | +0.265 | 0.989 | 0.965 |
| lab | pi=0.7,L=4 | 0 | PS | 1.079 | +0.164 | 0.991 | 0.971 |
| lab | pi=0.7,L=4 | 0 | cell | 11.957 | +141.981 | 0.450 | 0.392 |
| lab | pi=0.7,L=4 | 0 | cellfill | 1.195 | +0.428 | 0.983 | 0.965 |
| lab | pi=0.7,L=4 | 0 | cellinterp | 1.197 | +0.432 | 0.985 | 0.965 |
| lab | pi=0.7,L=4 | 1 | PS | 1.000 | +0.001 | 0.992 | 0.970 |
| lab | pi=0.7,L=4 | 1 | cell | 10.586 | +111.057 | 0.460 | 0.397 |
| lab | pi=0.7,L=4 | 1 | cellfill | 1.160 | +0.345 | 0.981 | 0.965 |
| lab | pi=0.7,L=4 | 1 | cellinterp | 1.167 | +0.363 | 0.980 | 0.965 |
| lab | pi=0.7,L=4 | 2 | PS | 1.079 | +0.163 | 0.992 | 0.973 |
| lab | pi=0.7,L=4 | 2 | cell | 15.022 | +224.658 | 0.411 | 0.360 |
| lab | pi=0.7,L=4 | 2 | cellfill | 1.192 | +0.421 | 0.985 | 0.965 |
| lab | pi=0.7,L=4 | 2 | cellinterp | 1.208 | +0.460 | 0.984 | 0.964 |
| lab | 기준 | -1 | PS | 1.000 | +0.000 | 0.992 | 0.971 |
| lab | 기준 | -1 | cell | 1.000 | +0.000 | 0.990 | 0.970 |
| lab | 기준 | -1 | cellfill | 1.000 | +0.000 | 0.990 | 0.970 |
| lab | 기준 | -1 | cellinterp | 1.000 | +0.000 | 0.990 | 0.970 |