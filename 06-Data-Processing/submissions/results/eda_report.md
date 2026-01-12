# EDA分析报告

## 1. 数据概览

- **数据形状**: (119808, 40)
- **内存使用**: 31.08 MB

## 3. 变量分布

### de_normal
- **偏度**: -0.0320
- **峰度**: -0.2800
- **变异系数**: 5.8453

### de_7_inner
- **偏度**: 0.1638
- **峰度**: 2.3951
- **变异系数**: 21.6401

### de_7_ball
- **偏度**: -0.0051
- **峰度**: -0.1719
- **变异系数**: 10.8469

### de_7_outer
- **偏度**: 0.0567
- **峰度**: 4.6536
- **变异系数**: 28.6921

### de_14_inner
- **偏度**: -0.0610
- **峰度**: 18.9541
- **变异系数**: 5.6231

### de_14_ball
- **偏度**: 0.2259
- **峰度**: 14.7174
- **变异系数**: 32.6909

### de_14_outer
- **偏度**: 0.0036
- **峰度**: -0.1717
- **变异系数**: 6.8470

### de_21_inner
- **偏度**: 0.3031
- **峰度**: 4.4769
- **变异系数**: 36.8637

### de_21_ball
- **偏度**: 0.0081
- **峰度**: 0.1196
- **变异系数**: 5.7725

### de_21_outer
- **偏度**: 0.1051
- **峰度**: 17.9358
- **变异系数**: 123.3208

### de_normal_zscore
- **偏度**: -0.0320
- **峰度**: -0.2800
- **变异系数**: 22663275544187012.0000

### de_normal_rank
- **偏度**: 0.0000
- **峰度**: -1.2000
- **变异系数**: 0.5773

### de_normal_bin
- **偏度**: 0.0011
- **峰度**: -1.3005
- **变异系数**: 0.7077

### de_7_inner_zscore
- **偏度**: 0.1638
- **峰度**: 2.3951
- **变异系数**: -206624656.0000

### de_7_inner_rank
- **偏度**: 0.0000
- **峰度**: -1.2000
- **变异系数**: 0.5773

### de_7_inner_bin
- **偏度**: 0.0003
- **峰度**: -1.3001
- **变异系数**: 0.7072

### de_7_ball_zscore
- **偏度**: -0.0051
- **峰度**: -0.1719
- **变异系数**: -31934615539536244.0000

### de_7_ball_rank
- **偏度**: 0.0000
- **峰度**: -1.2000
- **变异系数**: 0.5773

### de_7_ball_bin
- **偏度**: 0.0008
- **峰度**: -1.3005
- **变异系数**: 0.7077

### de_7_outer_zscore
- **偏度**: 0.0567
- **峰度**: 4.6536
- **变异系数**: 106104544.0000

### de_7_outer_rank
- **偏度**: 0.0000
- **峰度**: -1.2000
- **变异系数**: 0.5773

### de_7_outer_bin
- **偏度**: 0.0013
- **峰度**: -1.2998
- **变异系数**: 0.7074

### de_14_inner_zscore
- **偏度**: -0.0610
- **峰度**: 18.9541
- **变异系数**: -49073356.0000

### de_14_inner_rank
- **偏度**: 0.0000
- **峰度**: -1.2000
- **变异系数**: 0.5773

### de_14_inner_bin
- **偏度**: 0.0022
- **峰度**: -1.3011
- **变异系数**: 0.7080

### de_14_ball_zscore
- **偏度**: 0.2259
- **峰度**: 14.7174
- **变异系数**: -436207584.0000

### de_14_ball_rank
- **偏度**: -0.0000
- **峰度**: -1.2000
- **变异系数**: 0.5773

### de_14_ball_bin
- **偏度**: 0.0008
- **峰度**: -1.3003
- **变异系数**: 0.7075

### de_14_outer_zscore
- **偏度**: 0.0036
- **峰度**: -0.1717
- **变异系数**: -25091483638207048.0000

### de_14_outer_rank
- **偏度**: 0.0000
- **峰度**: -1.2000
- **变异系数**: 0.5773

### de_14_outer_bin
- **偏度**: 0.0003
- **峰度**: -1.2999
- **变异系数**: 0.7072

### de_21_inner_zscore
- **偏度**: 0.3031
- **峰度**: 4.4769
- **变异系数**: 135374768.0000

### de_21_inner_rank
- **偏度**: 0.0000
- **峰度**: -1.2000
- **变异系数**: 0.5773

### de_21_inner_bin
- **偏度**: 0.0007
- **峰度**: -1.2995
- **变异系数**: 0.7073

### de_21_ball_zscore
- **偏度**: 0.0081
- **峰度**: 0.1196
- **变异系数**: nan

### de_21_ball_rank
- **偏度**: -0.0000
- **峰度**: -1.2001
- **变异系数**: 0.5773

### de_21_ball_bin
- **偏度**: 0.0013
- **峰度**: -1.2991
- **变异系数**: 0.7073

### de_21_outer_zscore
- **偏度**: 0.1051
- **峰度**: 17.9358
- **变异系数**: 5234490880.0000

### de_21_outer_rank
- **偏度**: -0.0000
- **峰度**: -1.2000
- **变异系数**: 0.5773

### de_21_outer_bin
- **偏度**: 0.0017
- **峰度**: -1.2991
- **变异系数**: 0.7075


## 4. 相关性分析

### 强相关性对 (|r| > 0.7)

| 变量对 | 相关系数 |
|--------|----------|
| de_normal - de_normal_zscore | 1.0000 |
| de_normal - de_normal_rank | 0.9830 |
| de_normal - de_normal_bin | 0.9498 |
| de_7_inner - de_7_inner_zscore | 1.0000 |
| de_7_inner - de_7_inner_rank | 0.9228 |
| de_7_inner - de_7_inner_bin | 0.8761 |
| de_7_ball - de_7_ball_zscore | 1.0000 |
| de_7_ball - de_7_ball_rank | 0.9801 |
| de_7_ball - de_7_ball_bin | 0.9458 |
| de_7_outer - de_7_outer_zscore | 1.0000 |
| de_7_outer - de_7_outer_rank | 0.8358 |
| de_7_outer - de_7_outer_bin | 0.7758 |
| de_14_inner - de_14_inner_zscore | 1.0000 |
| de_14_inner - de_14_inner_rank | 0.7689 |
| de_14_inner - de_14_inner_bin | 0.7151 |
| de_14_ball - de_14_ball_zscore | 1.0000 |
| de_14_ball - de_14_ball_rank | 0.8521 |
| de_14_ball - de_14_ball_bin | 0.8029 |
| de_14_outer - de_14_outer_zscore | 1.0000 |
| de_14_outer - de_14_outer_rank | 0.9800 |
| de_14_outer - de_14_outer_bin | 0.9457 |
| de_21_inner - de_21_inner_zscore | 1.0000 |
| de_21_inner - de_21_inner_rank | 0.8962 |
| de_21_inner - de_21_inner_bin | 0.8478 |
| de_21_ball - de_21_ball_zscore | 1.0000 |
| de_21_ball - de_21_ball_rank | 0.9710 |
| de_21_ball - de_21_ball_bin | 0.9334 |
| de_21_outer - de_21_outer_zscore | 1.0000 |
| de_21_outer - de_21_outer_rank | 0.7170 |
| de_normal_zscore - de_normal_rank | 0.9830 |
| de_normal_zscore - de_normal_bin | 0.9498 |
| de_normal_rank - de_normal_bin | 0.9798 |
| de_7_inner_zscore - de_7_inner_rank | 0.9228 |
| de_7_inner_zscore - de_7_inner_bin | 0.8761 |
| de_7_inner_rank - de_7_inner_bin | 0.9798 |
| de_7_ball_zscore - de_7_ball_rank | 0.9801 |
| de_7_ball_zscore - de_7_ball_bin | 0.9458 |
| de_7_ball_rank - de_7_ball_bin | 0.9798 |
| de_7_outer_zscore - de_7_outer_rank | 0.8358 |
| de_7_outer_zscore - de_7_outer_bin | 0.7758 |
| de_7_outer_rank - de_7_outer_bin | 0.9798 |
| de_14_inner_zscore - de_14_inner_rank | 0.7689 |
| de_14_inner_zscore - de_14_inner_bin | 0.7151 |
| de_14_inner_rank - de_14_inner_bin | 0.9798 |
| de_14_ball_zscore - de_14_ball_rank | 0.8521 |
| de_14_ball_zscore - de_14_ball_bin | 0.8029 |
| de_14_ball_rank - de_14_ball_bin | 0.9798 |
| de_14_outer_zscore - de_14_outer_rank | 0.9800 |
| de_14_outer_zscore - de_14_outer_bin | 0.9457 |
| de_14_outer_rank - de_14_outer_bin | 0.9798 |
| de_21_inner_zscore - de_21_inner_rank | 0.8962 |
| de_21_inner_zscore - de_21_inner_bin | 0.8478 |
| de_21_inner_rank - de_21_inner_bin | 0.9798 |
| de_21_ball_zscore - de_21_ball_rank | 0.9710 |
| de_21_ball_zscore - de_21_ball_bin | 0.9334 |
| de_21_ball_rank - de_21_ball_bin | 0.9798 |
| de_21_outer_zscore - de_21_outer_rank | 0.7170 |
| de_21_outer_rank - de_21_outer_bin | 0.9798 |

## 6. 生成的可视化文件

- `results/figures/correlation_matrix.png`
- `results/figures/de_14_ball_bin_distribution.png`
- `results/figures/de_14_ball_bin_interactive.html`
- `results/figures/de_14_ball_distribution.png`
- `results/figures/de_14_ball_interactive.html`
- `results/figures/de_14_ball_rank_distribution.png`
- `results/figures/de_14_ball_rank_interactive.html`
- `results/figures/de_14_ball_zscore_distribution.png`
- `results/figures/de_14_ball_zscore_interactive.html`
- `results/figures/de_14_inner_bin_distribution.png`
- `results/figures/de_14_inner_bin_interactive.html`
- `results/figures/de_14_inner_distribution.png`
- `results/figures/de_14_inner_interactive.html`
- `results/figures/de_14_inner_rank_distribution.png`
- `results/figures/de_14_inner_rank_interactive.html`
- `results/figures/de_14_inner_zscore_distribution.png`
- `results/figures/de_14_inner_zscore_interactive.html`
- `results/figures/de_14_outer_bin_distribution.png`
- `results/figures/de_14_outer_bin_interactive.html`
- `results/figures/de_14_outer_distribution.png`
- `results/figures/de_14_outer_interactive.html`
- `results/figures/de_14_outer_rank_distribution.png`
- `results/figures/de_14_outer_rank_interactive.html`
- `results/figures/de_14_outer_zscore_distribution.png`
- `results/figures/de_14_outer_zscore_interactive.html`
- `results/figures/de_21_ball_bin_distribution.png`
- `results/figures/de_21_ball_bin_interactive.html`
- `results/figures/de_21_ball_distribution.png`
- `results/figures/de_21_ball_interactive.html`
- `results/figures/de_21_ball_rank_distribution.png`
- `results/figures/de_21_ball_rank_interactive.html`
- `results/figures/de_21_ball_zscore_distribution.png`
- `results/figures/de_21_ball_zscore_interactive.html`
- `results/figures/de_21_inner_bin_distribution.png`
- `results/figures/de_21_inner_bin_interactive.html`
- `results/figures/de_21_inner_distribution.png`
- `results/figures/de_21_inner_interactive.html`
- `results/figures/de_21_inner_rank_distribution.png`
- `results/figures/de_21_inner_rank_interactive.html`
- `results/figures/de_21_inner_zscore_distribution.png`
- `results/figures/de_21_inner_zscore_interactive.html`
- `results/figures/de_21_outer_bin_distribution.png`
- `results/figures/de_21_outer_bin_interactive.html`
- `results/figures/de_21_outer_distribution.png`
- `results/figures/de_21_outer_interactive.html`
- `results/figures/de_21_outer_rank_distribution.png`
- `results/figures/de_21_outer_rank_interactive.html`
- `results/figures/de_21_outer_zscore_distribution.png`
- `results/figures/de_21_outer_zscore_interactive.html`
- `results/figures/de_7_ball_bin_distribution.png`
- `results/figures/de_7_ball_bin_interactive.html`
- `results/figures/de_7_ball_distribution.png`
- `results/figures/de_7_ball_interactive.html`
- `results/figures/de_7_ball_rank_distribution.png`
- `results/figures/de_7_ball_rank_interactive.html`
- `results/figures/de_7_ball_zscore_distribution.png`
- `results/figures/de_7_ball_zscore_interactive.html`
- `results/figures/de_7_inner_bin_distribution.png`
- `results/figures/de_7_inner_bin_interactive.html`
- `results/figures/de_7_inner_distribution.png`
- `results/figures/de_7_inner_interactive.html`
- `results/figures/de_7_inner_rank_distribution.png`
- `results/figures/de_7_inner_rank_interactive.html`
- `results/figures/de_7_inner_zscore_distribution.png`
- `results/figures/de_7_inner_zscore_interactive.html`
- `results/figures/de_7_outer_bin_distribution.png`
- `results/figures/de_7_outer_bin_interactive.html`
- `results/figures/de_7_outer_distribution.png`
- `results/figures/de_7_outer_interactive.html`
- `results/figures/de_7_outer_rank_distribution.png`
- `results/figures/de_7_outer_rank_interactive.html`
- `results/figures/de_7_outer_zscore_distribution.png`
- `results/figures/de_7_outer_zscore_interactive.html`
- `results/figures/de_normal_bin_distribution.png`
- `results/figures/de_normal_bin_interactive.html`
- `results/figures/de_normal_distribution.png`
- `results/figures/de_normal_interactive.html`
- `results/figures/de_normal_rank_distribution.png`
- `results/figures/de_normal_rank_interactive.html`
- `results/figures/de_normal_zscore_distribution.png`
- `results/figures/de_normal_zscore_interactive.html`
- `results/figures/pca_analysis.png`
