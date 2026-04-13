# pglp-rl

Stable-Baselines3 を用いた Scalable Pyramid (SP) 環境と
PGLP 系内発的動機付けの実装。

## 初期構成
- envs: 環境
- intrinsic: 内発報酬
- training: 学習実行
- evaluation: 評価
- utils: 補助関数

## 研究の段階
1. SP + DQN ベースライン
2. PGLP-local（局所統計型 predictability）
3. PGLP-learned（学習型 gate）へ拡張