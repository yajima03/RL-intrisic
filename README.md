# RL-intrisic

Stable-Baselines3 を用いて、Scalable Pyramid（SP）環境上で  
探索用の内発的動機付け手法を比較・分析するための実装です。

現在は、以下を実装・検証対象としています。

- 外部報酬のみの DQN ベースライン
- count-based 内発報酬
- RND（Random Network Distillation）

今後はこれを土台として、研究の本題である **PGLP** を実装します。

---

## 概要

本リポジトリは、以下を目的としています。

1. **Scalable Pyramid（SP）環境**を用いて、探索挙動を状態単位で分析可能にする  
2. **ベースアルゴリズム**と**内発的動機付け手法**を分離して実験できるようにする  
3. 将来的に **PGLP-local** および **PGLP-learned** を追加し、既存手法と比較できるようにする  

現在の実行は、以下の 3 つの設定ファイルを組み合わせる構成です。

- `env`: 環境設定
- `algo`: ベースアルゴリズム設定
- `intrinsic`: 内発的動機付け設定

これにより、たとえば以下のような組み合わせができます。

- `SP + DQN + none`
- `SP + DQN + count`
- `SP + DQN + RND`
- `SP + PPO + RND`

---

## ディレクトリ構成

```text
src/
  envs/         # 環境実装（Scalable Pyramid など）
  intrinsic/    # 内発的動機付け手法（count, RND, 今後 PGLP）
  training/     # 学習実行・wrapper・train.py
  analysis/     # 学習結果の可視化
  evaluation/   # 評価用コード
  utils/        # 補助関数

configs/
  env/          # 環境設定
  algo/         # ベースアルゴリズム設定
  intrinsic/    # 内発的動機付け設定

outputs/
  <run_name>/
    logs/
    eval/
    checkpoints/
    models/
    tensorboard/
```

## 現在の実装状況

ベースアルゴリズム
- DQN
- PPO

内発的動機付け
- none: 外部報酬のみ
- count: count-based 内発報酬
- rnd: Random Network Distillation

今後の実装予定
1. PGLP-local : 局所統計型 predictability による原理検証
2. PGLP-learned : 学習型 gate による predictability 推定
3. 一般環境への展開と比較実験


## セットアップ
`uv` を用いて仮想環境と依存関係を管理します。

```bash
uv sync
```


## 実行方法
以下のように、環境・ベースアルゴリズム・内発的動機付け手法をそれぞれ指定して実行。
``` bash
uv run python -m src.training.train \
  --env-config configs/env/sp_base.yaml \
  --algo-config configs/algo/dqn.yaml \
  --intrinsic-config configs/intrinsic/rnd.yaml
```

例:
- 外部報酬のみ
``` bash
uv run python -m src.training.train \
  --env-config configs/env/sp_base.yaml \
  --algo-config configs/algo/dqn.yaml \
  --intrinsic-config configs/intrinsic/none.yaml
```

- count-based
``` bash
uv run python -m src.training.train \
  --env-config configs/env/sp_base.yaml \
  --algo-config configs/algo/dqn.yaml \
  --intrinsic-config configs/intrinsic/count.yaml
```

- PPO baseline
``` bash
uv run python -m src.training.train \
  --env-config configs/env/sp_base.yaml \
  --algo-config configs/algo/ppo.yaml \
  --intrinsic-config configs/intrinsic/none.yaml
```

- PPO + RND
``` bash
uv run python -m src.training.train \
  --env-config configs/env/sp_base.yaml \
  --algo-config configs/algo/ppo.yaml \
  --intrinsic-config configs/intrinsic/rnd.yaml
```

- minigrid_doorkey
``` bash
uv run python -m src.training.train \
  --env-config configs/env/minigrid_doorkey.yaml \
  --algo-config configs/algo/dqn_mlp.yaml \
  --intrinsic-config configs/intrinsic/rnd_mlp.yaml
```


## ログと保存内容
各 run は `outputs/<run_name>/` 以下に保存されます。

主な出力は以下です。
- `logs/train_monitor.csv`
- `logs/eval_monitor.csv`
- `eval/evaluations.npz`
- `models/final_model.zip`
- `checkpoints/`
- `tensorboard/`

LPM実験では `logs/` に次の診断CSVも保存されます。

- `lpm_transition.csv`: 現在誤差、旧誤差予測、signed LP、正規化・係数適用後のRL報酬
- `lpm_update.csv`: dynamics/error modelのloss、較正誤差、version、勾配・parameter norm
- `lpm_probe.csv`: 固定SP edge上の現在・直前モデル誤差とoracle/estimated progress
- `run_config.yaml`: seed、解決済み設定、git commit、ログ値の定義

`predicted_previous_error` と `lpm_signed` はlog-MSE空間です。現実装がRL報酬に
使用した値は `lpm_raw_used`、最終的にDQNへ渡した値は
`intrinsic_reward_to_rl` として別々に記録されます。

### monitor CSV の列
現在の実装では、train_monitor.csv と eval_monitor.csv の両方に、少なくとも以下の列が保存される。

```csv
r,l,t,episode_external_return,episode_intrinsic_return,episode_total_return
```

それぞれの意味は以下の通り。
- `r`: その環境がエージェントに実際に返したエピソード報酬
  - train では通常 total return
  - eval では通常 external return
- `l` : episode length
- `t` : 経過時間
- `episode_external_return` : 外部報酬の合計
- `episode_intrinsic_return` : 内発報酬の合計
- `pisode_total_return` : 合計報酬（external + intrinsic）

### train と eval の違い
- train : 内発報酬を含む `total reward` を用いて学習する
- eval : 実際の task performance を見るため、外部報酬のみで評価する

そのため、eval では通常になる
- `episode_intrinsic_return = 0`
- `episode_total_return = episode_external_return`



## 結果プロット
```bash
uv run python -m analysis.plot_learning_curve \        
  --run-dir outputs/(任意のoutputsのディレクトリ) 
```

### 生成されるグラフ 
#### train関連
1. `train_total_return_vs_episode.png` : 各 training episode ごとの 合計報酬
    - エージェントが学習で最適化している報酬の推移を見る
2. `train_external_return_vs_episode.png` : 各 training episode で得た 外部報酬の合計
    - 実際にタスクの真の報酬を取れているかを確認する
3. `train_intrinsic_return_vs_episode.png` : 各 training episode で得た 内発報酬の合計
    - 探索ボーナスがどの程度発生しているかを確認
4. `train_returns_combined_vs_timestep.png` : 1~3を重ねてプロットしたもの
    - total / external / intrinsic の関係を一度に確認
5. `train_episode_length_vs_episode.png` : 各 training episode の長さ
    - SP では通常、episode 長は環境の深さに対応


#### eval関連
1. `eval_callback_mean_reward_vs_timestep.png` : 各評価タイミング(`eval_freq`)で `n_eval_episodes` 本の episode を実行し、その 平均 reward を timestep ごとにプロット
2. `eval_total_return_vs_episode.png` : 各 evaluation episode ごとの total return
    - 現在の eval は外部報酬のみで評価するため、通常は external return と同じ
3. `eval_external_return_vs_episode.png` : 各 evaluation episode ごとの external return
    - 
4. `eval_intrinsic_return_vs_episode.png` : 各 evaluation episode ごとの intrinsic return
    - 現在の eval では外部報酬のみで評価するため、通常は 0
5. `eval_returns_combined_vs_episode.png` : 2~4を重ねてプロットしたもの
6. `eval_episode_length_vs_episode.png` : 各 evaluation episode の長さです。
    - SP では通常、episode 長は環境の深さに対応.環境挙動の sanity checkとして使用可能


### どのグラフを重視すべきか

#### 最重要
- `eval_callback_mean_reward_vs_timestep.png`
- `train_external_return_vs_episode.png`

この2つは、**本当に task performance が向上しているか** を確認するうえで最も重要です。

- `eval_callback_mean_reward_vs_timestep.png`  
  評価タイミングごとの平均性能を表します。  
  学習が進むにつれて、外部報酬ベースでどの程度タスクを解けるようになったかを確認できます。

- `train_external_return_vs_episode.png`  
  training 中に、実際の外部報酬をどの程度取れているかを表します。  
  train の total return が増えていても、この値が増えていなければ、**内発報酬だけで学習が進んでいるように見えている可能性**があります。

#### 内発的動機付けの分析で重要
- `train_intrinsic_return_vs_episode.png`
- `train_returns_combined_vs_timestep.png`

これらは、**探索ボーナスがどのように振る舞っているか** を見るために重要です。

- `train_intrinsic_return_vs_episode.png`  
  内発報酬がどの程度出ているかを表します。  
  これにより、探索信号が十分に存在しているか、逆に早く枯渇していないかを確認できます。

- `train_returns_combined_vs_timestep.png`  
  total / external / intrinsic の3つを同時に見られるため、  
  **「何が上がっていて、何が上がっていないのか」** を一目で確認できます。

#### 補助的に見るグラフ
- `train_episode_length_vs_episode.png`
- `eval_episode_length_vs_episode.png`
- `eval_external_return_vs_episode.png`

これらは、性能そのものというより、**挙動の安定性やばらつき、環境の sanity check** に役立ちます。

- `eval_external_return_vs_episode.png` は、各 evaluation episode の生データなので、  
  平均値だけでは見えない「たまに成功しているのか」「毎回少しずつ取れているのか」を見るのに有用です。


## 研究の今後の流れ

本リポジトリは、今後以下の順で発展させていく予定です。

1. **SP + DQN ベースライン**
2. **count-based / RND の比較**
3. **PGLP-local の実装**
4. **PGLP-learned への拡張**
5. **一般環境への展開**

### 1. SP + DQN ベースライン
まずは外部報酬のみの DQN を基準線として確立します。  
ここで、SP 環境そのものが安定して学習・評価できることを確認します。

### 2. count-based / RND の比較
次に、既存の探索手法として

- count-based intrinsic reward
- RND

を同じ基盤上で比較します。

ここでは特に、

- total return は伸びるか
- external return につながるか
- intrinsic return が早く枯渇しないか

を確認します。

### 3. PGLP-local の実装
その次に、本題である **PGLP-local** を実装します。  
これは、局所統計にもとづいて predictability を評価し、

- 予測しやすさ
- 学習進捗

を組み合わせた内発報酬を与える方式です。

### 4. PGLP-learned への拡張
PGLP-local の結果をもとに、predictability を直接推定する学習器を導入し、  
より一般的な環境でも使える **PGLP-learned** へ拡張します。

### 5. 一般環境への展開
最終的には、SP 以外の一般的なベンチマーク環境にも適用し、

- NoisyTV 的挙動への反応
- 探索信号の枯渇
- 外部報酬への接続

などを比較・整理します。



## メモ

- train では、通常 **外部報酬 + 内発報酬** を用いて学習します
- eval では、通常 **外部報酬のみ** で性能を評価します

そのため、

- `train_total_return` が上がっている
- しかし `eval_external_return` が上がっていない

という場合は、**探索 bonus は効いているが、task を解く方策にはまだつながっていない** 可能性があります。

これは、count-based や RND の評価において特に重要です。  
また、今後 PGLP を実装したときにも、

- intrinsic return の振る舞い
- external return への接続
- eval での平均性能

を分けて見ることが重要になります。
