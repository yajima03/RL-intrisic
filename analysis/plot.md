# 1. `mean_gate_vs_timestep.png`

## 何を見る図か

PGLP の **gate の強さ** の推移です。

## 意味

`gate` は

* 大きいほど「この遷移は予測可能そう」
* 小さいほど「この遷移は不安定・予測困難そう」

です。

## どう読むか

* ずっと 0 に近い
  → gate が強すぎて、ほとんど全部潰している
* ずっと 1 に近い
  → gate がほぼ効いていない
* 適度に中間
  → gate が選別として働いている可能性がある



# 2. `mean_progress_vs_timestep.png`

## 何を見る図か

PGLP の **progress 項** の推移です。

## 意味

いまの実装では、progress は
「その局所領域や prototype で、最近どれだけ改善しているか」
を表します。

## どう読むか

* ずっと 0 に近い
  → learning progress が出ていない
* 最初だけ高くてすぐ 0
  → すぐ学習済み扱いになっている
* 持続的に少し残る
  → 改善信号として機能している可能性が高い



# 3. `mean_raw_intrinsic_vs_timestep.png`

## 何を見る図か

**raw intrinsic reward** の推移です。

## 意味

これは

```text
raw intrinsic = gate × progress
```

の平均です。

## どう読むか

* これがほぼ 0
  → PGLP 全体として信号が出ていない
* gate は高いのに raw が低い
  → progress 側が死んでいる
* progress は高いのに raw が低い
  → gate 側が抑えすぎている

つまり、**PGLP 全体の出力の強さ**を見る図です。



# 4. `mean_current_error_vs_timestep.png`

## 何を見る図か

predictor の **現在の予測誤差** の推移です。

## 意味

`current_error` は
`(o_t, a_t) -> o_{t+1}` の latent 予測がどれだけ外れているか
を表します。

## どう読むか

* 下がっていく
  → predictor は学習している
* ずっと高い
  → predictor がうまく学習できていない
* 急激に下がってすぐ小さくなる
  → progress が早く死ぬ可能性がある



# 5. `mean_neighbor_count_vs_timestep.png`

## 何を見る図か

gate 計算で使った **近傍数** の推移です。

## 意味

same-action + latent k-NN で、実際に何個の neighbor を使えたかです。

## どう読むか

* 小さい
  → そもそも近傍が足りていない
* `min_candidates` よりかなり上で安定
  → 近傍検索は成立している
* 大きく揺れる
  → cache や same-action 候補が不安定



# 6. `mean_prototype_count_for_action_vs_timestep.png`

## 何を見る図か

各 action に対して、どれくらい prototype が育っているかの平均です。

## 意味

online prototype 方式では、action ごとに prototype を持ちます。
この図は、**その行動に対する prototype 群がどれくらい揃ってきたか**を見るものです。

## どう読むか

* 最初は小さく、徐々に `num_prototypes_per_action` へ近づく
  → 正常
* ずっと小さい
  → prototype が十分に埋まっていない
* 早く上限に達する
  → prototype 配置自体は進んでいる



# 7. `mean_prototype_long_vs_timestep.png`

## 何を見る図か

prototype ごとの **long EMA** の平均推移です。

## 意味

これは「昔寄りの難しさ」の平均です。

## どう読むか

* 高いまま
  → 長期基準が高い
* ゆっくり下がる
  → 長期基準として自然
* すぐ short と同じになる
  → long 側が遅くなっていない



# 8. `mean_prototype_short_vs_timestep.png`

## 何を見る図か

prototype ごとの **short EMA** の平均推移です。

## 意味

これは「最近の難しさ」の平均です。

## どう読むか

* long より速く下がる
  → 期待通り
* long とほぼ同じ
  → timescale separation が足りない
* かなり不安定
  → short 側が敏感すぎる可能性がある



# 9. `mean_prototype_index_vs_timestep.png`

## 何を見る図か

どの prototype が平均的に選ばれているかの推移です。

## 意味

これは **どの prototype がよく参照されているかの粗い指標** です。

## どう読むか

* 特定 index に偏る
  → 一部の prototype に集中している
* 途中で大きく変わる
  → 参照される局所領域が変わっている
* ただしこれは平均 index なので、解釈は少し粗いです

つまり、これは補助的な図です。



# 10. `num_samples_vs_timestep.png`

## 何を見る図か

各 log interval の間に、**何サンプル分の debug が集計されたか** です。

## 意味

`flush_debug_stats()` までに何回 `compute()` が呼ばれたかの目安です。

## どう読むか

* 安定している
  → ログ集計自体は正常
* 極端に少ない
  → サンプル数不足で平均値が不安定かもしれない
* 0 がある
  → その区間では debug がほぼ取れていない



# 11. `prototype_long_short_vs_timestep.png`

## 何を見る図か

`mean_prototype_long` と `mean_prototype_short` を **同じ図に重ねたもの**です。

## 意味

これは **progress が出る条件** を直接見る図です。

progress は概念的に

```text
long - short
```

なので、この2本の差が重要です。

## どう読むか

* long > short が保たれている
  → progress が出やすい
* すぐ重なる
  → progress が死にやすい
* 差が大きすぎる
  → 常に高い intrinsic が出る可能性がある

これはかなり重要な図です。



# 12. `pglp_core_metrics_vs_timestep.png`

## 何を見る図か

PGLP の主要指標をまとめて見られる図です。

通常は

* `mean_gate`
* `mean_progress`
* `mean_raw_intrinsic`
* `mean_current_error`

を同時に描きます。

## 意味

これは **何がボトルネックなのかを一度に見る** ための図です。

## どう読むか

たとえば、

* `current_error` は下がっている
* `progress` はすぐ 0
* `gate` はそこそこ高い
* `raw_intrinsic` は 0

なら
→ **predictor は学習しているが、progress が死んでいる**

逆に、

* `progress` はある
* `gate` がほぼ 0
* `raw_intrinsic` も 0

なら
→ **gate が抑えすぎている**

という切り分けができます。

これは最重要のまとめ図です。



# どの図を特に重視すべきか

私なら、まずこの順で見ます。

## 最重要

* `pglp_core_metrics_vs_timestep.png`
* `prototype_long_short_vs_timestep.png`

## 次に重要

* `mean_gate_vs_timestep.png`
* `mean_progress_vs_timestep.png`
* `mean_raw_intrinsic_vs_timestep.png`

## 補助

* `mean_neighbor_count_vs_timestep.png`
* `mean_prototype_count_for_action_vs_timestep.png`
* `mean_prototype_index_vs_timestep.png`
* `num_samples_vs_timestep.png`

