# CPU 移植メモ

本家 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) は**単一 NVIDIA GPU（H100 想定）専用**です。
このリポジトリはそれを **CPU で動く小規模学習環境**へ移植したフォークで、`cpu-local` ブランチに変更が入っています。
`master` ブランチは upstream そのままです。移植分はまだ未コミットなので、現時点では `git diff` で全差分を確認できます
（`cpu-local` にコミットした後は `git diff master cpu-local`）。

## なぜ移植が必要だったか

このマシンには NVIDIA GPU がありません（GPU は AMD Radeon 760M の内蔵のみ、`nvidia-smi` も不在）。
本家の `train.py` は import 直後に `torch.cuda.get_device_capability()` を呼ぶため、そのままでは 1 行も動きませんでした。

## 変更点

### 1. attention: FlashAttention-3 → SDPA

`kernels` 経由の FlashAttention-3 は CUDA 専用なので、PyTorch 標準の
`F.scaled_dot_product_attention` に置き換えました（`train.py` の `attention()`）。

- FA3 と同じ `(B, T, H, D)` レイアウトの入出力を保つラッパーにしてあるので、呼び出し側は 1 行の差し替えで済んでいます
- sliding window（`WINDOW_PATTERN` の `S`）は明示 bool マスクで表現し、`(T, window, device)` をキーにキャッシュします
- window が系列長以上なら `is_causal=True` の高速パスに落ちます
- 依存から `kernels` を削除しました

### 2. デバイス非依存化

`prepare.py` に `get_device()` / `DEVICE`（CUDA > MPS > CPU）を追加し、以下を差し替えました。

| 箇所 | 本家 | 移植後 |
| --- | --- | --- |
| `train.py` の device | `torch.device("cuda")` 固定 | `DEVICE` |
| autocast | `device_type="cuda"` | `device_type=DEVICE.type`（bf16 は維持） |
| `torch.cuda.synchronize()` | 直接呼び出し | `sync()`（CPU では no-op） |
| `torch.cuda.manual_seed(42)` | 無条件 | CUDA のときだけ |
| `peak_vram_mb` | `torch.cuda.max_memory_allocated()` | CPU では `0.0` |
| dataloader のバッファ | `pin_memory=True` + `device="cuda"` + `non_blocking=True` | CUDA のときだけ pin/非同期 |
| `evaluate_bpb` の token_bytes | `device="cuda"` | `device=DEVICE` |

### 3. torch.compile のゲート化

Windows の CPU では inductor が C++ コンパイラ（`cl.exe`）を要求して失敗します。
`USE_COMPILE = DEVICE.type == "cuda"` を導入し、モデル本体と `adamw_step_fused` /
`muon_step_fused` の compile を CUDA 限定にしました。デコレータではなく
`maybe_compile()` 経由に変えてあるので、CUDA 環境へ持っていけば本家と同じ挙動に戻ります。

### 4. データセット: climbmix → TinyStories

本家 README が小規模計算機向けに推奨している
[karpathy/tinystories-gpt4-clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)
に差し替えました。エントロピーが低いので、小さいモデルでも読める文章が出ます。

本家は 6543 個の shard ファイルに分かれており「末尾 shard を val に固定」する設計でしたが、
TinyStories は **単一 parquet（約 640MB / 2,732,634 行 / 2669 row group）** です。
そこで分割単位を row group に変え、**末尾 1 row group（約 1024 話）を val に固定**しています
（`row_group_split()`）。並列 shard ダウンロードは不要になったので削除しました。

### 5. スケールダウン

本家の値はソース各行のコメントに残してあります。

| 定数 | 本家 | このフォーク | 場所 |
| --- | --- | --- | --- |
| `MAX_SEQ_LEN` | 2048 | 256 | prepare.py |
| `EVAL_TOKENS` | 40×524288 | 2×65536 | prepare.py |
| `VOCAB_SIZE` | 8192 | 2048 | prepare.py |
| `DEPTH` | 8 | 4 | train.py |
| `HEAD_DIM` | 128 | 64 | train.py |
| `WINDOW_PATTERN` | `"SSSL"` | `"L"` | train.py |
| `TOTAL_BATCH_SIZE` | 2\*\*19 | 2\*\*14 | train.py |
| `DEVICE_BATCH_SIZE` | 128 | 16 | train.py |

`TIME_BUDGET`（300 秒）は本家のまま変えていません。

### 6. 依存

`torch` の取得先を CUDA wheel（`.../whl/cu128`、数 GB）から CPU wheel（`.../whl/cpu`）へ変更しました。
NVIDIA GPU 環境へ戻すときは `pyproject.toml` の index の url を `cu128` に戻し、`kernels` を依存に戻してください。

## 使い方

```bash
uv sync
```

```bash
uv run prepare.py
```

```bash
uv run train.py
```

自律研究モードに入るときは、このリポジトリでエージェントを起動して `program.md` を読ませてください。

## 注意点

- **val_bpb の絶対値は本家や他プラットフォームと比較できません。** モデル規模・語彙・系列長・データセットが全部違います。比較対象はこのマシンで取った自分の過去 run だけです
- `mfu_percent` は H100 のピーク FLOPS を基準にしているため、CPU 実行では 0.0% 付近に張り付きます。指標として意味を持ちません
- `peak_vram_mb` は CPU 実行では常に 0.0 です
- 5 分あたりの学習量は H100 比で桁違いに小さいので、得られるモデルの質ではなく「実験ループとコードを学ぶ場」として使うのが妥当です
