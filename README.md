# mlx-bertopic

**BERTopic on Apple Silicon — MLX (Metal GPU) backends.**

Independent MLX re-implementations of BERTopic's components, accelerated on
Apple Silicon via the Metal GPU. Think of it as the Apple-Silicon counterpart to
[RAPIDS cuML](https://rapids.ai) GPU backends for BERTopic.

👉 **[github.com/mlx-bertopic/mlx-bertopic](https://github.com/mlx-bertopic/mlx-bertopic)** — the monorepo (install with `pip install "git+https://github.com/mlx-bertopic/mlx-bertopic.git"`).

## Modules (all in the monorepo)

| Module | Role | Aligned to |
|---|---|---|
| `mlx_bertopic` | Umbrella: BERTopic backends | BERTopic 0.17.4 |
| `mlx_ctfidf` | Class-based TF-IDF (c-TF-IDF) | BERTopic `ClassTfidfTransformer` |
| `mlx_hdbscan` | HDBSCAN (Metal distance + MST) | hdbscan 0.8.x |
| `mlx_keybert` | KeyBERT-inspired + MMR | KeyBERT |
| `mlx_lda` | LDA (variational EM) | scikit-learn LDA / Blei 2003 |
| `mlx_dtm` | LSTM dynamic topic modeling | BERTopic topics-over-time |
| `mlx_outlier` | BERTopic outlier reduction | BERTopic outlier strategies |

Each is an **independent MLX reimplementation, not a port**; behavior is aligned
to the upstream reference and verified by golden tests.

## Performance (M4 Max, 128 GB)

Full BERTopic pipeline on 164K documents × 4096 dimensions:

| Stage | MLX (Metal) | CPU | Speedup |
|---|---|---|---|
| UMAP 5D | 20s | 480s | **24×** |
| HDBSCAN | 4s | 5s | ~1× (sparse mode) |
| c-TF-IDF | 0.6s | 0.8s | ~1× |
| **End-to-end** | **25s** | **486s** | **~20×** |

Smaller datasets are even faster: 32K docs in 4s, 11K in 0.5s.

## Current version: 0.5.0

- Includes workaround for [mlx-vis UMAP divergence](https://github.com/mlx-bertopic/mlx-bertopic/blob/main/patches/mlx_vis_umap_stability.md)
  on large (>100K) high-dimensional datasets.
- 63 tests passing, aligned to BERTopic 0.17.4 / hdbscan 0.8.x.

## License

MIT (© 2026 Felix Lin). Upstream attributions (BERTopic MIT; hdbscan &
scikit-learn BSD-3-Clause) in the repo's `NOTICES.md`.
