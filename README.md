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

## Where MLX (Metal) wins

Preliminary, M4 Max (full tables in the repo's `benchmarks/`):

- **Embedding** ~16×, **UMAP** ~12× faster → must use MLX.
- **HDBSCAN pairwise-distance + MST** 45–135× at 5K–20K points → large-scale.
- **HDBSCAN cluster extraction** stays on CPU (sequential tree traversal).
- **c-TF-IDF** crossover ≈ 100 topics × 50K vocab (scale-dependent).

## License

MIT (© 2026 Felix Lin). Upstream attributions (BERTopic MIT; hdbscan &
scikit-learn BSD-3-Clause) in the repo's `NOTICES.md`.
