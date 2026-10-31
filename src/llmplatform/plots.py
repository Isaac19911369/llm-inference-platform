from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sweep_figure(rows, path: Path, slo_ttft_ms: float, slo_itl_ms: float) -> Path:
    tput = [r.output_tokens_per_s for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=130)
    for ax, key, slo, label in [
        (axes[0], "ttft_p95_ms", slo_ttft_ms, "TTFT p95 [ms]"),
        (axes[1], "itl_p95_ms", slo_itl_ms, "inter-token latency p95 [ms]"),
    ]:
        ys = [getattr(r, key) for r in rows]
        ax.plot(tput, ys, "o-", color="#2f7ed8")
        for r, x, y in zip(rows, tput, ys, strict=True):
            ax.annotate(
                f"c={r.concurrency}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7
            )
        ax.axhline(slo, color="#e4572e", ls="--", lw=1, label="SLO")
        ax.set_xlabel("output tokens/s (one replica)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Latency vs. throughput as concurrency grows (simulated engine)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
