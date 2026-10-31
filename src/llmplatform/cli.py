"""llmplat: plan GPU capacity, simulate an LLM server, load-test an endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .capacity import BYTES, GPUS, MODELS, plan, replicas_needed
from .loadtest import max_concurrency_within_slo, run


def _plan(args) -> None:
    gpus = args.gpus or list(GPUS)
    print(
        f"| Model | GPU | TP | Weights | Weights/GPU (GiB) | KV/token (KiB) | "
        f"Max seqs @ {args.context} ctx | 1-stream tok/s | Replicas for load |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for model in args.models:
        for gpu in gpus:
            for tp in args.tp:
                p = plan(model, gpu, tp, args.dtype, args.kv_dtype, args.context)
                if not p.fits:
                    reps = "does not fit"
                elif p.single_stream_tokens_s < args.target_tps:
                    reps = f"too slow (< {args.target_tps:g} tok/s)"
                else:
                    reps = str(replicas_needed(p, args.rps, args.output_tokens, args.target_tps))
                print(
                    f"| {model} | {gpu} | {tp} | {args.dtype} | {p.weights_gib_per_gpu} | "
                    f"{p.kv_kib_per_token} | {p.max_concurrent_sequences} | "
                    f"{p.single_stream_tokens_s:.0f} | {reps} |"
                )


def _simulate(args) -> None:
    import uvicorn

    from .simulator import PerfModel, create_app

    perf = PerfModel(decode_ms=args.decode_ms, max_batch=args.max_batch)
    uvicorn.run(create_app(perf), host=args.host, port=args.port, log_level="warning")


def _sweep(args) -> None:
    rows = []
    print(
        "| Concurrency | Req/s | Output tok/s | TTFT p50 / p95 (ms) | ITL p50 / p95 (ms) | "
        "E2E p95 (s) | Errors |"
    )
    print("|---|---|---|---|---|---|---|")
    for c in args.concurrency:
        s = asyncio.run(
            run(
                args.url,
                c,
                max(args.requests_per_level, c * 3),
                args.model,
                args.prompt_tokens,
                args.max_tokens,
                args.api_key,
            )
        )
        rows.append(s)
        print(
            f"| {c} | {s.request_throughput} | {s.output_tokens_per_s} | "
            f"{s.ttft_p50_ms} / {s.ttft_p95_ms} | {s.itl_p50_ms} / {s.itl_p95_ms} | "
            f"{s.e2e_p95_s} | {s.errors} |",
            flush=True,
        )
    best = max_concurrency_within_slo(rows, args.slo_ttft_ms, args.slo_itl_ms)
    print()
    if best:
        print(
            f"Best SLO-compliant point (TTFT p95 <= {args.slo_ttft_ms} ms, ITL p95 <= "
            f"{args.slo_itl_ms} ms): concurrency {best.concurrency}, "
            f"{best.output_tokens_per_s} output tok/s per replica"
        )
    else:
        print("No concurrency level met the SLO.")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep.json").write_text(json.dumps([r.as_dict() for r in rows], indent=2))
        if args.plot:
            from .plots import sweep_figure

            sweep_figure(rows, out / "sweep.png", args.slo_ttft_ms, args.slo_itl_ms)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="llmplat", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="GPU memory / concurrency / replica estimate")
    pl.add_argument("--models", nargs="+", default=["llama-3.1-8b"], choices=list(MODELS))
    pl.add_argument("--gpus", nargs="+", choices=list(GPUS))
    pl.add_argument("--tp", nargs="+", type=int, default=[1])
    pl.add_argument("--dtype", default="bf16", choices=list(BYTES))
    pl.add_argument("--kv-dtype", default="fp16", choices=list(BYTES))
    pl.add_argument("--context", type=int, default=8192)
    pl.add_argument("--rps", type=float, default=2.0, help="peak requests per second")
    pl.add_argument("--output-tokens", type=int, default=300)
    pl.add_argument("--target-tps", type=float, default=20.0, help="tokens/s per user")

    sm = sub.add_parser("simulate", help="run the OpenAI-compatible LLM simulator")
    sm.add_argument("--host", default="127.0.0.1")
    sm.add_argument("--port", type=int, default=8000)
    sm.add_argument("--decode-ms", type=float, default=12.0)
    sm.add_argument("--max-batch", type=int, default=64)

    sw = sub.add_parser("sweep", help="load-test an endpoint over several concurrencies")
    sw.add_argument("--url", default="http://127.0.0.1:8000/v1")
    sw.add_argument("--model", default="simulated-llm")
    sw.add_argument("--api-key", default=None)
    sw.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 16, 32, 64, 96])
    sw.add_argument("--requests-per-level", type=int, default=64)
    sw.add_argument("--prompt-tokens", type=int, default=512)
    sw.add_argument("--max-tokens", type=int, default=128)
    sw.add_argument("--slo-ttft-ms", type=float, default=1000)
    sw.add_argument("--slo-itl-ms", type=float, default=50)
    sw.add_argument("--out", default="artifacts/sweep")
    sw.add_argument("--plot", action="store_true")

    args = p.parse_args(argv)
    {"plan": _plan, "simulate": _simulate, "sweep": _sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
