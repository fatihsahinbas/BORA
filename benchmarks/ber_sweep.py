"""Measure the error-rate curve with and without forward error correction.

    python benchmarks/ber_sweep.py --out docs/ber.png

Produces the headline plot: two thresholds, five decibels apart.  The gap is
what the Reed-Solomon parity buys, paid for in airtime.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from bora import Config, channel, link


def frame_error_rate(
    payload: bytes,
    cfg: Config,
    snr_db: float,
    *,
    use_fec: bool,
    trials: int,
    seed: int,
) -> tuple[float, float]:
    """Return ``(frame_error_rate, bit_error_rate)`` at one noise level."""
    wave = link.transmit(payload, cfg, use_fec=use_fec)
    failures = 0
    bit_errors = []

    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        result = link.receive(channel.awgn(wave, snr_db, rng), cfg)
        failures += not result.ok
        bit_errors.append(_ber(payload, result.payload))

    return failures / trials, float(np.mean(bit_errors))


def _ber(sent: bytes, got: bytes) -> float:
    if not sent:
        return 0.0
    n = min(len(sent), len(got))
    if n == 0:
        return 1.0
    xor = np.frombuffer(sent[:n], np.uint8) ^ np.frombuffer(got[:n], np.uint8)
    missing = len(sent) - n
    return float(np.unpackbits(xor).sum() + 8 * missing) / (8 * len(sent))


def plot(points: list[float], curves: dict[str, list[float]], out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)

    styles = {
        "no FEC": dict(color="#c1553b", marker="o", linestyle="--"),
        "Reed-Solomon": dict(color="#2f7d8f", marker="s", linestyle="-"),
    }
    for label, values in curves.items():
        ax.plot(points, values, label=label, linewidth=2, **styles.get(label, {}))

    ax.set_xlabel("channel SNR (dB)")
    ax.set_ylabel("frame error rate")
    ax.set_title("BORA: frame error rate against noise")
    ax.set_ylim(-0.05, 1.05)
    ax.invert_xaxis()
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4_000)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="docs/ber.png")
    parser.add_argument("--csv", default="docs/ber.csv")
    parser.add_argument(
        "--snr",
        type=float,
        nargs="+",
        default=[16, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3],
    )
    args = parser.parse_args()

    cfg = Config()
    payload = bytes(np.random.default_rng(args.seed).integers(
        0, 256, args.size, dtype=np.uint8
    ))

    print(f"payload {args.size} bytes, {args.trials} trials per point")
    print(f"{'SNR':>6} {'no FEC FER':>12} {'FEC FER':>10} {'FEC BER':>10}")

    curves: dict[str, list[float]] = {"no FEC": [], "Reed-Solomon": []}
    rows = [("snr_db", "fer_nofec", "fer_fec", "ber_fec")]
    started = time.time()

    for snr in args.snr:
        bare, _ = frame_error_rate(
            payload, cfg, snr, use_fec=False, trials=args.trials, seed=args.seed
        )
        coded, ber = frame_error_rate(
            payload, cfg, snr, use_fec=True, trials=args.trials, seed=args.seed
        )
        curves["no FEC"].append(bare)
        curves["Reed-Solomon"].append(coded)
        rows.append((snr, f"{bare:.4f}", f"{coded:.4f}", f"{ber:.3e}"))
        print(f"{snr:>6.0f} {bare:>12.2f} {coded:>10.2f} {ber:>10.2e}")

    print(f"\n{time.time() - started:.1f} s")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.csv).write_text(
            "\n".join(",".join(map(str, r)) for r in rows) + "\n"
        )
        print(f"wrote {args.csv}")

    if args.out:
        plot(args.snr, curves, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
