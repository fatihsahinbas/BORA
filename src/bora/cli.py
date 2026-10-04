"""Command line interface.

    bora info
    bora send secret.jpg -o out.wav
    bora recv out.wav -o recovered.jpg
    bora loopback secret.jpg --snr 12
    bora sweep --out ber.csv
    bora play out.wav
    bora listen -o heard.wav --seconds 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import audio, channel, link
from .config import Config

__all__ = ["main", "build_parser"]


def _config_from(args: argparse.Namespace) -> Config:
    return Config(
        fs=args.rate,
        n_fft=args.fft,
        cp_len=args.cp,
        f_lo=args.f_lo,
        f_hi=args.f_hi,
    )


def _report(result: link.Reception, *, verbose: bool = True) -> None:
    print(result.summary())
    if not verbose:
        return
    if result.sync:
        print(
            f"  sync         offset {result.sync.offset}, "
            f"confidence {result.sync.confidence:.1f}"
        )
    if result.quality:
        print(
            f"  constellation EVM {result.quality.evm_degrees:.1f} deg, "
            f"est SNR {result.quality.est_snr_db:.1f} dB"
        )
    if result.fec and result.fec.blocks:
        print(
            f"  fec          {result.fec.blocks} blocks, "
            f"{result.fec.repaired} repaired, {result.fec.failed} failed, "
            f"{result.fec.corrected_bytes} byte errors fixed"
        )


# ------------------------------------------------------------------ commands


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    cfg.validate()
    print("BORA - Byte Over Resonant Air")
    print(cfg.describe())
    for size in (1_000, 10_000, 100_000):
        print(
            f"  {size:>7} bytes  ->  {link.airtime(size, cfg):>6.1f} s  "
            f"({link.goodput(size, cfg):>5.0f} bps with FEC)"
        )
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    payload = Path(args.file).read_bytes()
    wave = link.transmit(payload, cfg, use_fec=not args.no_fec)
    audio.write_wav(args.out, wave, cfg)

    seconds = len(wave) / cfg.fs
    print(f"{len(payload)} bytes -> {args.out}")
    print(f"  airtime      {seconds:.2f} s")
    print(f"  goodput      {len(payload) * 8 / seconds:.0f} bps")
    print(f"  fec          {'off' if args.no_fec else 'on'}")
    return 0


def cmd_recv(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    signal, fs = audio.read_wav(args.file)
    if fs != cfg.fs:
        print(f"warning: file is {fs} Hz, decoder expects {cfg.fs} Hz", file=sys.stderr)

    result = link.receive(signal, cfg)
    _report(result)

    if result.payload:
        Path(args.out).write_bytes(result.payload)
        print(f"  wrote        {args.out}")
    return 0 if result.ok else 1


def cmd_loopback(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    rng = np.random.default_rng(args.seed)

    payload = (
        Path(args.file).read_bytes()
        if args.file
        else bytes(rng.integers(0, 256, args.size, dtype=np.uint8))
    )

    wave = link.transmit(payload, cfg, use_fec=not args.no_fec)
    if args.realistic:
        rx = channel.room(wave, cfg.fs, rng, snr_db=args.snr, ppm=args.ppm,
                          bursts=args.bursts)
    else:
        rx = channel.awgn(wave, args.snr, rng)

    result = link.receive(rx, cfg)
    _report(result)
    print(f"  identical    {result.payload == payload}")
    return 0 if result.ok else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    rng = np.random.default_rng(args.seed)
    payload = bytes(rng.integers(0, 256, args.size, dtype=np.uint8))

    rows = [("snr_db", "fec", "success", "trials", "ber")]
    print(f"{'SNR dB':>7}  {'FEC':>4}  {'success':>9}  {'BER':>10}")

    for use_fec in (False, True):
        wave = link.transmit(payload, cfg, use_fec=use_fec)
        for snr in args.snr_points:
            good = 0
            ber_acc = []
            for _ in range(args.trials):
                rx = channel.awgn(wave, snr, rng)
                result = link.receive(rx, cfg)
                good += result.ok
                ber_acc.append(_ber(payload, result.payload))
            ber = float(np.mean(ber_acc))
            label = "on" if use_fec else "off"
            print(f"{snr:>7}  {label:>4}  {good:>4}/{args.trials:<4}  {ber:>10.2e}")
            rows.append((snr, label, good, args.trials, f"{ber:.6e}"))

    if args.out:
        Path(args.out).write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n")
        print(f"wrote {args.out}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    signal, _ = audio.read_wav(args.file)
    print(f"playing {len(signal) / cfg.fs:.1f} s ...")
    audio.play(signal, cfg)
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    print(f"recording {args.seconds:.1f} s ...")
    signal = audio.record(args.seconds, cfg)
    audio.write_wav(args.out, signal, cfg)
    print(f"wrote {args.out}")

    result = link.receive(signal, cfg)
    _report(result)
    if result.payload and args.payload:
        Path(args.payload).write_bytes(result.payload)
        print(f"  wrote        {args.payload}")
    return 0 if result.ok else 1


def _ber(a: bytes, b: bytes) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    xor = np.frombuffer(a[:n], np.uint8) ^ np.frombuffer(b[:n], np.uint8)
    errors = float(np.unpackbits(xor).sum())
    return (errors + 8 * (len(a) - n)) / (8 * len(a))


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bora", description="Byte Over Resonant Air - acoustic OFDM modem"
    )
    parser.add_argument("--rate", type=int, default=48_000, help="sample rate in Hz")
    parser.add_argument("--fft", type=int, default=512, help="FFT size")
    parser.add_argument("--cp", type=int, default=128, help="cyclic prefix in samples")
    parser.add_argument("--f-lo", type=float, default=1_000.0, help="lowest carrier Hz")
    parser.add_argument("--f-hi", type=float, default=6_000.0, help="highest carrier Hz")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="print the physical layer parameters")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("send", help="encode a file into a WAV")
    p.add_argument("file")
    p.add_argument("-o", "--out", default="bora.wav")
    p.add_argument("--no-fec", action="store_true", help="transmit without parity")
    p.set_defaults(func=cmd_send)

    p = sub.add_parser("recv", help="decode a WAV back into a file")
    p.add_argument("file")
    p.add_argument("-o", "--out", default="received.bin")
    p.set_defaults(func=cmd_recv)

    p = sub.add_parser("loopback", help="encode and decode through a simulated channel")
    p.add_argument("file", nargs="?", help="payload file; random bytes if omitted")
    p.add_argument("--size", type=int, default=4_000, help="random payload size")
    p.add_argument("--snr", type=float, default=15.0)
    p.add_argument("--ppm", type=float, default=60.0, help="receiver clock error")
    p.add_argument("--bursts", type=int, default=0, help="interference bursts")
    p.add_argument("--realistic", action="store_true",
                   help="echoes, transducer response and clock drift, not just noise")
    p.add_argument("--no-fec", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_loopback)

    p = sub.add_parser(
        "sweep", help="measure error rate against SNR, with and without FEC"
    )
    p.add_argument("--size", type=int, default=4_000)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write results as CSV")
    p.add_argument(
        "--snr-points",
        type=float,
        nargs="+",
        default=[18, 15, 13, 11, 10, 9, 8, 7, 6, 5, 4],
    )
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("play", help="play a WAV through the speaker")
    p.add_argument("file")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("listen", help="record from the microphone and decode")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("-o", "--out", default="heard.wav")
    p.add_argument("--payload", help="write the decoded payload here")
    p.set_defaults(func=cmd_listen)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
