"""Punto de entrada: python -m motion_art_lab.main"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Motion Art Lab — convierte el movimiento en arte generativo.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Inicia con la demo sintética, sin abrir la cámara.",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--frames", type=int, default=130, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    from motion_art_lab.module2_experience.app import run_app

    frames = args.frames if args.self_test else None
    return run_app(prefer_demo=args.demo or args.self_test, self_test_frames=frames)


if __name__ == "__main__":
    raise SystemExit(main())
