"""CLI entry point for rm-stream."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rm_stream.pipeline import AsyncPipeline

LEGACY = Path(__file__).resolve().parents[3] / "compress-ai-gray-minimal"
DEFAULT_MBT = str(LEGACY / "mbt2018-mean-1-e522738d.pth.tar")
DEFAULT_SR = str(
    LEGACY / "checkpoints" / "expA_baseline_ch64_n4_15ep" / "e2e_sr_best.pth.tar"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RoboMaster low-bandwidth camera stream client (M0 dummy)"
    )
    parser.add_argument("--input", "-i", required=True, help="path to input video file")
    parser.add_argument("--mbt-checkpoint", default=DEFAULT_MBT, help="path to MBT2018-mean q1 checkpoint")
    parser.add_argument("--sr-checkpoint", default=DEFAULT_SR, help="path to MSA²-SR checkpoint")
    parser.add_argument("--max-frames", type=int, default=0, help="max frames to process (0 = all)")
    args = parser.parse_args(argv)

    pipeline = AsyncPipeline(
        video_path=args.input,
        mbt_checkpoint=args.mbt_checkpoint,
        sr_checkpoint=args.sr_checkpoint,
        max_frames=args.max_frames,
    )

    stats = asyncio.run(pipeline.run())
    print(f"\nPipeline finished:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
