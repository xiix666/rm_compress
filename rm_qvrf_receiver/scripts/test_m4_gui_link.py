#!/usr/bin/env python3
"""Real GUI link test: C++ TX -> receiver -> GUI decode/display.

The test starts the real PyQt GUI, sends a real compressed stream over either
the serial/MQTT path or local TCP IPC, then parses the GUI RX
debug log and TX log.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "compress-ai-gray-minimal" / "12月19日 中科大 东南.mp4"


def default_sender_root() -> Path:
    """Find a sender tree for either source checkout or dist packages."""
    candidates = [
        ROOT,
        ROOT.parent / "rm_m4_sender_test",
        ROOT.parent / "sender",
    ]
    for candidate in candidates:
        if (candidate / "onboard/build/rm_compress_cli").exists() or (candidate / "bin/rm_compress_cli").exists():
            return candidate
    return ROOT


def sender_binary(sender_root: Path) -> Path:
    candidates = [
        sender_root / "onboard/build/rm_compress_cli",
        sender_root / "bin/rm_compress_cli",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, text=True, **kwargs)


def generate_raw(video: Path, raw: Path, frames: int, width: int, height: int) -> None:
    code = f"""
import cv2
cap = cv2.VideoCapture({str(video)!r})
count = 0
with open({str(raw)!r}, 'wb') as fout:
    for _ in range({frames}):
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.resize(f, ({width}, {height}), interpolation=cv2.INTER_AREA)
        fout.write(f.tobytes())
        count += 1
cap.release()
print(count)
"""
    proc = run(["uv", "run", "--directory", str(ROOT / "client"), "python", "-c", code],
               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError("raw frame generation failed")


def parse_rx_log(path: Path) -> dict:
    chunk_re = re.compile(
        r"chunk frame=(\d+) cid=(\d+)/(\d+) .* assembled=(True|False).* done=(\d+) stale=(\d+)"
    )
    ready_re = re.compile(
        r"frame_ready count=(\d+) fps=([0-9.]+) codec_ms=([0-9.]+) sr_ms=([0-9.]+)"
    )
    frames: dict[int, set[int]] = defaultdict(set)
    rows = []
    readies = []
    recovered = 0
    decode_errors = 0
    if not path.exists():
        return {
            "chunks": 0,
            "frames_seen": 0,
            "complete_chunksets": 0,
            "only0": [],
            "only1": [],
            "ready_count": 0,
            "ready_last": None,
            "recovered": 0,
            "decode_errors": 0,
        }
    for line in path.read_text(errors="replace").splitlines():
        if "frame_recovered_single" in line:
            recovered += 1
        if "decode_error" in line:
            decode_errors += 1
        m = chunk_re.search(line)
        if m:
            fid = int(m.group(1))
            cid = int(m.group(2))
            total = int(m.group(3))
            frames[fid].add(cid)
            rows.append((fid, cid, total, int(m.group(5)), int(m.group(6))))
        m = ready_re.search(line)
        if m:
            readies.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    expected_counts: dict[int, int] = {}
    for fid, _cid, total, _done, _stale in rows:
        expected_counts[fid] = total
    complete = [
        fid for fid, cids in frames.items()
        if len(cids) == expected_counts.get(fid, 0)
        and cids == set(range(expected_counts.get(fid, 0)))
    ]
    only0 = [fid for fid, cids in frames.items() if cids == {0}]
    only1 = [fid for fid, cids in frames.items() if cids == {1}]
    return {
        "chunks": len(rows),
        "frames_seen": len(frames),
        "complete_chunksets": len(complete),
        "only0": sorted(only0),
        "only1": sorted(only1),
        "ready_count": readies[-1][0] if readies else 0,
        "ready_samples": len(readies),
        "ready_last": readies[-1] if readies else None,
        "fps_tail": [r[1] for r in readies[-10:]],
        "recovered": recovered,
        "decode_errors": decode_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--codec-profile", choices=("legacy128x2x24", "codec192x4x12", "codec256x4x12", "codec256x6x8", "codec320x8x6", "codec448x9x5"),
                        default="legacy128x2x24")
    parser.add_argument("--codec-size", type=int, default=0)
    parser.add_argument("--display-size", type=int, default=0)
    parser.add_argument("--chunks-per-frame", type=int, default=0)
    parser.add_argument("--fec-data-chunks", type=int, default=-1)
    parser.add_argument("--tx-device", default="GPU.0")
    parser.add_argument("--rx-backend", default=os.environ.get("RM_STREAM_BACKEND", "cuda"))
    parser.add_argument("--torch-device", default=os.environ.get("RM_STREAM_TORCH_DEVICE", "cuda:0"))
    parser.add_argument("--client-id", default="1")
    parser.add_argument("--mqtt-host", default="192.168.12.1")
    parser.add_argument("--mqtt-port", type=int, default=3333)
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--transport", choices=("serial", "offline-debug"), default="serial",
                        help="serial uses 0x0310->MQTT; offline-debug sends raw 300B chunks over local TCP IPC")
    parser.add_argument("--offline-debug", action="store_true",
                        help="Shortcut for --transport offline-debug")
    parser.add_argument("--ipc-host", default="127.0.0.1")
    parser.add_argument("--ipc-port", type=int, default=49031)
    parser.add_argument("--prebuffer-chunks", type=int, default=4)
    parser.add_argument("--tail-flush-chunks", type=int, default=4)
    parser.add_argument("--chunk-rate-hz", type=float, default=48.0)
    parser.add_argument("--max-queue-chunks", type=int, default=16)
    parser.add_argument("--chunk-order", default="0312")
    parser.add_argument("--debug-rx-chunks", action="store_true")
    parser.add_argument("--enable-sr", action="store_true")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--sender-root", type=Path, default=default_sender_root())
    parser.add_argument("--raw", type=Path, default=Path("/tmp/m4_gui_link_frames.rgb"))
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--gui-wait", type=float, default=8.0)
    parser.add_argument("--settle", type=float, default=4.0)
    parser.add_argument("--min-ready-ratio", type=float, default=0.95)
    args = parser.parse_args()
    if args.offline_debug:
        args.transport = "offline-debug"

    profile_defaults = {
        "legacy128x2x24": (128, 256, 2, 24.0),
        "codec192x4x12": (192, 512, 4, 12.0),
        "codec256x4x12": (256, 512, 4, 12.0),
        "codec256x6x8": (256, 512, 6, 8.0),
        "codec320x8x6": (320, 512, 8, 6.0),
        "codec448x9x5": (448, 512, 9, 5.0),
    }
    default_codec, default_display, default_chunks, default_fps = profile_defaults[args.codec_profile]
    if args.codec_size <= 0:
        args.codec_size = default_codec
    if args.display_size <= 0:
        args.display_size = default_display
    if args.chunks_per_frame <= 0:
        args.chunks_per_frame = default_chunks
    if args.fec_data_chunks < 0:
        args.fec_data_chunks = 0
    if args.fps <= 0:
        args.fps = default_fps

    args.log_dir.mkdir(parents=True, exist_ok=True)
    rx_log = args.log_dir / "rm_m4_gui_link_rx.log"
    gui_log = args.log_dir / "rm_m4_gui_link_gui.log"
    tx_log = args.log_dir / "rm_m4_gui_link_tx.log"
    for p in (rx_log, gui_log, tx_log):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    args.sender_root = args.sender_root.resolve()
    tx_bin = sender_binary(args.sender_root)
    tx_model = args.sender_root / (
        "models/mbt_g_a.xml" if args.codec_size == 128 else f"models/mbt_g_a_{args.codec_size}.xml"
    )
    if not tx_bin.exists():
        print(f"FAIL: sender binary not found: {tx_bin}")
        print("Hint: pass --sender-root /path/to/rm_m4_sender_test")
        return 1
    if not tx_model.exists():
        print(f"FAIL: sender model not found: {tx_model}")
        return 1

    print("=== Generate raw frames ===")
    generate_raw(args.video, args.raw, args.frames, args.codec_size, args.codec_size)

    print("=== Start GUI ===")
    env = os.environ.copy()
    env["RM_STREAM_DEBUG_RX_LOG"] = str(rx_log)
    env["RM_STREAM_DEBUG_RX_CHUNKS"] = "1" if args.debug_rx_chunks else "0"
    env["RM_STREAM_BACKEND"] = args.rx_backend
    if args.torch_device:
        env["RM_STREAM_TORCH_DEVICE"] = args.torch_device
    gui_cmd = [
        "uv", "run", "--directory", str(ROOT / "client"), "python", "-m", "rm_stream.gui",
        "--rx-backend", args.rx_backend,
        "--codec-profile", args.codec_profile,
        "--codec-size", str(args.codec_size),
        "--display-size", str(args.display_size),
    ]
    if args.transport == "offline-debug":
        gui_cmd += [
            "--receive-mode", "ipc",
            "--ipc-host", args.ipc_host,
            "--ipc-port", str(args.ipc_port),
        ]
    else:
        gui_cmd += [
            "--receive-mode", "mqtt",
            "--mqtt-host", args.mqtt_host,
            "--mqtt-port", str(args.mqtt_port),
            "--client-id", args.client_id,
        ]
    if args.torch_device:
        gui_cmd += ["--torch-device", args.torch_device]
    if args.enable_sr:
        gui_cmd.append("--enable-sr")
    gui_f = gui_log.open("w")
    gui_proc = subprocess.Popen(gui_cmd, cwd=str(ROOT), env=env, stdout=gui_f, stderr=subprocess.STDOUT)
    print(f"GUI PID: {gui_proc.pid}")
    time.sleep(args.gui_wait)

    print("=== Start TX ===")
    tx_env = os.environ.copy()
    tx_env["LD_LIBRARY_PATH"] = (
        f"{ROOT / 'client/.venv/lib/python3.11/site-packages/openvino/libs'}:"
        f"{args.sender_root / 'onboard/build'}:"
        f"{args.sender_root / 'bin'}:"
        f"{tx_env.get('LD_LIBRARY_PATH', '')}"
    )
    tx_cmd = [
        str(tx_bin),
        "-n", str(args.frames),
        "-d", args.tx_device,
        "-m", str(tx_model),
        "--input", str(args.raw),
        "-w", str(args.codec_size),
        "-H", str(args.codec_size),
        "--fps", str(args.fps),
        "--codec-size", str(args.codec_size),
        "--chunks-per-frame", str(args.chunks_per_frame),
        "--fec-data-chunks", str(args.fec_data_chunks),
        "--prebuffer-chunks", str(args.prebuffer_chunks),
        "--tail-flush-chunks", str(args.tail_flush_chunks),
        "--chunk-rate-hz", str(args.chunk_rate_hz),
        "--max-queue-chunks", str(args.max_queue_chunks),
        "--chunk-order", args.chunk_order,
    ]
    if args.transport == "offline-debug":
        tx_cmd += ["--ipc-host", args.ipc_host, "--ipc-port", str(args.ipc_port)]
    else:
        tx_cmd += [
            "-p", args.serial_port,
            "-b", "921600",
            "-r", "1",
            "--serial-wait",
        ]
    with tx_log.open("w") as f:
        tx_proc = run(tx_cmd, cwd=str(ROOT), env=tx_env, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(args.settle)

    print("=== Stop GUI ===")
    if gui_proc.poll() is None:
        gui_proc.terminate()
        try:
            gui_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            gui_proc.kill()
            gui_proc.wait(timeout=3)
    gui_f.close()

    rx = parse_rx_log(rx_log)
    tx_text = tx_log.read_text(errors="replace") if tx_log.exists() else ""
    print("=== TX summary ===")
    for line in tx_text.splitlines():
        if any(k in line for k in (
            "TX prebuffer", "Tail flush", "Packet mode", "Chunks sent",
            "Errors:", "Over budget:", "Duration:", "Compress:"
        )):
            print(line)
    print("=== GUI RX summary ===")
    print(f"chunks={rx['chunks']} frames_seen={rx['frames_seen']} complete_chunksets={rx['complete_chunksets']}")
    print(f"only0={rx['only0'][:30]} count={len(rx['only0'])}")
    print(f"only1={rx['only1'][:30]} count={len(rx['only1'])}")
    print(f"frame_ready={rx['ready_count']} ready_samples={rx['ready_samples']} last={rx['ready_last']}")
    print(f"fps_tail={rx['fps_tail']}")
    print(f"recovered={rx['recovered']} decode_errors={rx['decode_errors']}")
    print(f"logs: rx={rx_log} gui={gui_log} tx={tx_log}")

    if tx_proc.returncode != 0:
        print(f"FAIL: TX exited {tx_proc.returncode}")
        return 1
    min_ready = int(args.frames * args.min_ready_ratio)
    if rx["ready_count"] < min_ready:
        print(f"FAIL: frame_ready {rx['ready_count']} < {min_ready}")
        return 2
    if rx["decode_errors"] > 0:
        print("FAIL: decode errors present")
        return 3
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
