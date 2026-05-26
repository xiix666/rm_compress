#!/usr/bin/env python3
import argparse
import time

import paho.mqtt.client as mqtt


def probe(host: str, port: int, client_id: str, timeout: float) -> str:
    result: list[str] = []

    def on_connect(client, userdata, flags, reason, props):
        result.append(str(reason))

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        protocol=mqtt.MQTTv311,
    )
    client.on_connect = on_connect
    try:
        client.connect(host, port, keepalive=10)
        client.loop_start()
        deadline = time.time() + timeout
        while time.time() < deadline and not result:
            time.sleep(0.05)
        client.loop_stop()
        client.disconnect()
    except Exception as exc:
        return f"EXC {exc!r}"
    return result[0] if result else "NO_CALLBACK"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.12.1")
    parser.add_argument("--port", type=int, default=3333)
    parser.add_argument("--ids", default="1,2,3,101,102,103,104,105")
    parser.add_argument("--timeout", type=float, default=1.5)
    args = parser.parse_args()
    for cid in [x.strip() for x in args.ids.split(",") if x.strip()]:
        print(f"{cid}: {probe(args.host, args.port, cid, args.timeout)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
