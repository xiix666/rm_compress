#!/usr/bin/env python3
import argparse
import socket
import time
import random
import paho.mqtt.client as mqtt
from rm_custom_client.proto import rm_custom_client_pb2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=49031)
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=3333)
    parser.add_argument("--topic", default="CustomByteBlock")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--client-id", default="ipc_to_mqtt_bridge")

    parser.add_argument("--loss-rate", type=float, default=0.10)

    args = parser.parse_args()

    mc = mqtt.Client(client_id=args.client_id)
    mc.connect(args.mqtt_host, args.mqtt_port, keepalive=30)
    mc.loop_start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.listen_host, args.listen_port))
    srv.listen(1)

    print(f"[bridge] TCP listen {args.listen_host}:{args.listen_port}")
    print(f"[bridge] MQTT publish {args.mqtt_host}:{args.mqtt_port} topic={args.topic}")
    print(f"[bridge] chunk_size={args.chunk_size}")
    print(f"[bridge] loss_rate={args.loss_rate * 100:.1f}%")

    published = 0
    dropped = 0
    total = 0

    while True:
        print("[bridge] waiting sender connection...")
        conn, addr = srv.accept()
        print(f"[bridge] sender connected from {addr}")

        buf = bytearray()

        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    print("[bridge] sender disconnected")
                    break

                buf.extend(data)

                while len(buf) >= args.chunk_size:
                    chunk = bytes(buf[:args.chunk_size])
                    del buf[:args.chunk_size]

                    total += 1

                    if random.random() < args.loss_rate:
                        dropped += 1
                        if dropped % 20 == 0:
                            print(f"[bridge] dropped chunks={dropped}, total={total}")
                        continue

                    msg = rm_custom_client_pb2.CustomByteBlock()
                    msg.data = chunk
                    payload = msg.SerializeToString()

                    mc.publish(args.topic, payload, qos=1)
                    published += 1

                    if published % 48 == 0:
                        print(
                            f"[bridge] published={published}, "
                            f"dropped={dropped}, total={total}"
                        )

        except ConnectionResetError:
            print("[bridge] sender connection reset")
        finally:
            conn.close()
            time.sleep(0.2)


if __name__ == "__main__":
    main()