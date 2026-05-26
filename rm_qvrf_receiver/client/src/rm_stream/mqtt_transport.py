"""MQTT transport for compressed bitstream chunks over rm_stream/chunks topic.

Producer side: publish_frame() packs bitstream into 300B chunks and publishes to MQTT.
Consumer side: start_receiving() subscribes, reassembles via FrameAssembler,
               calls callback(frame_id, bitstream) on complete frames.

Designed for producer and consumer to run in separate processes, connected by the MQTT broker.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import paho.mqtt.client as mqtt

from rm_stream.protocol import pack_frame, parse_chunk_header
from rm_stream.frame_assembler import FrameAssembler

LOGGER = logging.getLogger(__name__)


class MqttTransport:
    """MQTT transport for compressed bitstream chunks.

    Producer (onboard) side:
        transport = MqttTransport(broker_host="192.168.12.1", broker_port=3333)
        transport.publish_frame(frame_id, bitstream)
        transport.stop()

    Consumer (client) side:
        def on_frame(frame_id, bitstream):
            # decode and display
        transport = MqttTransport(broker_host="192.168.12.1", broker_port=3333)
        transport.start_receiving(on_frame)
        # ... wait ...
        transport.stop()
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic: str = "rm_stream/chunks",
        client_id: str = "",
    ) -> None:
        self._host = broker_host
        self._port = broker_port
        self._topic = topic
        self._client_id = client_id or f"rm_stream_{id(self):x}"

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self._client_id,
            protocol=mqtt.MQTTv311,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        self._assembler = FrameAssembler(timeout_sec=0.12)
        self._callback: Callable[[int, bytes], None] | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._started = False

        # Stats
        self.chunks_sent: int = 0
        self.chunks_received: int = 0
        self.chunks_corrupt: int = 0
        self.frames_complete: int = 0

    # ------------------------------------------------------------------
    # Producer (onboard) side
    # ------------------------------------------------------------------

    def publish_frame(self, frame_id: int, bitstream: bytes) -> None:
        """Pack bitstream into 300B chunks and publish each to MQTT topic.

        Auto-connects to the broker on first call.
        """
        if not self._started:
            self._client.connect(self._host, self._port, keepalive=30)
            self._client.loop_start()
            self._started = True
            LOGGER.info(
                "MQTT publisher [%s] connected to %s:%d",
                self._client_id, self._host, self._port,
            )

        chunks = pack_frame(frame_id, bitstream)
        for chunk in chunks:
            self._client.publish(self._topic, chunk, qos=0)
            self.chunks_sent += 1

    # ------------------------------------------------------------------
    # Consumer (client) side
    # ------------------------------------------------------------------

    def start_receiving(self, callback: Callable[[int, bytes], None]) -> None:
        """Start async MQTT receive loop.

        Connects to broker, subscribes to topic, and starts the paho network
        loop in a background thread.  Calls ``callback(frame_id, bitstream)``
        from the MQTT thread whenever a complete frame is reassembled.
        """
        self._callback = callback
        self._client.connect(self._host, self._port, keepalive=30)
        self._client.loop_start()
        self._started = True
        LOGGER.info(
            "MQTT receiver [%s] connected to %s:%d, topic=%s",
            self._client_id, self._host, self._port, self._topic,
        )

    def stop(self) -> None:
        """Disconnect from MQTT broker and stop network loop."""
        if self._started:
            self._client.loop_stop()
            self._client.disconnect()
            self._started = False
            self._connected = False
            LOGGER.info("MQTT transport [%s] stopped", self._client_id)

    # ------------------------------------------------------------------
    # MQTT callbacks (private)
    # ------------------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        _userdata: object,
        _flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        _properties: mqtt.Properties | None,
    ) -> None:
        if reason_code != 0:
            LOGGER.error("MQTT connect failed: %s", reason_code)
            return

        self._connected = True
        LOGGER.info(
            "MQTT [%s] connected, subscribing to %s",
            self._client_id, self._topic,
        )
        client.subscribe(self._topic, qos=0)

    def _on_disconnect(
        self,
        _client: mqtt.Client,
        _userdata: object,
        _disconnect_flags: mqtt.DisconnectFlags,
        reason_code: mqtt.ReasonCode,
        _properties: mqtt.Properties | None,
    ) -> None:
        self._connected = False
        LOGGER.warning("MQTT [%s] disconnected: %s", self._client_id, reason_code)

    def _on_message(
        self, _client: mqtt.Client, _userdata: object, msg: mqtt.MQTTMessage
    ) -> None:
        chunk = bytes(msg.payload)
        self.chunks_received += 1

        # Parse header for validation and to capture frame_id
        header = parse_chunk_header(chunk)
        if header is None:
            self.chunks_corrupt += 1
            return

        with self._lock:
            result = self._assembler.add_chunk(chunk)

        if result is not None and self._callback is not None:
            self.frames_complete += 1
            self._callback(header.frame_id, result)
