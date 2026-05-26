"""MQTT worker — paho-mqtt wrapper running in a QThread."""

import paho.mqtt.client as mqtt
from PyQt5.QtCore import QObject, QTimer, pyqtSignal


class MqttWorker(QObject):
    """QObject that wraps paho-mqtt and emits Qt signals from a QThread.

    Move this object to a QThread via worker.moveToThread(thread),
    then call start() — all paho callbacks will run in the thread's
    event loop without blocking the GUI.
    """

    msg_received = pyqtSignal(str, bytes)
    connection_changed = pyqtSignal(bool)

    def __init__(self, broker_host, broker_port, client_id, topics):
        super().__init__()
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._client_id = client_id
        self._topics = list(topics) if topics else []
        self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Create paho client, set callbacks, connect, and start loop."""
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self._client_id,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        self._client.connect_async(self._broker_host, self._broker_port)
        self._client.loop_start()

    def stop(self):
        """Stop the event loop and disconnect gracefully."""
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

    # ------------------------------------------------------------------
    # paho callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self.connection_changed.emit(True)
        for topic in self._topics:
            client.subscribe(topic)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connection_changed.emit(False)
        # Auto-reconnect after 2 seconds
        if self._client is not None and reason_code != 0:
            QTimer.singleShot(2000, self._try_reconnect)

    def _on_message(self, client, userdata, msg):
        self.msg_received.emit(msg.topic, msg.payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_reconnect(self):
        """Attempt reconnection. Called by QTimer after disconnect."""
        if self._client is not None:
            try:
                self._client.connect(self._broker_host, self._broker_port)
            except Exception:
                # Retry after another 2 s
                QTimer.singleShot(2000, self._try_reconnect)
