"""
Unity visualization bridge — sends environment state to Unity via TCP socket.
Unity acts as a TCP client that connects to this server.
"""

import json
import socket
import threading
import time
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

_server_socket = None
_client_socket = None
_server_thread = None
_connection_attempted = False
_lock = threading.Lock()
_connected_event = threading.Event()

HOST = "127.0.0.1"
PORT = 9876


def _start_server():
    """Start TCP server in background thread, waiting for Unity to connect."""
    global _server_socket, _client_socket

    try:
        _server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _server_socket.bind((HOST, PORT))
        _server_socket.listen(1)
        _server_socket.settimeout(120)
        print(f"[Rendering] Waiting for Unity to connect on {HOST}:{PORT}...")

        client, addr = _server_socket.accept()
        with _lock:
            _client_socket = client
        _connected_event.set()
        print(f"[Rendering] Unity connected from {addr}")
    except socket.timeout:
        print("[Rendering] No Unity client connected (timeout). Running headless.")
        _connected_event.set()  # unblock main thread
        if _server_socket:
            _server_socket.close()
            _server_socket = None
    except OSError as e:
        print(f"[Rendering] Server error: {e}. Running headless.")
        _connected_event.set()  # unblock main thread
        if _server_socket:
            _server_socket.close()
            _server_socket = None


def ensure_server():
    """Start the TCP server (non-blocking). Call wait_for_connection() to block."""
    global _server_thread, _connection_attempted
    with _lock:
        if _connection_attempted:
            return
        _connection_attempted = True
    _server_thread = threading.Thread(target=_start_server, daemon=True)
    _server_thread.start()


def wait_for_connection(timeout=120):
    """Block until Unity connects or timeout. Call after ensure_server()."""
    ensure_server()
    _connected_event.wait(timeout=timeout)


def send_state_to_unity(state_dict):
    """Serialize environment state and send to Unity client."""
    global _client_socket

    with _lock:
        client = _client_socket

    if client is None:
        return  # no Unity client connected, skip silently

    try:
        data = json.dumps(state_dict, cls=NumpyEncoder) + "\n"
        client.sendall(data.encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError, OSError):
        print("[Rendering] Unity disconnected.")
        with _lock:
            _client_socket = None


def close_connection():
    """Clean up sockets."""
    global _server_socket, _client_socket, _connection_attempted
    if _client_socket:
        _client_socket.close()
        _client_socket = None
    if _server_socket:
        _server_socket.close()
        _server_socket = None
    _connection_attempted = False
    _connected_event.clear()
