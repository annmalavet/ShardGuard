from __future__ import annotations

import json
import os
import selectors
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from rich.console import Console

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
console = Console()


def _headers(session_id: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    base = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "x-session-id": session_id,
    }
    if extra:
        base.update(extra)
    return base


def _http_rpc(
    url: str,
    method: str,
    params: dict[str, Any],
    *,
    session_id: str,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }
    r = requests.post(
        url,
        headers=_headers(session_id, headers),
        data=json.dumps(payload),
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"{url} {method} error: {data['error']}")
    return data.get("result", {})


def _http_notify(
    url: str,
    method: str,
    params: dict[str, Any],
    *,
    session_id: str,
    headers: dict[str, str] | None = None,
) -> None:
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    }
    requests.post(
        url,
        headers=_headers(session_id, headers),
        data=json.dumps(payload),
        timeout=5.0,
    )


class _StdioRPC:
    """
    JSON-RPC over stdio with two framing modes:
      - 'jsonl' : one JSON message per line
      - 'lsp'   : Content-Length: N\r\n\r\n<body>
    """

    def __init__(
        self,
        cmd: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        framing: str = "jsonl",
    ):
        self.framing = framing
        self.proc = subprocess.Popen(
            [cmd, *(args or [])],
            cwd=cwd or PROJECT_ROOT,
            env={**os.environ, **(env or {})},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("Failed to create stdio pipes")

        # Drain stderr to console for visibility
        def _drain_stderr(pipe):
            try:
                for line in iter(pipe.readline, b""):
                    try:
                        console.print(
                            f"[STDERR] {line.decode(errors='replace').rstrip()}"
                        )
                    except Exception:
                        pass
            except Exception:
                pass

        threading.Thread(
            target=_drain_stderr, args=(self.proc.stderr,), daemon=True
        ).start()

        self.sel = selectors.DefaultSelector()
        self.sel.register(self.proc.stdout, selectors.EVENT_READ)
        self._buf = bytearray()

    def _read_available(self, max_bytes: int = 65536, timeout: float = 0.2) -> bytes:
        """Read whatever is available into buffer and return"""
        new = bytearray()
        events = self.sel.select(timeout=timeout)
        for key, _ in events:
            fd = key.fileobj.fileno()
            try:
                chunk = os.read(fd, max_bytes)
            except BlockingIOError:
                chunk = b""
            if chunk:
                self._buf.extend(chunk)
                new.extend(chunk)
        return bytes(new)

    # ----- LSP helpers -----
    def _read_lsp_frame(self, timeout: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        # read headers until \r\n\r\n - a double new line, this indicates end of headers
        while b"\r\n\r\n" not in self._buf:
            if self.proc.poll() is not None:
                raise RuntimeError(f"STDIO server exited: {self.proc.returncode}")
            if time.time() > deadline:
                raise TimeoutError("STDIO header read timeout")
            self._read_available(timeout=0.2)

        header_end = self._buf.find(b"\r\n\r\n")
        headers_blob = bytes(self._buf[:header_end])
        del self._buf[
            : header_end + 4
        ]  # drop headers + CRLFCRLF (The first Carriage Return Line Feed ends the last header line, the second creates the blank line. )

        content_length = None
        for line in headers_blob.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                try:
                    content_length = int(line.split(b":", 1)[1].strip())
                except Exception:
                    pass
        if content_length is None:
            raise RuntimeError("Missing Content-Length in stdio headers")

        deadline = time.time() + timeout
        while len(self._buf) < content_length:
            if self.proc.poll() is not None:
                raise RuntimeError("STDIO EOF while reading body")
            if time.time() > deadline:
                raise TimeoutError("STDIO body read timeout")
            self._read_available(timeout=0.2)

        body = bytes(self._buf[:content_length])
        del self._buf[:content_length]
        return json.loads(body.decode("utf-8"))

    # ----- JSONL helpers -----
    def _read_jsonl_frame(self, timeout: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        while b"\n" not in self._buf:
            if self.proc.poll() is not None:
                raise RuntimeError(f"STDIO server exited: {self.proc.returncode}")
            if time.time() > deadline:
                raise TimeoutError("STDIO line read timeout")
            self._read_available(timeout=0.2)

        line_end = self._buf.find(b"\n")
        line = bytes(self._buf[:line_end])
        del self._buf[: line_end + 1]
        if not line.strip():
            # skip blank lines
            return self._read_jsonl_frame(max(0.0, deadline - time.time()))
        return json.loads(line.decode("utf-8"))

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self.proc.poll() is not None:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        data = json.dumps(payload).encode("utf-8")
        if self.framing == "lsp":
            hdr = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
            self.proc.stdin.write(hdr)
            self.proc.stdin.write(data)
        else:
            self.proc.stdin.write(data + b"\n")
        self.proc.stdin.flush()

    def request(
        self, method: str, params: dict[str, Any], *, timeout: float = 30.0
    ) -> dict[str, Any]:
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"STDIO server already exited with code {self.proc.returncode}"
            )

        req_id = str(uuid.uuid4())
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        data = json.dumps(payload).encode("utf-8")

        # write according to framing
        if self.framing == "lsp":
            hdr = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
            self.proc.stdin.write(hdr)
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        else:  # jsonl
            self.proc.stdin.write(data + b"\n")
            self.proc.stdin.flush()

        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"STDIO RPC timeout waiting for {method}")
            frame = (
                self._read_lsp_frame(timeout)
                if self.framing == "lsp"
                else self._read_jsonl_frame(timeout)
            )
            if frame.get("id") == req_id:
                if "error" in frame:
                    raise RuntimeError(f"STDIO {method} error: {frame['error']}")
                return frame.get("result", {})

    def close(self):
        try:
            self.sel.unregister(self.proc.stdout)
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass


class MCPClient:
    def __init__(
        self,
        transport: str,
        http_url: str | None = None,
        http_headers: dict[str, str] | None = None,
        stdio_cmd: str | None = None,
        stdio_args: list[str] | None = None,
        stdio_cwd: str | None = None,
        stdio_env: dict[str, str] | None = None,
        session_id: str | None = None,
        stdio_framing: str = "jsonl",
    ):
        self.transport = transport
        self.session_id = session_id or f"cli-{uuid.uuid4().hex[:8]}"
        self.http_url = http_url.rstrip("/") if http_url else None
        self.http_headers = http_headers or {}
        self.stdio = None
        self._initialized = False
        if transport == "streamable-http":
            if not self.http_url:
                raise ValueError("streamable-http requires http_url")
        elif transport == "stdio":
            if not stdio_cmd:
                raise ValueError("stdio requires stdio_cmd")
            self.stdio = _StdioRPC(
                stdio_cmd,
                stdio_args,
                stdio_cwd or PROJECT_ROOT,
                stdio_env,
                framing=stdio_framing,
            )
        else:
            raise ValueError(f"Unsupported transport: {transport}")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification to the server."""
        if self.transport == "streamable-http":
            _http_notify(
                self.http_url,
                method,
                params or {},
                session_id=self.session_id,
                headers=self.http_headers,
            )
        elif self.stdio:
            self.stdio.notify(method, params or {})

    def initialize(
        self, *, timeout: float = 15.0, protocol_version: str | None = None
    ) -> dict[str, Any]:
        if self._initialized and self.transport == "stdio":
            return {}

        versions = (
            [protocol_version] if protocol_version else ["2025-06-18", "2025-05-01"]
        )
        last = None
        for ver in [v for v in versions if v]:
            params = {
                "clientInfo": {"name": "shardguard-cli", "version": "1.0.0"},
                "protocolVersion": ver,
                "capabilities": {"tools": {}},
            }
            try:
                if self.transport == "streamable-http":
                    res = _http_rpc(
                        self.http_url,
                        "initialize",
                        params,
                        session_id=self.session_id,
                        headers=self.http_headers,
                        timeout=timeout,
                    )
                else:
                    res = self.stdio.request("initialize", params, timeout=timeout)

                self.notify("notifications/initialized")
                self._initialized = True
                return res
            except Exception as e:
                last = e
        raise last or RuntimeError("initialize failed for all protocol versions")

    def tools_list(self, *, timeout: float = 15.0) -> list[dict[str, Any]]:
        if self.transport == "streamable-http":
            res = _http_rpc(
                self.http_url,
                "tools/list",
                {},
                session_id=self.session_id,
                headers=self.http_headers,
                timeout=timeout,
            )
        else:
            res = self.stdio.request("tools/list", {}, timeout=timeout)
        return res.get("tools", [])

    def resources_list(self, *, timeout: float = 15.0) -> list[dict[str, Any]]:
        try:
            if self.transport == "streamable-http":
                res = _http_rpc(
                    self.http_url,
                    "resources/list",
                    {},
                    session_id=self.session_id,
                    headers=self.http_headers,
                    timeout=timeout,
                )
            else:
                res = self.stdio.request("resources/list", {}, timeout=timeout)
            return res.get("resources", [])
        except Exception:
            return []

    def tools_call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float = 60.0,
    ) -> list[dict[str, Any]]:
        params = {"name": name, "arguments": arguments or {}}
        if self.transport == "streamable-http":
            res = _http_rpc(
                self.http_url,
                "tools/call",
                params,
                session_id=self.session_id,
                headers=self.http_headers,
                timeout=timeout,
            )
        else:
            res = self.stdio.request("tools/call", params, timeout=timeout)
        return res.get("content", [])

    def close(self):
        if self.stdio:
            self.stdio.close()
