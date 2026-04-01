from __future__ import annotations

import argparse
import codecs
import os
import selectors
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_prefixed_text(handle: object, buffer: str, text: str) -> str:
    buffer += text
    while True:
        newline_index = buffer.find("\n")
        if newline_index < 0:
            return buffer
        line = buffer[: newline_index + 1]
        buffer = buffer[newline_index + 1 :]
        handle.write(f"[{utc_now_iso()}] {line}")
        handle.flush()


def _flush_remaining(handle: object, buffer: str) -> None:
    if not buffer:
        return
    handle.write(f"[{utc_now_iso()}] {buffer}")
    handle.flush()


def _stream_with_timestamps(log_path: Path, cwd: Path, cmd: list[str]) -> int:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, data="stderr")
    decoders = {
        "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
    }
    buffers = {"stdout": "", "stderr": ""}

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now_iso()}] launching: {' '.join(cmd)}\n")
        handle.flush()
        while selector.get_map():
            for key, _ in selector.select():
                chunk = os.read(key.fileobj.fileno(), 4096)
                stream_name = key.data
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                text = decoders[stream_name].decode(chunk)
                if text:
                    buffers[stream_name] = _write_prefixed_text(handle, buffers[stream_name], text)

        for stream_name, decoder in decoders.items():
            tail = decoder.decode(b"", final=True)
            if tail:
                buffers[stream_name] = _write_prefixed_text(handle, buffers[stream_name], tail)
            _flush_remaining(handle, buffers[stream_name])

        returncode = proc.wait()
        handle.write(f"[{utc_now_iso()}] modal run exited with code {returncode}\n")
        handle.flush()
        return returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = list(args.cmd)
    if cmd[:1] == ["--"]:
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command provided.")

    return _stream_with_timestamps(
        log_path=Path(args.log_path),
        cwd=Path(args.cwd),
        cmd=cmd,
    )


if __name__ == "__main__":
    raise SystemExit(main())
