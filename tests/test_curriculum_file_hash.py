"""Hash verification can restart an interrupted read without using partial data."""
import errno
import hashlib
import io
from pathlib import Path

import pytest

from src.mylonas_ijepa_wm import curriculum_runtime as runtime


def test_hash_restarts_from_beginning_after_transient_eio(monkeypatch):
    class Interrupted(io.BytesIO):
        def read(self, size):
            if self.tell():
                raise OSError(errno.EIO, "Input/output error")
            return super().read(2)

    payload = b"benchmark checkpoint"
    attempts = iter([Interrupted(payload), io.BytesIO(payload)])
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: next(attempts))
    monkeypatch.setattr("time.sleep", lambda _: None)
    assert runtime.file_sha256(Path("/content/drive/artifact.pt")) == hashlib.sha256(payload).hexdigest()


def test_persistent_eio_identifies_the_failed_file(monkeypatch):
    opens = []

    def fail(path, *_args, **_kwargs):
        opens.append(path)
        raise OSError(errno.EIO, "Input/output error")

    path = Path("/content/drive/artifact.pt")
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(OSError) as error:
        runtime.file_sha256(path)
    assert error.value.errno == errno.EIO
    assert error.value.filename == str(path)
    assert opens == [path, path, path]
