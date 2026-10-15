"""Minimal WebHDFS client (create / read / list / mkdirs) built on requests."""
import requests

from . import config


class HDFS:
    def __init__(self, base_url: str = config.WEBHDFS_URL, user: str = config.HDFS_USER):
        self.base = base_url.rstrip("/")
        self.user = user
        self.s = requests.Session()

    def _url(self, path: str, op: str, **params) -> str:
        q = "&".join(f"{k}={v}" for k, v in {"op": op, "user.name": self.user, **params}.items())
        return f"{self.base}{path}?{q}"

    def mkdirs(self, path: str) -> None:
        r = self.s.put(self._url(path, "MKDIRS"), timeout=30)
        r.raise_for_status()

    def write(self, path: str, data: bytes, overwrite: bool = True) -> None:
        """Two-step WebHDFS create: NameNode answers 307 with the DataNode URL, then we PUT the bytes."""
        r = self.s.put(
            self._url(path, "CREATE", overwrite=str(overwrite).lower()),
            allow_redirects=False, timeout=30,
        )
        if r.status_code != 307:
            r.raise_for_status()
        r2 = self.s.put(r.headers["Location"], data=data, timeout=120)
        r2.raise_for_status()

    def read(self, path: str) -> bytes:
        r = self.s.get(self._url(path, "OPEN"), timeout=300)
        r.raise_for_status()
        return r.content

    def list(self, path: str) -> list[dict]:
        r = self.s.get(self._url(path, "LISTSTATUS"), timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()["FileStatuses"]["FileStatus"]

    def walk_files(self, path: str):
        """Yield full paths of every file under `path` (recursive)."""
        for st in self.list(path):
            child = f"{path.rstrip('/')}/{st['pathSuffix']}"
            if st["type"] == "DIRECTORY":
                yield from self.walk_files(child)
            else:
                yield child
