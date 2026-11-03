"""Read members of a large zip on a public UCLA Box share via HTTP byte ranges, without downloading it.

  <file_id> may also be a path to a local .zip (e.g. a fully downloaded archive).
  python boxzip.py list    <file_id>                      summary of the archive's contents
  python boxzip.py extract <file_id> <out_dir> <suffix> [--only-existing-dirs]
                                                          extract members ending in <suffix> (never overwrites);
                                                          optionally only into folders that already exist locally
"""
import io, os, sys, time, zipfile
from collections import Counter
import requests

SHARE = "vxetzti0z0dv97yeh1jgxdprxxqucs2a"
UA = {"User-Agent": "Mozilla/5.0"}


class BoxRangeFile(io.RawIOBase):
    """Seekable read-only file over a Box shared-file download, using explicit `bytes=a-b` ranges."""

    def __init__(self, file_id, block=1 << 20):
        self.src = f"https://ucla.app.box.com/index.php?rm=box_download_shared_file&shared_name={SHARE}&file_id={file_id}"
        self.s = requests.Session(); self.s.headers.update(UA)
        self.block, self.cache, self.pos = block, {}, 0
        self._resolve()

    def _resolve(self):
        r = self.s.get(self.src, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url = r.url                                   # signed boxcloud URL (expires; re-resolved on error)
        self.size = int(r.headers["Content-Range"].split("/")[1])

    def _get(self, a, b):
        for attempt in range(6):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                if r.status_code == 206:
                    return r.content
                self._resolve()
            except requests.RequestException:
                time.sleep(2 ** attempt); self._resolve()
        raise IOError(f"range {a}-{b} failed")

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos)); out = bytearray()
        while n > 0:
            bi = self.pos // self.block
            if bi not in self.cache:
                if len(self.cache) > 64: self.cache.clear()
                a = bi * self.block
                self.cache[bi] = self._get(a, min(a + self.block, self.size) - 1)
            chunk = self.cache[bi][self.pos - bi * self.block:][:n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def main():
    cmd, fid = sys.argv[1], sys.argv[2]
    z = zipfile.ZipFile(fid if os.path.isfile(fid) else BoxRangeFile(fid))
    info = z.infolist()
    if cmd == "list":
        ext = Counter(os.path.splitext(i.filename)[1] or "<dir>" for i in info)
        tops = Counter("/".join(i.filename.split("/")[:2]) for i in info)
        print(f"entries={len(info)} by extension={dict(ext)}")
        print("top-level dirs:", dict(list(tops.items())[:8]))
        for suf in (".yaml", ".pcd"):
            sel = [i for i in info if i.filename.endswith(suf)]
            print(f"{suf}: n={len(sel)}  compressed={sum(i.compress_size for i in sel)/1e9:.2f} GB  e.g. {[i.filename for i in sel[:2]]}")
    elif cmd == "extract":
        out, suf = sys.argv[3], sys.argv[4]
        only_existing = "--only-existing-dirs" in sys.argv[5:]   # skip members whose folder isn't already on disk
        sel = [i for i in info if i.filename.endswith(suf)]
        if only_existing:
            n0 = len(sel)
            sel = [i for i in sel if os.path.isdir(os.path.dirname(os.path.join(out, i.filename)))]
            print(f"--only-existing-dirs: {len(sel)} of {n0} '{suf}' members have an existing local folder", flush=True)
        done = skipped = 0; t0 = time.time()
        for i in sorted(sel, key=lambda i: i.header_offset):        # sequential offsets => good cache reuse
            dst = os.path.realpath(os.path.join(out, i.filename))
            if not dst.startswith(os.path.realpath(out) + os.sep):          # refuse path traversal
                print(f"  REFUSED unsafe path: {i.filename}"); continue
            if os.path.exists(dst): skipped += 1; continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".part"
            with z.open(i) as src, open(tmp, "wb") as f: f.write(src.read())
            os.replace(tmp, dst); done += 1
            if done % 500 == 0: print(f"  {done}/{len(sel)} extracted ({time.time()-t0:.0f}s)", flush=True)
        print(f"extracted={done} skipped_existing={skipped} of {len(sel)} '{suf}' members into {out}")


if __name__ == "__main__":
    main()
