"""D2 (Bregar 2023, Zenodo record 8280736) data_set.zip 다운로드 + MD5 검증.

HTTP Range 로 고정 크기 조각을 받아 파일의 정확한 위치에 쓴다 (이어받기 시 중복 기록·덧붙이기 방지).
조각별 완료 기록은 <zip>.part.done 에 남겨 중단 후 다시 실행하면 남은 조각만 받는다.
검증(MD5) 통과 후에만 data_set.zip 으로 바꾼다.
"""
import hashlib
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "data_ext" / "D2_bregar" / "data_set.zip"
PART = DST.with_suffix(".zip.part")
DONE = DST.with_suffix(".zip.part.done")
URL = "https://zenodo.org/api/records/8280736/files/data_set.zip/content"
API = "https://zenodo.org/api/records/8280736"
CHUNK = 64 << 20
WORKERS = 4

meta = json.load(urllib.request.urlopen(API))["files"][0]
SIZE, MD5 = int(meta["size"]), meta["checksum"].split(":")[1]
print(f"size={SIZE:,} md5={MD5}", flush=True)

if not PART.exists() or PART.stat().st_size != SIZE:
    with open(PART, "wb") as f:
        f.truncate(SIZE)
done = set(int(x) for x in DONE.read_text().split()) if DONE.exists() else set()
lock = threading.Lock()
todo = [i for i in range((SIZE + CHUNK - 1) // CHUNK) if i not in done]
print(f"chunks total={(SIZE + CHUNK - 1) // CHUNK} todo={len(todo)}", flush=True)


def fetch(i):
    a = i * CHUNK
    b = min(a + CHUNK, SIZE) - 1
    for attempt in range(10):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206 or r.headers["Content-Range"] != f"bytes {a}-{b}/{SIZE}":
                    raise IOError(f"bad range response {r.status} {r.headers.get('Content-Range')}")
                buf = r.read()
            if len(buf) != b - a + 1:
                raise IOError(f"short read {len(buf)}")
            with lock:
                with open(PART, "r+b") as f:
                    f.seek(a)
                    f.write(buf)
                with open(DONE, "a") as f:
                    f.write(f"{i}\n")
            return i
        except Exception as e:  # noqa: BLE001
            wait = 15 * (attempt + 1)
            print(f"  chunk {i} attempt {attempt}: {e}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"chunk {i} failed")


t0 = time.time()
with ThreadPoolExecutor(WORKERS) as ex:
    for k, i in enumerate(ex.map(fetch, todo), 1):
        mb = k * CHUNK / 1e6
        print(f"  {k}/{len(todo)} chunks, {mb / (time.time() - t0):.1f} MB/s", flush=True)

h = hashlib.md5()
with open(PART, "rb") as f:
    for blk in iter(lambda: f.read(16 << 20), b""):
        h.update(blk)
ok = h.hexdigest() == MD5
print(f"md5 {h.hexdigest()} expected {MD5} -> {'OK' if ok else 'MISMATCH'}", flush=True)
if not ok:
    sys.exit(1)
if DST.exists():
    DST.unlink()
PART.rename(DST)
DONE.unlink()
print("saved", DST, flush=True)
