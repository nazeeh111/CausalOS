"""Fetch the no-charge GSAP runtime required only to render the launch film.

GSAP is third-party, not covered by this project's MIT license. Review its
license at https://gsap.com/standard-license/ before use. Its original copyright
header is retained. The application itself does not use GSAP.
"""
from pathlib import Path
import hashlib
import urllib.request

URL = "https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"
SHA256 = "c174bfce53a729418d57a8ad8625e7247c793a22fef8e2851e3cfa3de9cd8280"
path = Path(__file__).resolve().parent / "composition" / "assets" / "gsap.vendor.js"
data = urllib.request.urlopen(URL, timeout=30).read()
if hashlib.sha256(data).hexdigest() != SHA256:
    raise SystemExit("GSAP download did not match the pinned checksum; nothing written.")
if path.exists() and path.read_bytes() != data:
    raise SystemExit("Existing vendor file differs; preserve it before replacing.")
path.write_bytes(data)
print(path)
