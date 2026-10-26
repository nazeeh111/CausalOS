"""Download IHDP replication 1 for local research; data are not redistributed.

The archived upstream repository does not declare a license. Check applicable
terms before using or redistributing. This script never downloads code to execute.
"""
from pathlib import Path
import argparse
import hashlib
import urllib.request

URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
SHA256 = "2bdc8b94fdf2f669b1d3d49325fd9359d88418566545003867841312a1653411"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise SystemExit("Destination exists; select a fresh path.")
    raw = urllib.request.urlopen(URL, timeout=30).read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit("Upstream bytes changed; checksum mismatch. Review the source.")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes(raw)
    print(f"Saved checksum-verified IHDP replication 1 to {args.destination}")


if __name__ == "__main__":
    main()
