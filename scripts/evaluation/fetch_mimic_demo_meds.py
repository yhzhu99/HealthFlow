from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


FILES = [
    "LICENSE.txt",
    "README.txt",
    "SHA256.txt",
    "SHA256SUMS.txt",
    "data/train/0.parquet",
    "data/tuning/0.parquet",
    "data/held_out/0.parquet",
    "metadata/codes.parquet",
    "metadata/dataset.json",
    "metadata/subject_splits.parquet",
]

BASE_URL = "https://physionet.org/files/mimic-iv-demo-meds/0.0.1/"


def find_workspace_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "data" / "TJH.csv").exists():
            return parent
    raise FileNotFoundError("Could not locate workspace root containing data/TJH.csv")


WORKSPACE_ROOT = find_workspace_root()
DEFAULT_DESTINATION = WORKSPACE_ROOT / "data" / "mimic_iv_demo_meds"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_file(destination_root: Path, relative_path: str) -> None:
    target = destination_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BASE_URL + relative_path, target)


def load_expected_hashes(destination_root: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    checksum_path = destination_root / "SHA256SUMS.txt"
    if not checksum_path.exists():
        return expected
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative_path = line.split("  ", 1)
        expected[relative_path.strip()] = digest.strip()
    return expected


def verify_downloads(destination_root: Path) -> None:
    expected_hashes = load_expected_hashes(destination_root)
    for relative_path in FILES:
        target = destination_root / relative_path
        if not target.exists():
            raise FileNotFoundError(f"Missing downloaded file: {target}")
        expected_hash = expected_hashes.get(relative_path)
        if expected_hash and sha256_file(target) != expected_hash:
            raise ValueError(f"Checksum mismatch for {relative_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the open MIMIC-IV demo MEDS snapshot and verify checksums.")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    for relative_path in FILES:
        target = args.destination / relative_path
        if not target.exists():
            print(f"Downloading {relative_path}")
            download_file(args.destination, relative_path)
    verify_downloads(args.destination)
    print(f"Verified {len(FILES)} files in {args.destination}")


if __name__ == "__main__":
    main()
