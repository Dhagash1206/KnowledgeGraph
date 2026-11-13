"""Setup script for isolated Stanford CoreNLP and OpenJDK 17 runtime dependencies.

Downloads pinned official binaries into .runtime/, verifies each against a pinned
checksum before use, and records SHA-256 hashes in a runtime manifest.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent.parent / ".runtime"
STANFORD_DIR = RUNTIME_ROOT / "stanford_classical"
LIB_DIR = STANFORD_DIR / "lib"
JDK_DIR = RUNTIME_ROOT / "jdk17"

JDK_RELEASE_URL = "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9"
# SHA-256 values published by Adoptium next to each release asset (*.sha256.txt).
JDK_BUILDS = {
    ("mac", "aarch64"): ("OpenJDK17U-jdk_aarch64_mac_hotspot_17.0.11_9.tar.gz",
                         "09a162c58dd801f7cfacd87e99703ed11fb439adc71cfa14ceb2d3194eaca01c"),
    ("mac", "x64"): ("OpenJDK17U-jdk_x64_mac_hotspot_17.0.11_9.tar.gz",
                     "f8b96724618f4df557c47f11048d1084e98ed3eb87f0dbd5b84f768a80c3348e"),
    ("linux", "aarch64"): ("OpenJDK17U-jdk_aarch64_linux_hotspot_17.0.11_9.tar.gz",
                           "a900acf3ae56b000afc35468a083b6d6fd695abec87a8abdb02743d5c72f6d6d"),
    ("linux", "x64"): ("OpenJDK17U-jdk_x64_linux_hotspot_17.0.11_9.tar.gz",
                       "aa7fb6bb342319d227a838af5c363bfa1b4a670c209372f9e6585bd79da6220c"),
    ("windows", "x64"): ("OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.zip",
                         "fdd6664d4131370398fbc8bfbb7b46dbfec4a22a090a511fe5c379dae188c390"),
}

# "sha1" values are the checksums Maven Central publishes for each jar (*.jar.sha1).
CORE_NLP_ARTIFACTS = [
    {
        "name": "stanford-corenlp-4.5.8.jar",
        "sha1": "9de8fb15e2f408c2316f11aeca601f1a5cd8ba56",
        "url": "https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/4.5.8/stanford-corenlp-4.5.8.jar",
        "target_dir": STANFORD_DIR,
        "license": "GPLv3",
    },
    {
        "name": "stanford-corenlp-4.5.8-models.jar",
        "sha1": "4daba7d4efb11acad6649dd16530821db0ea8283",
        "url": "https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/4.5.8/stanford-corenlp-4.5.8-models.jar",
        "target_dir": STANFORD_DIR,
        "license": "GPLv3",
    },
    {
        "name": "joda-time-2.13.0.jar",
        "sha1": "a4d4db9e91878d964a4544c8ea6d263dae2f37bc",
        "url": "https://repo1.maven.org/maven2/joda-time/joda-time/2.13.0/joda-time-2.13.0.jar",
        "target_dir": LIB_DIR,
        "license": "Apache-2.0",
    },
    {
        "name": "xom-1.3.9.jar",
        "sha1": "c1012b95027799beabd123a3f651aea19d13edc5",
        "url": "https://repo1.maven.org/maven2/xom/xom/1.3.9/xom-1.3.9.jar",
        "target_dir": LIB_DIR,
        "license": "LGPL-2.1",
    },
    {
        "name": "ejml-core-0.39.jar",
        "sha1": "26cb3a9a456fa8bb4c292b6031cfdcfebf64255c",
        "url": "https://repo1.maven.org/maven2/org/ejml/ejml-core/0.39/ejml-core-0.39.jar",
        "target_dir": LIB_DIR,
        "license": "Apache-2.0",
    },
    {
        "name": "ejml-ddense-0.39.jar",
        "sha1": "01ea52701a46f9a9005f4b24a9d41c4813c36412",
        "url": "https://repo1.maven.org/maven2/org/ejml/ejml-ddense/0.39/ejml-ddense-0.39.jar",
        "target_dir": LIB_DIR,
        "license": "Apache-2.0",
    },
    {
        "name": "ejml-simple-0.39.jar",
        "sha1": "e355bf277f877ba8eefd3947b9c246fdf921d21b",
        "url": "https://repo1.maven.org/maven2/org/ejml/ejml-simple/0.39/ejml-simple-0.39.jar",
        "target_dir": LIB_DIR,
        "license": "Apache-2.0",
    },
    {
        "name": "slf4j-api-1.7.36.jar",
        "sha1": "6c62681a2f655b49963a5983b8b0950a6120ae14",
        "url": "https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.36/slf4j-api-1.7.36.jar",
        "target_dir": LIB_DIR,
        "license": "MIT",
    },
    {
        "name": "slf4j-simple-1.7.36.jar",
        "sha1": "a41f9cfe6faafb2eb83a1c7dd2d0dfd844e2a936",
        "url": "https://repo1.maven.org/maven2/org/slf4j/slf4j-simple/1.7.36/slf4j-simple-1.7.36.jar",
        "target_dir": LIB_DIR,
        "license": "MIT",
    },
    {
        "name": "jakarta.json-1.1.6.jar",
        "sha1": "a39aba0ef0a70c3f54a44a2c539a4f11c9bb07e8",
        "url": "https://repo1.maven.org/maven2/org/glassfish/jakarta.json/1.1.6/jakarta.json-1.1.6.jar",
        "target_dir": LIB_DIR,
        "license": "EPL-2.0 / GPLv2+CE",
    },
    {
        "name": "protobuf-java-3.25.5.jar",
        "sha1": "5ae5c9ec39930ae9b5a61b32b93288818ec05ec1",
        "url": "https://repo1.maven.org/maven2/com/google/protobuf/protobuf-java/3.25.5/protobuf-java-3.25.5.jar",
        "target_dir": LIB_DIR,
        "license": "BSD-3-Clause",
    },
]


def compute_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, dest: Path, algorithm: str, expected_hash: str) -> None:
    """Download url to dest unless dest already matches expected_hash.

    The file is written to a temporary path and only moved into place after its
    length and checksum are verified, so an interrupted or tampered download is
    never left behind as if it were complete.
    """
    if dest.exists() and compute_hash(dest, algorithm) == expected_hash:
        print(f"Already verified: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(dest.name + ".part")
    print(f"Downloading {dest.name} from {url}...")
    request = urllib.request.Request(url, headers={"User-Agent": "Doc2Graph-Classical-Setup/1.0"})
    try:
        with urllib.request.urlopen(request) as response, open(tmp_dest, "wb") as out:
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded_bytes = 0
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                downloaded_bytes += len(chunk)
                if total_bytes:
                    percent = downloaded_bytes / total_bytes * 100
                    sys.stdout.write(
                        f"\r  {downloaded_bytes / 1e6:.1f}MB / {total_bytes / 1e6:.1f}MB ({percent:.1f}%)"
                    )
                    sys.stdout.flush()
        print()
        if total_bytes and downloaded_bytes != total_bytes:
            raise RuntimeError(
                f"Incomplete download for {dest.name}: {downloaded_bytes} of {total_bytes} bytes."
            )
        actual_hash = compute_hash(tmp_dest, algorithm)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}: expected {algorithm} {expected_hash}, got {actual_hash}."
            )
    except BaseException:
        tmp_dest.unlink(missing_ok=True)
        raise
    tmp_dest.replace(dest)
    print(f"Saved and verified: {dest} ({dest.stat().st_size} bytes)")


def detect_platform() -> tuple[str, str]:
    """Return the Adoptium (os, arch) names for this machine."""
    os_names = {"darwin": "mac", "linux": "linux", "win32": "windows"}
    arch_names = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x64", "amd64": "x64"}
    os_name = os_names.get(sys.platform)
    arch = arch_names.get(platform.machine().lower())
    if (os_name, arch) not in JDK_BUILDS:
        raise RuntimeError(
            f"No pinned OpenJDK 17 build for platform {sys.platform}/{platform.machine()}. "
            f"Supported: {sorted(JDK_BUILDS)}."
        )
    return os_name, arch


def java_binary(jdk_home: Path) -> Path:
    return jdk_home / "bin" / ("java.exe" if sys.platform == "win32" else "java")


def _extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(destination)
        return
    with tarfile.open(archive, "r:gz") as tar_file:
        # The "data" filter rejects absolute paths and links escaping the destination.
        tar_file.extractall(destination, filter="data")


def _find_jdk_home(extract_root: Path) -> Path | None:
    """macOS archives nest the JDK under Contents/Home; others put bin/ at the top."""
    for top_dir in extract_root.iterdir():
        for candidate in (top_dir, top_dir / "Contents" / "Home"):
            if java_binary(candidate).exists():
                return candidate
    return None


def setup_jdk() -> Path:
    java_path = java_binary(JDK_DIR)
    if java_path.exists():
        print(f"Java already exists at {java_path}")
        return java_path

    archive_name, archive_sha256 = JDK_BUILDS[detect_platform()]
    archive_path = RUNTIME_ROOT / archive_name
    download_verified(f"{JDK_RELEASE_URL}/{archive_name}", archive_path, "sha256", archive_sha256)

    print("Extracting OpenJDK 17...")
    extract_root = RUNTIME_ROOT / "jdk17_extract"
    shutil.rmtree(extract_root, ignore_errors=True)
    extract_root.mkdir(parents=True)
    try:
        _extract_archive(archive_path, extract_root)
        jdk_home = _find_jdk_home(extract_root)
        if jdk_home is None:
            raise RuntimeError(f"No java binary found inside {archive_name}.")
        shutil.rmtree(JDK_DIR, ignore_errors=True)
        jdk_home.rename(JDK_DIR)
    finally:
        shutil.rmtree(extract_root, ignore_errors=True)

    if not java_path.exists():
        raise RuntimeError(f"Failed to find java at {java_path} after extraction.")
    print(f"OpenJDK 17 ready at {java_path}")
    return java_path


def setup_stanford_dependencies() -> dict:
    STANFORD_DIR.mkdir(parents=True, exist_ok=True)
    LIB_DIR.mkdir(parents=True, exist_ok=True)

    manifest_entries = {}
    for artifact in CORE_NLP_ARTIFACTS:
        target = artifact["target_dir"] / artifact["name"]
        download_verified(artifact["url"], target, "sha1", artifact["sha1"])
        sha256 = compute_hash(target, "sha256")
        manifest_entries[artifact["name"]] = {
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "sha1": artifact["sha1"],
            "sha256": sha256,
            "url": artifact["url"],
            "license": artifact["license"],
        }
        print(f"Verified {artifact['name']}: {target.stat().st_size} bytes, sha256={sha256[:12]}...")

    manifest_path = STANFORD_DIR / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")
    return manifest_entries


def check_java_version(java_path: Path) -> None:
    result = subprocess.run([str(java_path), "-version"], capture_output=True, text=True, check=True)
    print("Java version check output:\n", (result.stderr or result.stdout).strip())


def main() -> None:
    print("=== Setting up Classical Stanford NLP & JDK 17 Runtime ===")
    check_java_version(setup_jdk())
    setup_stanford_dependencies()
    print("=== Setup Complete ===")


if __name__ == "__main__":
    main()
