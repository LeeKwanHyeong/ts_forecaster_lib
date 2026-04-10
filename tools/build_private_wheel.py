from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "dist"
PRIVATE_DIST_DIR = DIST_DIR / "private"
PUBLIC_SOURCE_ROOTS = (
    "modeling_module/api/",
)
PUBLIC_SOURCE_FILES = {
    "modeling_module/__init__.py",
}


def is_public_source(rel_path: str) -> bool:
    rel_path = rel_path.replace("\\", "/")
    if rel_path in PUBLIC_SOURCE_FILES:
        return True
    return any(rel_path.startswith(prefix) for prefix in PUBLIC_SOURCE_ROOTS)


def iter_python_sources(root: Path):
    for path in root.rglob("*.py"):
        rel_path = path.relative_to(root).as_posix()
        if ".dist-info/" in rel_path or ".egg-info/" in rel_path:
            continue
        yield path, rel_path


def convert_internal_sources_to_sourceless(root: Path) -> list[str]:
    converted: list[str] = []

    for source_path, rel_path in iter_python_sources(root):
        if is_public_source(rel_path):
            continue

        pyc_path = source_path.with_suffix(".pyc")
        py_compile.compile(str(source_path), cfile=str(pyc_path), doraise=True)
        source_path.unlink()
        converted.append(rel_path)

    return converted


def assert_only_public_python_sources(root: Path) -> None:
    unexpected = [
        rel_path
        for _path, rel_path in iter_python_sources(root)
        if not is_public_source(rel_path)
    ]
    if unexpected:
        joined = ", ".join(sorted(unexpected)[:10])
        raise RuntimeError(f"Unexpected internal source files remained in private wheel layout: {joined}")


def build_public_wheel(*, no_isolation: bool) -> Path:
    cmd = [sys.executable, "-m", "build", "--wheel"]
    if no_isolation:
        cmd.append("--no-isolation")

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    wheels = sorted(
        p for p in DIST_DIR.glob("*.whl")
        if "-private" not in p.name and "private" not in p.parent.as_posix()
    )
    if not wheels:
        raise FileNotFoundError("No public wheel was produced in dist/.")
    return wheels[-1]


def unpack_wheel(wheel_path: Path, dest_dir: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "wheel", "unpack", str(wheel_path), "--dest", str(dest_dir)],
        check=True,
    )

    unpacked = [p for p in dest_dir.iterdir() if p.is_dir()]
    if len(unpacked) != 1:
        raise RuntimeError(f"Expected exactly one unpacked wheel directory, found {len(unpacked)}")
    return unpacked[0]


def pack_private_wheel(unpacked_root: Path, *, dest_dir: Path, build_number: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "wheel",
            "pack",
            str(unpacked_root),
            "--dest-dir",
            str(dest_dir),
            "--build-number",
            build_number,
        ],
        check=True,
    )

    wheels = sorted(dest_dir.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError("Private wheel pack step did not produce a wheel.")
    return wheels[-1]


def build_private_wheel(
    *,
    wheel_path: Path | None = None,
    no_isolation: bool = True,
    build_number: str = "private1",
    dest_dir: Path | None = None,
) -> Path:
    source_wheel = wheel_path or build_public_wheel(no_isolation=no_isolation)
    output_dir = dest_dir or PRIVATE_DIST_DIR

    with tempfile.TemporaryDirectory(prefix="private-wheel-") as tmpdir:
        unpack_root = unpack_wheel(source_wheel, Path(tmpdir))
        convert_internal_sources_to_sourceless(unpack_root)
        assert_only_public_python_sources(unpack_root)
        private_wheel = pack_private_wheel(
            unpack_root,
            dest_dir=output_dir,
            build_number=build_number,
        )

    return private_wheel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a private wheel that keeps only public API sources as .py and converts internal modules to .pyc."
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="Existing public wheel to transform. If omitted, a public wheel is built first.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=PRIVATE_DIST_DIR,
        help="Destination directory for the private wheel.",
    )
    parser.add_argument(
        "--build-number",
        type=str,
        default="private1",
        help="Wheel build tag used for the private artifact filename.",
    )
    parser.add_argument(
        "--with-isolation",
        action="store_true",
        help="Use isolated PEP 517 build when creating the source wheel.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    private_wheel = build_private_wheel(
        wheel_path=args.wheel,
        no_isolation=not args.with_isolation,
        build_number=args.build_number,
        dest_dir=args.dest_dir,
    )
    print(private_wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

