"""Bounded, no-symlink, atomic file operations for an explicitly trusted local root.

Directory-fd traversal limits symlink races on POSIX. It does not defend against
an attacker able to change the trusted root or execute code as the current user.
"""
from __future__ import annotations

import os
from pathlib import Path
import secrets
import stat


class UnsafePath(ValueError):
    """A path escapes the output root, traverses a symlink or would overwrite."""


def _location(path: str | Path, root: str | Path | None) -> tuple[Path, list[str]]:
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise UnsafePath("Parent-directory traversal is forbidden")
    p = Path(os.path.abspath(raw))
    home = Path(os.path.abspath(Path(root).expanduser() if root is not None else p.parent))
    try:
        parts = list(p.relative_to(home).parts)
    except ValueError as exc:
        raise UnsafePath("Path lies outside the allowed root") from exc
    if not parts or not all(part not in ("", ".", "..") for part in parts):
        raise UnsafePath("Expected a file path under the root")
    # Check each pre-existing root component; open(O_NOFOLLOW) protects again.
    for parent in (home, *home.parents):
        if parent.is_symlink():
            raise UnsafePath("Root or ancestor is a symlink")
    return home, parts


def _open_parent(path: str | Path, root: str | Path | None, create: bool) -> tuple[int, str]:
    home, parts = _location(path, root)
    if create:
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(home, flags)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd, parts[-1]
    except Exception:
        os.close(fd)
        raise


def safe_write_bytes(path: str | Path, content: bytes, *, root: str | Path | None = None,
                     overwrite: bool = False) -> None:
    fd, name = _open_parent(path, root, create=True)
    tmp = ".wslab-" + secrets.token_hex(16) + ".tmp"
    created = False
    try:
        try:
            existing = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (not overwrite or not stat.S_ISREG(existing.st_mode)):
            raise UnsafePath("Refusing to overwrite an existing file or symlink")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        tempfd = os.open(tmp, flags, 0o600, dir_fd=fd)
        created = True
        with os.fdopen(tempfd, "wb") as writer:
            writer.write(content)
            writer.flush()
            os.fsync(writer.fileno())
        if overwrite:
            os.replace(tmp, name, src_dir_fd=fd, dst_dir_fd=fd)
        else:
            # Atomic no-clobber even if another process creates the target meanwhile.
            try:
                os.link(tmp, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            except FileExistsError as exc:
                raise UnsafePath("Destination already exists") from exc
            os.unlink(tmp, dir_fd=fd)
        created = False
        os.fsync(fd)
    finally:
        if created:
            try:
                os.unlink(tmp, dir_fd=fd)
            except FileNotFoundError:
                pass
        os.close(fd)


def safe_read_bytes(path: str | Path, *, root: str | Path | None = None,
                    max_bytes: int = 64 * 1024 * 1024) -> bytes:
    fd, name = _open_parent(path, root, create=False)
    try:
        readfd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd)
        try:
            st = os.fstat(readfd)
            if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
                raise UnsafePath("Input must be a regular file within the size limit")
            with os.fdopen(readfd, "rb", closefd=False) as source:
                data = source.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise UnsafePath("Input is larger than permitted")
            return data
        finally:
            os.close(readfd)
    finally:
        os.close(fd)
