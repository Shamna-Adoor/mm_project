"""Resolve external video sources for playback and offline analysis."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable, TypedDict
from urllib.parse import urlparse

from analyzer._logging import get_logger

log = get_logger(__name__)

ProgressCallback = Callable[[int, str], None]

DIRECT_MEDIA_EXTS = {
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".ogg", ".ogv", ".m3u8", ".mpd",
}

MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".ogg": "video/ogg",
    ".ogv": "video/ogg",
    ".m3u8": "application/vnd.apple.mpegurl",
    ".mpd": "application/dash+xml",
}


class SourceInfo(TypedDict, total=False):
    job_id: str
    source_url: str
    webpage_url: str
    playback_url: str
    title: str
    extractor: str
    source_id: str
    duration: float | None
    ext: str
    protocol: str
    mime_type: str
    is_live: bool
    is_stream: bool
    can_analyze: bool


def normalize_source_url(url: str) -> str:
    """Validate and normalize a remote HTTP(S) video URL."""
    clean = url.strip()
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Source must be a valid http(s) URL.")
    return clean


def resolve_source(url: str) -> SourceInfo:
    """Resolve a URL into playback metadata.

    The resolver first handles direct media URLs, then falls back to yt-dlp for
    YouTube, Twitch, and the many other extractors yt-dlp supports.
    """
    url = normalize_source_url(url)
    direct = _direct_media_info(url)
    if direct:
        return direct

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required for YouTube/Twitch/external URLs.") from exc

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        # Prefer one browser-playable URL. The download step uses a richer format.
        "format": "best[vcodec!=none][acodec!=none]/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if "entries" in info:
        entries = [e for e in info.get("entries") or [] if e]
        if not entries:
            raise RuntimeError("No playable entries found in playlist.")
        info = entries[0]

    playback_url = _pick_playback_url(info) or url
    duration = _float_or_none(info.get("duration"))
    is_live = bool(info.get("is_live") or info.get("live_status") == "is_live")
    protocol = str(info.get("protocol") or "")
    ext = str(info.get("ext") or Path(urlparse(playback_url).path).suffix.lstrip(".") or "mp4")
    is_stream = _is_stream_protocol(protocol, playback_url)
    extractor = str(info.get("extractor_key") or info.get("extractor") or "external")
    title = str(info.get("title") or info.get("fulltitle") or urlparse(url).netloc)
    source_id = str(info.get("id") or "")

    resolved: SourceInfo = {
        "job_id": make_job_id(url, extractor=extractor, source_id=source_id, title=title),
        "source_url": url,
        "webpage_url": str(info.get("webpage_url") or url),
        "playback_url": playback_url,
        "title": title,
        "extractor": extractor,
        "source_id": source_id,
        "duration": duration,
        "ext": ext,
        "protocol": protocol,
        "mime_type": _mime_for(playback_url, ext),
        "is_live": is_live,
        "is_stream": is_stream,
        "can_analyze": not is_live,
    }
    log.info(
        "Resolved source %s via %s (live=%s, stream=%s, duration=%s)",
        resolved["job_id"], extractor, is_live, is_stream, duration,
    )
    return resolved


def download_source(
    url: str,
    output_dir: str | Path,
    job_id: str,
    *,
    progress: ProgressCallback | None = None,
    max_height: int = 1080,
) -> Path:
    """Download a finite remote video and return the local media path."""
    url = normalize_source_url(url)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required to download external video URLs.") from exc

    def hook(data: dict) -> None:
        if not progress or data.get("status") != "downloading":
            return
        total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
        done = data.get("downloaded_bytes") or 0
        if total:
            pct = max(0, min(100, int(done / total * 100)))
            progress(min(40, max(1, int(pct * 0.4))), f"Downloading source... {pct}%")
        else:
            progress(5, "Downloading source...")

    out_template = str(output_dir / f"{job_id}.%(ext)s")
    ydl_opts = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
        "progress_hooks": [hook],
    }

    if progress:
        progress(1, "Starting source download...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    candidates = sorted(
        output_dir.glob(f"{job_id}.*"),
        key=lambda p: (p.suffix.lower() != ".mp4", p.stat().st_mtime),
        reverse=True,
    )
    media = [p for p in candidates if p.suffix.lower() in DIRECT_MEDIA_EXTS - {".m3u8", ".mpd"}]
    if not media:
        raise FileNotFoundError(f"Download completed but no media file was found for {job_id}.")

    log.info("Downloaded source %s -> %s", job_id, media[0])
    return media[0].resolve()


def make_job_id(
    url: str,
    *,
    extractor: str | None = None,
    source_id: str | None = None,
    title: str | None = None,
) -> str:
    """Create a stable, filesystem-safe job id for a source URL."""
    extractor_l = (extractor or "").lower()
    if source_id and "youtube" in extractor_l:
        return f"yt_{_slug(source_id, max_len=32)}"
    if source_id and "twitch" in extractor_l:
        return f"twitch_{_slug(source_id, max_len=40)}"

    parsed = urlparse(url)
    base = source_id or title or Path(parsed.path).stem or parsed.netloc
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{_slug(base, max_len=48)}_{digest}"


def _direct_media_info(url: str) -> SourceInfo | None:
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext not in DIRECT_MEDIA_EXTS:
        return None

    is_stream = ext in {".m3u8", ".mpd"}
    title = Path(parsed.path).name or parsed.netloc
    return {
        "job_id": make_job_id(url, title=title),
        "source_url": url,
        "webpage_url": url,
        "playback_url": url,
        "title": title,
        "extractor": "direct",
        "source_id": "",
        "duration": None,
        "ext": ext.lstrip("."),
        "protocol": "direct",
        "mime_type": _mime_for(url, ext.lstrip(".")),
        "is_live": is_stream,
        "is_stream": is_stream,
        "can_analyze": not is_stream,
    }


def _pick_playback_url(info: dict) -> str | None:
    if isinstance(info.get("url"), str):
        return info["url"]

    formats = info.get("formats") or []
    playable = [
        f for f in formats
        if f.get("url") and f.get("vcodec") != "none" and f.get("acodec") != "none"
    ]
    if playable:
        playable.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
        return str(playable[0]["url"])

    streamable = [f for f in formats if f.get("url") and _is_stream_protocol(str(f.get("protocol") or ""), str(f.get("url")))]
    if streamable:
        return str(streamable[0]["url"])
    return None


def _is_stream_protocol(protocol: str, url: str) -> bool:
    protocol_l = protocol.lower()
    path_l = urlparse(url).path.lower()
    return "m3u8" in protocol_l or "dash" in protocol_l or path_l.endswith((".m3u8", ".mpd"))


def _mime_for(url: str, ext: str) -> str:
    ext = ext.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    if ext in MIME_BY_EXT:
        return MIME_BY_EXT[ext]
    path_ext = Path(urlparse(url).path).suffix.lower()
    return MIME_BY_EXT.get(path_ext, "video/mp4")


def _float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str, *, max_len: int = 64) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    clean = re.sub(r"_+", "_", clean)
    return (clean or "source")[:max_len]
