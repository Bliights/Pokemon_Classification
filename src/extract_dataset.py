import json
import logging
import re
import warnings
from io import BytesIO
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ------------------- LOGGING CONFIGURATION --------------------------
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[92m",  # Green
        logging.INFO: "\033[96m",  # Cyan
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[91;1m",  # Bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


log_format = "[%(levelname)s] : %(message)s"

logging.basicConfig(level=logging.INFO, format=log_format)

for handler in logging.getLogger().handlers:
    handler.setFormatter(ColorFormatter(log_format))

logging.getLogger("httpx").setLevel(logging.WARNING)

# ----------------------------------------------
#           Pillow safety settings
# ----------------------------------------------

Image.MAX_IMAGE_PIXELS = 40_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "image/*,*/*;q=0.8",
}


# ----------------------------------------------
#                   Download
# ----------------------------------------------
def clean_url(u: str) -> str:
    """
    Normalize a URL-like string extracted from a dataset.

    Parameters
    ----------
    u : str
        Raw URL string.

    Returns
    -------
    str
        Cleaned URL string.
    """
    return u.strip().strip("'").strip('"').rstrip(",")


def is_image_content_type(resp: requests.Response) -> bool:
    """
    Check whether the HTTP response advertises an image MIME type.

    Notes
    -----
    This is a *hint*, not a proof. Servers can lie or use generic MIME types.
    The final authority in this pipeline is Pillow verification.

    Parameters
    ----------
    resp : requests.Response
        HTTP response object.

    Returns
    -------
    bool
        True if Content-Type starts with "image/", else False.
    """
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return ctype.startswith("image/")


def verify_image_bytes_with_pillow(data: bytes) -> bool:
    """
    Verify that a bytes buffer looks like a valid image.

    This uses Pillow's `verify()` to validate that the header/structure is
    consistent. It does not decode the full image into an array.

    Parameters
    ----------
    data : bytes
        Initial bytes (typically the first few hundred KB of the file).

    Returns
    -------
    bool
        True if Pillow can identify and verify the image structure, else False.
    """
    try:
        img = Image.open(BytesIO(data))
        img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def stream_to_file_with_limit(
    resp: requests.Response,
    tmp_path: Path,
    head: bytes,
    max_bytes: int,
    chunk_size: int = 64 * 1024,
) -> int | None:
    """
    Stream an HTTP response to disk with a strict size limit.

    Parameters
    ----------
    resp : requests.Response
        Streaming HTTP response.
    tmp_path : pathlib.Path
        Temporary file path to write to.
    head : bytes
        Initial bytes already read from the response stream.
    max_bytes : int
        Maximum allowed file size (in bytes).
    chunk_size : int, default=64*1024
        Chunk size used for streaming.

    Returns
    -------
    int | None
        Total number of bytes written if successful, otherwise None.
    """
    written = len(head)
    if written > max_bytes:
        return None

    with open(tmp_path, "wb") as f:
        f.write(head)
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            written += len(chunk)
            if written > max_bytes:
                return None
            f.write(chunk)

    return written


def verify_image_file_with_pillow(path: Path) -> bool:
    """
    Verify that a fully downloaded file is a valid image.

    Parameters
    ----------
    path : pathlib.Path
        Path to the downloaded file.

    Returns
    -------
    bool
        True if Pillow can verify the file, else False.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def safe_unlink(path: Path) -> None:
    """
    Remove a file if it exists, without raising an error if it does not.

    Parameters
    ----------
    path : pathlib.Path
        File path to remove.
    """
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def download_strict_image(
    url: str,
    out_path: str | Path,
    max_mb: int = 15,
    timeout: int = 30,
    sniff_kb: int = 512,
) -> Path | None:
    """Download a URL only if it is verified to be an image.

    This function applies multiple checks:
        1) HTTP status code must be OK.
        2) Content-Type must start with "image/" (cheap pre-filter).
        3) The first `sniff_kb` KB must be identifiable/valid by Pillow.
        4) The file is streamed to disk with a strict `max_mb` size limit.
        5) The final file is re-verified by Pillow.

    The download is written to a temporary ".part" file first, then atomically
    moved to `out_path` on success.

    Parameters
    ----------
    url : str
        Image URL to download.
    out_path : str | pathlib.Path
        Destination path for the downloaded file.
    max_mb : int, default=15
        Maximum allowed file size in megabytes.
    timeout : int, default=30
        Network timeout in seconds.
    sniff_kb : int, default=512
        Number of kilobytes to read initially for early Pillow verification.

    Returns
    -------
    pathlib.Path | None
        Path to the downloaded file if successful, else None.
    """
    url = clean_url(url)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    max_bytes = max_mb * 1024 * 1024
    sniff_bytes = sniff_kb * 1024

    try:
        with requests.get(
            url,
            headers=DEFAULT_HEADERS,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        ) as r:
            r.raise_for_status()

            # 1) Quick header pre-filter
            if not is_image_content_type(r):
                safe_unlink(tmp)
                return None

            # 2) Early sniff + Pillow verify on the first bytes
            head = r.raw.read(sniff_bytes)
            if not head or not verify_image_bytes_with_pillow(head):
                safe_unlink(tmp)
                return None

            # 3) Stream the rest with a strict size limit
            written = stream_to_file_with_limit(r, tmp, head, max_bytes=max_bytes)
            if written is None:
                safe_unlink(tmp)
                return None

        # 4) Re-verify the fully downloaded file
        if not verify_image_file_with_pillow(tmp):
            safe_unlink(tmp)
            return None

        tmp.replace(out_path)
        return out_path

    except Exception:
        safe_unlink(tmp)
        return None


# ----------------------------------------------
#           Create final file name
# ----------------------------------------------


def parse_tags(tags_field: str | dict) -> dict:
    """
    Parse the `tags` field from a dataset row.

    Parameters
    ----------
    tags_field : str | dict
        JSON string or a parsed dictionary.

    Returns
    -------
    dict
        Parsed tags dictionary (empty dict on empty input).
    """
    return tags_field if isinstance(tags_field, dict) else json.loads(tags_field or "{}")


def slug(s: str) -> str:
    """
    Convert an arbitrary label string into a filesystem-safe token.
        - Lowercases
        - Replaces whitespace with underscores
        - Removes non [a-z0-9_-] characters

    Parameters
    ----------
    s : str
        Input label.

    Returns
    -------
    str
        Slugified token, or "unknown" if empty after cleaning.
    """
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "unknown"


def pokemon_names(row: dict) -> list[str]:
    """
    Extract a stable list of Pokémon names from a dataset row.

    Parameters
    ----------
    row : dict[str, Any]
        One dataset sample.

    Returns
    -------
    list[str]
        Sorted, unique Pokémon name tokens.
    """
    tags = parse_tags(row.get("tags", "{}"))
    names = tags.get("Pokémon") or tags.get("Pokemon") or []
    names = [slug(x) for x in names if x]
    return sorted(set(names))


def build_stem(download_index: int, names: list[str]) -> str:
    """
    Build a filename stem from an index and a list of Pokémon names.

    The stem is deterministic:
        - A fixed-width increasing index prefix (e.g., 000123__)
        - Then Pokémon names joined by "__"
        - Fallback to "no_pokemon" if the list is empty.

    Parameters
    ----------
    download_index : int
        Increasing download counter.
    names : list[str]
        Pokémon name tokens.

    Returns
    -------
    str
        Filename stem without extension.
    """
    prefix = f"{download_index:06d}__"
    base = "__".join(names) if names else "no_pokemon"
    return prefix + base


# ----------------------------------------------
#                   Start of script
# ----------------------------------------------


def download_dataset() -> None:
    """
    Download images from the Hugging Face dataset and save them as JPG.

    Pipeline:
    - Loads the dataset split.
    - For each row, builds a deterministic filename from Pokémon tags.
    - Downloads the URL only if it is verified to be an image.
    - Converts the image to RGB JPEG (uniform format for downstream processing).
    - Deletes the temporary downloaded file.
    """
    # Load dataset (Hugging Face)
    dataset = load_dataset("Kev0208/PokeFA-pokemon-fanart-captioned", trust_remote_code=False)
    dataset = dataset["train"]
    data_dir = Path(__file__).resolve().parents[1] / "data" / "fanart-dataset"
    data_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    total = len(dataset)
    tmp_path: Path | None = None

    try:
        with tqdm(
            range(total),
            desc="Downloading dataset",
            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
            colour="green",
        ) as pbar:
            for i in pbar:
                row = dataset[i]
                url = row["source_url"]
                names = pokemon_names(row)
                stem = build_stem(ok, names)

                # 1) Strict download into a temporary file
                tmp_path = data_dir / f"{stem}.bin"
                saved = download_strict_image(url, tmp_path, max_mb=15, timeout=30, sniff_kb=512)
                if saved is None:
                    skipped += 1
                    pbar.set_postfix_str(f"ok={ok} skipped={skipped} (Skip at download)")
                    logger.debug("Skip (not an image / too big / failed): %s", url)
                    tmp_path = None
                    continue

                # 2) Convert to JPG
                final_path = data_dir / f"{stem}.jpg"
                try:
                    with Image.open(saved) as im:
                        im = im.convert("RGB")
                        im.save(final_path, "JPEG", quality=95, optimize=True)
                except Exception as e:
                    skipped += 1
                    pbar.set_postfix_str(f"ok={ok} skipped={skipped} (Skip at convert)")
                    logger.debug("Skip (convert failed): %s | %s", url, e)
                    continue
                finally:
                    saved.unlink(missing_ok=True)
                    tmp_path = None

                ok += 1
                pbar.set_postfix_str(f"ok={ok} skipped={skipped} last={final_path.name}")

            logger.info(
                "Dataset download in %s finished ! (Saved=%d | Skipped=%d)",
                data_dir,
                ok,
                skipped,
            )
    except KeyboardInterrupt:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
            part = tmp_path.with_suffix(tmp_path.suffix + ".part")
            part.unlink(missing_ok=True)
        logger.info(
            "Dataset download stopped. OutDir=%s | Saved=%d | Skipped=%d",
            data_dir,
            ok,
            skipped,
        )


if __name__ == "__main__":
    download_dataset()
