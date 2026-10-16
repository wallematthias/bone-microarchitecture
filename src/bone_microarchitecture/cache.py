"""Final-output cache validation; distance fields are deliberately not cached."""

from hashlib import sha256
from pathlib import Path


def input_fingerprints(paths):
    """Hash input contents, including geometry headers, not just record IDs."""
    result = {}
    for path in paths:
        path = Path(path).resolve()
        with path.open("rb") as stream:
            fingerprint = sha256_stream(stream)
        if path.suffix.lower() in {".mhd", ".nhdr", ".hdr", ".img"}:
            # Header-only hashing misses edits to detached pixel data. Let the
            # image reader resolve payload names/LIST layouts, not our parser.
            import SimpleITK as sitk

            image = sitk.ReadImage(str(path))
            digest = sha256(fingerprint.encode())
            digest.update(
                repr(
                    (image.GetSpacing(), image.GetOrigin(), image.GetDirection())
                ).encode()
            )
            digest.update(sitk.GetArrayViewFromImage(image))
            fingerprint = digest.hexdigest()
        result[str(path)] = fingerprint
    return result


def sha256_stream(stream):
    digest = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def output_file_states(paths):
    """Detect missing, replaced, or partially rewritten final outputs cheaply."""
    states = {}
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        states[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return states


def cache_is_current(metadata):
    """Validate recorded inputs/outputs; metadata without a cache is not certified."""
    try:
        inputs = metadata["input_fingerprints"]
        outputs = metadata["output_file_states"]
        return (
            bool(inputs and outputs)
            and inputs == input_fingerprints(inputs)
            and outputs == output_file_states(outputs)
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False
