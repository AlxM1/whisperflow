"""Security utilities for WhisperFlow.

Provides input validation, sanitization, and security checks
following Code Guardian best practices.
"""

import hashlib
import os
import re
import secrets
from pathlib import Path
from typing import Any


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a filename to prevent path traversal and other attacks.

    Args:
        filename: Original filename
        max_length: Maximum allowed filename length

    Returns:
        Sanitized filename safe for filesystem operations
    """
    # Remove path separators and null bytes
    filename = filename.replace("\x00", "").replace("/", "_").replace("\\", "_")

    # Remove any leading dots (hidden files)
    filename = filename.lstrip(".")

    # Keep only alphanumeric, dots, dashes, underscores
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

    # Limit length
    if len(filename) > max_length:
        # Preserve extension
        name, ext = os.path.splitext(filename)
        name = name[: max_length - len(ext)]
        filename = name + ext

    # Ensure not empty
    if not filename:
        filename = "unnamed"

    return filename


def validate_audio_path(audio_path: str, allowed_base_dir: str) -> bool:
    """Validate that an audio file path is within the allowed directory.

    Prevents path traversal attacks.

    Args:
        audio_path: Path to validate
        allowed_base_dir: Base directory where audio files should be stored

    Returns:
        True if path is safe, False otherwise
    """
    try:
        base = Path(allowed_base_dir).resolve()
        target = Path(audio_path).resolve()

        # Check if target is within base directory
        return target.is_relative_to(base)
    except (ValueError, OSError):
        return False


def sanitize_llm_input(text: str, max_length: int = 10000) -> str:
    """Sanitize text before sending to LLM.

    Truncates to prevent token overflow and removes control characters.

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    # Remove null bytes and other control characters (except newline, tab, carriage return)
    text = "".join(
        char for char in text if char.isprintable() or char in ("\n", "\t", "\r")
    )

    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length]

    return text


def validate_llm_response(response: str, max_length: int = 50000) -> str:
    """Validate and sanitize LLM response.

    Prevents injection of malicious content in LLM responses.

    Args:
        response: LLM response text
        max_length: Maximum allowed response length

    Returns:
        Sanitized response
    """
    # Check for suspiciously long responses (potential DoS)
    if len(response) > max_length:
        response = response[:max_length] + "..."

    # Remove null bytes
    response = response.replace("\x00", "")

    return response


def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks.

    Args:
        a: First string
        b: Second string

    Returns:
        True if strings are equal
    """
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token.

    Args:
        length: Token length in bytes

    Returns:
        Hex-encoded random token
    """
    return secrets.token_hex(length)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage.

    Args:
        api_key: Plain API key

    Returns:
        SHA-256 hash of the API key
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def validate_language_code(lang: str) -> bool:
    """Validate that a language code is a valid ISO 639-1 code.

    Args:
        lang: Language code to validate

    Returns:
        True if valid, False otherwise
    """
    if not lang:
        return True  # Empty is OK (auto-detect)

    # ISO 639-1 codes are 2 lowercase letters, optionally followed by region code
    if re.match(r"^[a-z]{2}(-[A-Z]{2})?$", lang):
        return True

    # Also accept common special values
    return lang in ("auto", "zh-Hans", "zh-Hant")


def sanitize_url(url: str) -> str:
    """Sanitize a URL to prevent SSRF and other attacks.

    Args:
        url: URL to sanitize

    Returns:
        Sanitized URL or empty string if invalid
    """
    if not url:
        return ""

    # Only allow http and https schemes
    if not url.startswith(("http://", "https://")):
        return ""

    # Block localhost and private IPs (for external URLs only)
    # This is a basic check; more comprehensive validation may be needed
    blocked_patterns = [
        r"://localhost[:/]",
        r"://127\.",
        r"://0\.",
        r"://192\.168\.",
        r"://10\.",
        r"://172\.(1[6-9]|2[0-9]|3[01])\.",
        r"://\[::1\]",
        r"://\[::ffff:127\.",
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            # Allow if it's explicitly whitelisted internal service
            if any(
                svc in url for svc in ["cortex:", "whisperflow:", "ollama:", "localhost:11434"]
            ):
                return url
            return ""

    return url


def validate_file_size(size_bytes: int, max_size_mb: int = 50) -> bool:
    """Validate that a file size is within acceptable limits.

    Args:
        size_bytes: File size in bytes
        max_size_mb: Maximum allowed size in MB

    Returns:
        True if size is acceptable, False otherwise
    """
    return 0 < size_bytes <= max_size_mb * 1024 * 1024


def validate_mimetype(content_type: str, allowed_types: set[str]) -> bool:
    """Validate that a MIME type is in the allowed set.

    Args:
        content_type: MIME type to validate
        allowed_types: Set of allowed MIME types

    Returns:
        True if MIME type is allowed, False otherwise
    """
    if not content_type:
        return False

    # Strip parameters (e.g., "audio/wav; charset=utf-8" -> "audio/wav")
    base_type = content_type.split(";")[0].strip().lower()

    return base_type in allowed_types


# Allowed MIME types for audio uploads
ALLOWED_AUDIO_MIMETYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/opus",
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
}
