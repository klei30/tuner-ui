"""
Text Utilities
===============

Common text processing utilities used across the backend.
"""

import re


def strip_ansi_codes(text: str) -> str:
    """
    Strip ANSI escape codes from text.

    ANSI codes are used for terminal colors and formatting.
    This function removes them to get clean text.

    Args:
        text: Text that may contain ANSI escape codes

    Returns:
        Text with ANSI codes removed

    Example:
        >>> strip_ansi_codes("\x1B[31mRed text\x1B[0m")
        "Red text"
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length (default: 100)
        suffix: Suffix to add when truncated (default: "...")

    Returns:
        Truncated text with suffix if needed

    Example:
        >>> truncate_text("This is a long text", 10)
        "This is..."
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing dangerous characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem

    Example:
        >>> sanitize_filename("my/file<name>.txt")
        "my_file_name_.txt"
    """
    # Remove or replace dangerous characters
    dangerous_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(dangerous_chars, "_", filename)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")

    # Limit length to avoid filesystem issues
    max_length = 255
    if len(sanitized) > max_length:
        name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
        name = name[:max_length - len(ext) - 1]
        sanitized = f"{name}.{ext}" if ext else name

    return sanitized or "unnamed"
