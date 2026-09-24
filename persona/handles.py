"""Validate user-supplied X handles before using them in queries or file names."""
import re


def normalized_handle(value: str) -> str:
    handle = value.strip().removeprefix('@')
    if not re.fullmatch(r'[A-Za-z0-9_]{1,15}', handle):
        raise ValueError('Use an X handle of 1–15 letters, digits or underscores; omit the profile URL.')
    # These names cannot be used as file stems on Windows, including CON.jsonl.
    if handle.casefold() in {'con', 'prn', 'aux', 'nul'} or re.fullmatch(
        r'(?:com|lpt)[1-9]', handle, re.IGNORECASE
    ):
        raise ValueError('This handle is a reserved Windows file name and cannot be exported by Studio.')
    return handle
