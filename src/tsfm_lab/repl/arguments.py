"""One quoting policy for native slash-command dispatch and path completion."""

import os
import shlex


def split_arguments(text, *, windows=None):
    windows = os.name == "nt" if windows is None else windows
    if not windows:
        return shlex.split(text)
    parser = shlex.shlex(text, posix=True)
    parser.whitespace_split = True
    parser.commenters = ""
    # These are application commands, not a shell. A Windows backslash always
    # belongs to the path, including immediately before a closing quote.
    parser.escape = ""
    return list(parser)


def quote_argument(text, *, windows=None):
    windows = os.name == "nt" if windows is None else windows
    if not windows:
        return shlex.quote(text)
    if '"' in text:
        raise ValueError("Windows paths cannot contain double quotes")
    return f'"{text}"'
