"""Sanitize stored CI artifacts before upload.

``python -m infx.results.redact PATH [PATH ...]`` rewrites text files, ``.gz`` files, and
``.tar.gz`` archives in place.

The rules port ``packages/db/src/etl/log-redaction.ts`` from InferenceX-app, and both repositories
test against the same vectors (``tests/results/redaction_vectors.json`` is a byte-for-byte copy of
the app's ``log-redaction.vectors.json``). Only the standard library is used, because the result
tooling runs on a bare interpreter. So the app's variable-length lookbehinds become alternations of
fixed-width lookbehinds or bounded checks in Python, and every class is spelled out to match exactly
what JavaScript's ``u`` and ``iu`` flags match, ASCII ``\\w``, ``\\d`` and ``\\b`` included.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import ipaddress
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import IO, Literal

Outcome = Literal["changed", "unchanged", "guarded"]

REDACTED = "***REDACTED***"
_SNIFF_BYTES = 8192
_SPOOL_BYTES = 64 * 1024 * 1024
_BLOCK_BYTES = 1 << 20

# JavaScript's \s, which also covers Unicode spaces that Python's ASCII \s leaves out.
_SPACE = r"\t\n\v\f\r \xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_SPACE_CHARS = "".join(
    [
        *"\t\n\v\f\r \xa0\u1680",
        *map(chr, range(0x2000, 0x200B)),
        *"\u2028\u2029\u202f\u205f\u3000\ufeff",
    ]
)
_SPACE_SET = frozenset(_SPACE_CHARS)
# With the i and u flags, JavaScript folds ſ (U+017F) to s and the Kelvin sign (U+212A) to k.
_FOLDS = "\u017f\u212a"
_FOLD_OF = {"s": "\u017f", "k": "\u212a"}


def _ci(word: str) -> str:
    """``word`` matched the way JavaScript's ``iu`` flags match it."""
    return "".join(
        f"[{char}{char.upper()}{_FOLD_OF.get(char, '')}]" if char.isalpha() else re.escape(char)
        for char in word
    )


# The ANSI CSI introducers, ESC or its escaped spellings in a repr or JSON, grouped by width.
_CSI_INTROS = (r"\x1b", r"\\x1b|\\033", r"\\u001b")
_CSI_INTROS_CI = (r"\x1b", rf"\\{_ci('x1b')}|\\033", rf"\\{_ci('u001b')}")
_CSI_INTRO_TEXTS = ("\x1b", "\\x1b", "\\u001b", "\\033")
_ESCAPE_END = r"(?:(?:\x1b|\\x1b|\\u001b|\\033)\[[0-9;?]{0,16}[A-Za-z]|\\[nrt])"


def _after_escape(skip: int, *, ci: bool = False) -> str:
    """Assert the app's ``AFTER_ESCAPE`` ``skip`` characters back.

    That is a CSI sequence ending in a letter (``\\x1b[0m``, ``\\x1b[2K``) or an escaped ``\\n``,
    ``\\r`` or ``\\t``. Python needs fixed-width lookbehinds, so this is one per parameter count,
    gated on the cheap two-character check that all of them share.
    """
    skipped = rf"[\s\S]{{{skip}}}" if skip else ""
    letter = f"[A-Za-z{_FOLDS}]" if ci else "[A-Za-z]"
    escaped = r"\\[nrtNRT]" if ci else r"\\[nrt]"
    intros = _CSI_INTROS_CI if ci else _CSI_INTROS
    sequences = "|".join(
        rf"(?<=\[[0-9;?]{{{params}}}{letter}{skipped})(?:"
        + "|".join(rf"(?<=(?:{intro})[\s\S]{{{params + 2 + skip}}})" for intro in intros)
        + ")"
        for params in range(17)
    )
    return rf"(?:(?<={escaped}{skipped})|(?<=[0-9;?\[]{letter}{skipped})(?:{sequences}))"


def _start(skip: int, *, ci: bool = False) -> str:
    """Assert the app's ``START`` (no word character before, or an escape) ``skip`` characters back."""
    skipped = rf"[\s\S]{{{skip}}}" if skip else ""
    word = f"[A-Za-z0-9_{_FOLDS}]" if ci else "[A-Za-z0-9_]"
    return rf"(?:(?<!{word}{skipped})|{_after_escape(skip, ci=ci)})"


def _token(head: str, width: int, rest: str) -> re.Pattern[str]:
    """A token that starts with ``head`` (``width`` characters), checking ``START`` after it."""
    return re.compile(head + _start(width) + rest, re.ASCII)


_TOKENS = (
    _token("hf_", 3, r"[A-Za-z0-9]{30,}\b"),
    _token("gh[pousr]_", 4, r"[A-Za-z0-9]{36,}\b"),
    _token("github_pat_", 11, r"[A-Za-z0-9_]{50,}\b"),
    _token("(?:AKIA|ASIA)", 4, r"[0-9A-Z]{16}\b"),
    _token("xox[abprs]-", 5, r"[A-Za-z0-9-]{10,}"),
    _token("sk-", 3, r"(?:ant-|proj-)?[A-Za-z0-9_-]{32,}"),
    _token("nvapi-", 6, r"[A-Za-z0-9_-]{20,}"),
    # A JWT, which is a base64url `{"…` header and payload and a signature (HF Xet tokens use it).
    _token("eyJ", 3, r"[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
)
_AUTHORIZATION = re.compile(
    rf"(?P<prefix>{_ci('authorization')}{_start(13, ci=True)}"
    rf"\\?[\"']?[ \t]{{0,4}}[:=][ \t]{{0,4}}\\?[\"']?"
    rf"(?:{_ci('bearer')}|{_ci('basic')}|{_ci('token')})[ \t]{{1,4}})"
    rf"[^{_SPACE}\"',;\\]{{8,}}",
    re.ASCII,
)
_BEARER = re.compile(
    rf"(?P<prefix>{_ci('bearer')}{_start(6, ci=True)}[ \t]{{1,4}})"
    rf"(?P<value>[A-Za-z0-9._~+/={_FOLDS}-]{{20,}})",
    re.ASCII,
)
_UPPER_KEY = r"(?:TOKEN|API_?KEY|SECRET(?:_KEY)?|ACCESS_KEY|PASSWORD|PASSWD)"
_LOWER_KEY = r"(?:token|api_?key|secret|secret_key|password|passwd)"
# The alternatives of the app's key words in the order its regex tries them.
_UPPER_KEYS = (
    "TOKEN",
    "API_KEY",
    "APIKEY",
    "SECRET_KEY",
    "SECRET",
    "ACCESS_KEY",
    "PASSWORD",
    "PASSWD",
)
_LOWER_KEYS = ("token", "api_key", "apikey", "secret", "secret_key", "password", "passwd")
_KEY_WORD = re.compile(r"TOKEN|API_?KEY|SECRET|ACCESS_KEY|PASSW|token|api_?key|secret|passw")
# The app's key lookbehinds, run forward over a window that ends at the key word. 160 characters
# hold the longest key (24 for an escape, 99 for three lower-case segments, 10 for the word).
_KEY_WINDOW = 160
_UPPER_KEY_BEFORE = re.compile(
    rf"(?:\A|[^A-Za-z0-9_]|{_ESCAPE_END})(?:[A-Z][A-Z0-9_]{{0,63}}_)?{_UPPER_KEY}\Z", re.ASCII
)
_LOWER_KEY_BEFORE = re.compile(
    rf"(?:\A|[^A-Za-z0-9_]|{_ESCAPE_END})(?:[a-z][a-z0-9]{{0,31}}_){{0,3}}{_LOWER_KEY}\Z", re.ASCII
)
_ASSIGNED = r"\\?[\"']?[ \t]{0,4}[=:][ \t]{0,4}\\?[\"']?"
_SECRET_VALUE = rf"[^{_SPACE}\"',;}}\\]"
_ASSIGNMENT_HEAD = re.compile(rf"{_ASSIGNED}{_SECRET_VALUE}{{8}}", re.ASCII)
_ASSIGNMENT = re.compile(rf"{_ASSIGNED}(?P<value>{_SECRET_VALUE}{{8,}})", re.ASCII)
_SECRET_FLAG = re.compile(
    "--"
    + _start(2)
    + r"(?:[a-z0-9]{1,32}-){0,4}(?:token|api-key|secret|password)(?:=|[ \t]{1,4})[\"']?"
    + rf"(?P<value>{_SECRET_VALUE}{{8,}})",
    re.ASCII,
)
_PLACEHOLDER = re.compile(
    r"[$<{%*\[]|(?:"
    + "|".join(_ci(word) for word in ("none", "null", "true", "false", "unset", "changeme"))
    + r"|[xX]+|[0-9.]+)\Z",
    re.ASCII,
)
_LETTER = re.compile("[A-Za-z]")
_DIGIT = re.compile("[0-9]")
_URL_SEPARATOR = re.compile(r":(?://|\\/\\/)")
_URL_PASSWORD = re.compile(
    rf"(?P<userinfo>:(?://|\\/\\/)[^{_SPACE}:/@\"',;<>\\]*:)[^{_SPACE}@/\"',;<>]+@", re.ASCII
)
_SCHEME_WINDOW = 64
_SCHEME_BEFORE = re.compile(
    rf"(?:\A|[^A-Za-z0-9_]|{_ESCAPE_END})[A-Za-z][A-Za-z0-9+.-]{{0,31}}\Z", re.ASCII
)
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
# An email's domain, from the `@`. Its local part and the boundary before it are checked in Python.
_EMAIL_DOMAIN = re.compile(
    rf"@{_LABEL}(?:\.{_LABEL}){{0,8}}\.[A-Za-z]{{2,24}}"
    rf"(?![A-Za-z0-9_-]|\.[A-Za-z0-9]|:[^{_SPACE}]|\()",
    re.ASCII,
)
_EMAIL_HINT = re.compile(r"@[A-Za-z0-9-]{1,63}\.[A-Za-z]", re.ASCII)
_EMAIL_LOCAL = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+-")
_BEFORE_EMAIL = _EMAIL_LOCAL | {"\\"}
_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_CSI_PARAMS = frozenset("0123456789;?")
_INTERNAL_TLD = re.compile(r"\.(?:internal|local|lan|localdomain)\Z", re.ASCII | re.IGNORECASE)
_ALNUM_CI = f"A-Za-z0-9{_FOLDS}"
_ORACLE_VCN = re.compile(
    rf"(?<![{_ALNUM_CI}_])(?P<host>[{_ALNUM_CI}](?:[{_ALNUM_CI}-]{{0,61}}[{_ALNUM_CI}])?)"
    rf"(?:\.[{_ALNUM_CI}-]{{1,63}}){{0,8}}\.{_ci('oraclevcn')}\.{_ci('com')}(?![{_ALNUM_CI}_])",
    re.ASCII,
)
_ORACLE_VCN_HINT = re.compile(r"oraclevcn\.com", re.ASCII | re.IGNORECASE)
# The leading lookaheads only skip positions where the address cannot start, which is cheaper
# than trying the lookbehinds there.
_IPV6 = re.compile(
    rf"(?=[0-9A-Fa-f]{{0,4}}:[0-9A-Fa-f]{{0,4}}:)(?:(?<![A-Za-z0-9_:.])|{_after_escape(0)})"
    r"(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![A-Za-z0-9_:]|\.[0-9])",
    re.ASCII,
)
_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
_IPV4 = re.compile(
    rf"(?=(?:[0-9]{{1,3}}\.){{3}}[0-9])(?:(?<![A-Za-z0-9_.])|{_after_escape(0)})"
    rf"(?:{_OCTET}\.){{3}}{_OCTET}(?![A-Za-z0-9_]|\.[0-9])",
    re.ASCII,
)
_VERSION_OPERATORS = frozenset(("==", "!=", ">=", "<=", "~="))
_ALNUM = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_PIP_SATISFIED = "Requirement already satisfied:"
_ENVIRONMENT_DUMP = re.compile(r"\[(?:conda|pip3?)\][ \t]+")
# The characters of a package name. `_package_name` also requires a `-` or `_` after the first, so
# the app's `[-_]` split needs no backtracking here.
_PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z", re.ASCII)
_LOWER = re.compile("[a-z]")
# The app's line rules look this many UTF-16 code units back.
_LINE_WINDOW = 1024
_ADDRESS_KEY = re.compile(
    "|".join(
        _ci(word)
        for word in (
            "host", "addr", "ip", "url", "uri", "endpoint", "server",
            "peer", "remote", "bind", "master", "node", "target",
        )
    ),
    re.ASCII,
)  # fmt: skip
# The JSON key lookup of a later block needs at most the text since the second-to-last quote.
# Longer stretches are summarized, see `_compact_context`.
_CONTEXT_LIMIT = 4096
_DIGEST_FIELD = re.compile(rb'"[^"]*sha256"\s*:\s*"([0-9A-Fa-f]{64})"', re.IGNORECASE)


def _bearer(match: re.Match[str]) -> str:
    """A bare ``Bearer`` value is a token unless it is a slash path without digits."""
    value = match["value"]
    if "/" in value and not _DIGIT.search(value):
        return match[0]
    return match["prefix"] + REDACTED


def _secret_value(match: str, value: str) -> str:
    """``match`` with ``value`` redacted, unless a placeholder or a lower-case setting."""
    if _PLACEHOLDER.match(value):
        return match
    lower = not "A" <= match[0] <= "Z"
    if lower and not (_LETTER.search(value) and _DIGIT.search(value)):
        return match
    return match[: len(match) - len(value)] + REDACTED


def _assignment_at(text: str, start: int) -> re.Match[str] | None:
    """The app's ``SECRET_ASSIGNMENT`` tried at ``start``, its alternatives in order."""
    upper = text[start] <= "Z"
    keys, before = (_UPPER_KEYS, _UPPER_KEY_BEFORE) if upper else (_LOWER_KEYS, _LOWER_KEY_BEFORE)
    for key in keys:
        if not text.startswith(key, start):
            continue
        end = start + len(key)
        # The bounded head first, so a run of failing keys never rescans a long value.
        if _ASSIGNMENT_HEAD.match(text, end) and before.search(
            text, max(0, end - _KEY_WINDOW), end
        ):
            return _ASSIGNMENT.match(text, end)
    return None


def _secret_assignments(text: str) -> str:
    pieces: list[str] = []
    last = position = 0
    while (word := _KEY_WORD.search(text, position)) is not None:
        start = word.start()
        assignment = _assignment_at(text, start)
        if assignment is None:
            position = start + 1
            continue
        end = assignment.end()
        pieces += (text[last:start], _secret_value(text[start:end], assignment["value"]))
        last = position = end
    return "".join([*pieces, text[last:]]) if pieces else text


def _url_passwords(text: str) -> str:
    pieces: list[str] = []
    last = position = 0
    while (separator := _URL_SEPARATOR.search(text, position)) is not None:
        start = separator.start()
        position = start + 1
        if not _SCHEME_BEFORE.search(text, max(0, start - _SCHEME_WINDOW), start):
            continue
        if (password := _URL_PASSWORD.match(text, start)) is not None:
            pieces += (text[last:start], password["userinfo"], REDACTED, "@")
            last = position = password.end()
    return "".join([*pieces, text[last:]]) if pieces else text


def _csi_open_before(text: str, end: int) -> bool:
    """Whether a CSI introducer and up to 16 parameter characters end at ``end``."""
    for _ in range(17):
        if end and text[end - 1] == "[" and text.endswith(_CSI_INTRO_TEXTS, 0, end - 1):
            return True
        if not end or text[end - 1] not in _CSI_PARAMS:
            return False
        end -= 1
    return False


def _escape_before(text: str, end: int) -> bool:
    """The app's ``AFTER_ESCAPE`` at ``end``."""
    if not end or text[end - 1] not in _LETTERS:
        return False
    if end >= 2 and text[end - 2] == "\\" and text[end - 1] in "nrt":
        return True
    return _csi_open_before(text, end - 1)


def _email_may_start(text: str, start: int) -> bool:
    """The lookbehinds that open the app's ``EMAIL``."""
    previous = text[start - 1] if start else ""
    bare = previous not in _BEFORE_EMAIL and not _csi_open_before(text, start)
    if not (bare or _escape_before(text, start)):
        return False
    return start < 3 or text[start - 3 : start] != "://"


def _emails(text: str) -> str:
    """The app's ``EMAIL`` rule, driven by each ``@`` whose domain matches."""
    pieces: list[str] = []
    last = position = 0
    while (domain := _EMAIL_DOMAIN.search(text, position)) is not None:
        at = domain.start()
        position = at + 1
        first = at
        floor = max(last, at - 64)
        while first > floor and text[first - 1] in _EMAIL_LOCAL:
            first -= 1
        start = next((index for index in range(first, at) if _email_may_start(text, index)), None)
        if start is None:
            continue
        email = text[start : domain.end()]
        pieces += (text[last:start], email if _INTERNAL_TLD.search(email) else REDACTED)
        last = position = domain.end()
    return "".join([*pieces, text[last:]]) if pieces else text


def _oracle_vcn_host(match: re.Match[str]) -> str:
    return f"{match['host']}.{REDACTED}"


def _oracle_vcn(text: str) -> str:
    """The app's ``ORACLE_VCN`` rule, run only on lines that name ``oraclevcn.com``.

    A match never spans a line break, so each such line gives what the whole text would.
    """
    pieces: list[str] = []
    last = 0
    while (hint := _ORACLE_VCN_HINT.search(text, last)) is not None:
        start = text.rfind("\n", last, hint.start()) + 1 or last
        end = text.find("\n", hint.end())
        end = len(text) if end < 0 else end
        pieces += (text[last:start], _ORACLE_VCN.sub(_oracle_vcn_host, text[start:end]))
        last = end
    return "".join([*pieces, text[last:]]) if pieces else text


def _ipv6(match: re.Match[str]) -> str:
    address = match[0]
    try:
        ipaddress.IPv6Address(address)
    except ValueError:
        return address
    return REDACTED if int(address.split(":", 1)[0] or "0", 16) >> 13 == 1 else address


def _space_before(text: str, index: int) -> int:
    """The index of the last non-space character before ``index``, or -1."""
    index -= 1
    while index >= 0 and text[index] in _SPACE_SET:
        index -= 1
    return index


def _json_key(text: str, start: int, context: str) -> str | None:
    """Key of the quoted JSON value opening right before ``start``, as the app's ``jsonValueKey``.

    ``context`` is what ``_next_context`` kept of the text before ``text``, or empty at its start.
    """
    at = start - 1
    if at < 0 or text[at] != '"':
        return None
    at = _space_before(text, at)
    if at >= 0:
        if text[at] != ":":
            return None
        at = _space_before(text, at)
        if at >= 0 and text[at] != '"':
            return None
        if at > 0:
            opening = text.rfind('"', 0, at)
            if opening >= 0:
                return None if opening == at - 1 else text[opening + 1 : at]
    # The lookup reached the start of `text`.
    if context:
        return _json_key(context + text[:start], len(context) + start, "")
    # At the very start, String.prototype.lastIndexOf clamps its -1 start to 0 and finds this quote.
    return "" if at == 0 else None


def _line_prefix(text: str, start: int, continued: bool) -> str | None:
    """The line up to ``start``, or None if the line is longer than the app's window.

    ``continued`` says whether a line break precedes ``text``.
    """
    newline = text.rfind("\n", max(0, start - _LINE_WINDOW), start)
    if newline < 0 and start > _LINE_WINDOW:
        return None
    prefix = text[newline + 1 : start]
    units = (
        len(prefix) if prefix.isascii() else len(prefix.encode("utf-16-le", "surrogatepass")) // 2
    )
    # The window must also hold the line break when there is one.
    return prefix if units <= _LINE_WINDOW - (newline >= 0 or continued) else None


def _package_name(prefix: str) -> str | None:
    """The package column of a ``pip list``/``conda list`` row ending in ``prefix``, if any."""
    name = prefix.rstrip(" \t")
    padding = len(prefix) - len(name)
    if not padding:
        return None
    dump = _ENVIRONMENT_DUMP.match(name)
    if dump is not None:
        name = name[dump.end() :]
    elif padding < 2:
        return None
    if not _PACKAGE_NAME.match(name) or ("-" not in name and "_" not in name):
        return None
    return name


def _compact_rest(rest: str) -> str:
    """``rest``, text after the last quote, shortened without changing a later JSON key lookup.

    A lookup walks back over spaces and one colon to the quote, or uses all of ``rest`` as the
    start of a key that is only tested for an address word.
    """
    body = rest.rstrip(_SPACE_CHARS)
    if body.endswith(":"):
        body = body[:-1].rstrip(_SPACE_CHARS)
    walk = ":".join(" " if part else "" for part in rest[len(body) :].split(":"))
    if not body:
        return walk
    # A non-space, non-colon character stops a walk, and the last 7 characters keep any address
    # word that crosses into the next block.
    return ("host\0" if _ADDRESS_KEY.search(body) else "\0") + body[-7:] + walk


def _compact_context(context: str) -> str:
    if len(context) <= _CONTEXT_LIMIT:
        return context
    first, last = context.find('"'), context.rfind('"')
    head = context[: first + 1]
    if first < last:
        key = context[first + 1 : last]
        if len(key) > _CONTEXT_LIMIT:
            key = "host" if _ADDRESS_KEY.search(key) else "#"
        head += key + '"'
    rest = context[last + 1 :]
    return head + (_compact_rest(rest) if len(rest) > _CONTEXT_LIMIT else rest)


def _next_context(context: str, text: str) -> str:
    """``context + text`` from its second-to-last quote, which is all a later JSON key lookup reads.

    A NUL stands for the earlier text when there is some, so the lookup at the very start of the
    whole text stays the only one that sees index 0.
    """
    last = text.rfind('"')
    if last < 0:
        return _compact_context(context + text) if '"' in context else "\0"
    second = text.rfind('"', 0, last)
    if second >= 0:
        return _compact_context(("\0" if context or second else "") + text[second:])
    previous = context.rfind('"')
    if previous >= 0:
        return _compact_context(("\0" if previous else "") + context[previous:] + text)
    return _compact_context(("\0" if context or last else "") + text[last:])


class _Redactor:
    """Redacts one text, whole or in consecutive blocks of whole lines, with the same result."""

    def __init__(self, *, emails: bool | None = None) -> None:
        # None runs the email rule only when the text holds an email hint, as the app does. A file
        # streamed in blocks runs it everywhere and is redone without it if no block had a hint.
        self.emails = emails
        self.context = ""
        self.hinted = False
        self.emailed = False
        self.changed = False

    @property
    def redo(self) -> bool:
        """Whether emails were redacted although the app would have skipped the rule."""
        return self.emailed and not self.hinted

    def redact(self, text: str) -> str:
        original = text
        for pattern in _TOKENS:
            text = pattern.sub(REDACTED, text)
        text = _AUTHORIZATION.sub(lambda match: match["prefix"] + REDACTED, text)
        text = _BEARER.sub(_bearer, text)
        text = _secret_assignments(text)
        text = _SECRET_FLAG.sub(lambda match: _secret_value(match[0], match["value"]), text)
        text = _url_passwords(text)
        hinted = _EMAIL_HINT.search(text) is not None
        self.hinted |= hinted
        if self.emails or (self.emails is None and hinted):
            emailed = _emails(text)
            self.emailed |= emailed != text
            text = emailed
        text = _oracle_vcn(text)
        text = _IPV6.sub(_ipv6, text)
        text = _IPV4.sub(self._ipv4, text)
        self.context = _next_context(self.context, text)
        self.changed |= text != original
        return text

    def _ipv4(self, match: re.Match[str]) -> str:
        address = match[0]
        first, second, _ = address.split(".", 2)
        a, b = int(first), int(second)
        if (
            a in {0, 10, 127}
            or a >= 224
            or (a == 100 and 64 <= b <= 127)
            or (a == 169 and b == 254)
            or (a == 172 and 16 <= b <= 31)
            or (a == 192 and b == 168)
            or self._version_context(match.string, match.start(), match.end())
        ):
            return address
        return REDACTED

    def _version_context(self, text: str, start: int, end: int) -> bool:
        before = text[max(0, start - 2) : start]
        if before in _VERSION_OPERATORS or (
            len(before) == 2 and before[1] == "-" and before[0] in _ALNUM
        ):
            return True
        opening = text[start - 1] if start else ""
        close = text[end] if end < len(text) else ""
        if opening == "(" and close == ")":
            prefix = _line_prefix(text, start, bool(self.context))
            if prefix is not None and _PIP_SATISFIED in prefix:
                return True
        elif opening in {" ", "\t"} and (not close or close in _SPACE_SET):
            prefix = _line_prefix(text, start, bool(self.context))
            name = None if prefix is None else _package_name(prefix)
            if name is not None and _LOWER.search(name) and not _ADDRESS_KEY.search(name):
                return True
        if close != '"':
            return False
        key = _json_key(text, start, self.context)
        return key is not None and not _ADDRESS_KEY.search(key)

    def copy(self, source: IO[bytes], sink: IO[bytes]) -> None:
        """Write the redacted text of ``source`` to ``sink``, decoding invalid UTF-8 losslessly."""
        for block in _blocks(source):
            decoded = block.decode("utf-8", "surrogateescape")
            redacted = self.redact(decoded)
            sink.write(
                block if redacted == decoded else redacted.encode("utf-8", "surrogateescape")
            )


def redact_text(text: str) -> str:
    """``text`` sanitized. The output is a fixed point, so sanitizing it again changes nothing."""
    return _Redactor().redact(text)


def _blocks(stream: IO[bytes]) -> Iterator[bytes]:
    """``stream`` in blocks of about ``_BLOCK_BYTES`` that end at a ``\\n``, except the last."""
    pending = bytearray()
    for chunk in iter(lambda: stream.read(_BLOCK_BYTES), b""):
        pending += chunk
        newline = chunk.rfind(b"\n")
        if newline >= 0:
            cut = len(pending) - len(chunk) + newline + 1
            yield bytes(pending[:cut])
            del pending[:cut]
    if pending:
        yield bytes(pending)


def _redact_stream(
    source: IO[bytes],
    sink: IO[bytes],
    wrap: Callable[
        [IO[bytes]], contextlib.AbstractContextManager[IO[bytes]]
    ] = contextlib.nullcontext,
) -> bool:
    """Write ``source`` redacted to ``sink`` through ``wrap``, from their current positions.

    Both must be seekable, for the rare text whose only email lacks the app's email hint.
    """
    source_start, sink_start = source.tell(), sink.tell()
    redactor = _Redactor(emails=True)
    with wrap(sink) as target:
        redactor.copy(source, target)
    if redactor.redo:
        source.seek(source_start)
        sink.seek(sink_start)
        sink.truncate()
        redactor = _Redactor(emails=False)
        with wrap(sink) as target:
            redactor.copy(source, target)
    return redactor.changed


def _sha256(handle: IO[bytes]) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def _replace(path: Path, write: Callable[[IO[bytes]], bool], digests: frozenset[str]) -> Outcome:
    """Write ``path``'s redacted form to a temp file and swap it in if it differs."""
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".redact")
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w+b") as sink:
            changed = write(sink)
        if not changed:
            return "unchanged"
        if digests and _file_sha256(path) in digests:
            return "guarded"
        shutil.copymode(path, temp)
        temp.replace(path)
        return "changed"
    finally:
        temp.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256(handle)


def _redact_member(data: IO[bytes], buffer: IO[bytes], digests: frozenset[str], label: str) -> bool:
    """Write member ``data`` to ``buffer``, redacted unless binary or referenced by a digest."""
    head = data.read(_SNIFF_BYTES)
    if b"\0" in head:
        buffer.write(head)
        shutil.copyfileobj(data, buffer)
        return False
    data.seek(0)
    if not digests:
        return _redact_stream(data, buffer)
    with tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES) as original:
        shutil.copyfileobj(data, original)
        original.seek(0)
        if not _redact_stream(original, buffer):
            return False
        original.seek(0)
        if _sha256(original) not in digests:
            return True
        print(f"::warning::left {label} unredacted because a JSON sha256 field records its digest")
        buffer.seek(0)
        buffer.truncate()
        original.seek(0)
        shutil.copyfileobj(original, buffer)
        return False


def _redact_tar(path: Path, sink: IO[bytes], digests: frozenset[str]) -> bool:
    changed = False
    with (
        tarfile.open(path, "r:gz") as source,
        tarfile.open(fileobj=sink, mode="w:gz", compresslevel=6) as archive,
    ):
        for member in source:
            data = source.extractfile(member) if member.isfile() else None
            if data is None:
                archive.addfile(member)
                continue
            with tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES) as buffer:
                changed |= _redact_member(data, buffer, digests, f"{path}:{member.name}")
                member.size = buffer.tell()
                buffer.seek(0)
                archive.addfile(member, buffer)
    return changed


def redact_file(path: Path, digests: frozenset[str]) -> Outcome:
    """Redact ``path`` in place unless it is binary or its sha256 is in ``digests``."""
    if path.name.endswith((".tar.gz", ".tgz")):
        return _replace(path, lambda sink: _redact_tar(path, sink, digests), digests)
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as source:
            if b"\0" in source.read(_SNIFF_BYTES):
                return "unchanged"
            source.seek(0)

            def compress(sink: IO[bytes]) -> gzip.GzipFile:
                return gzip.GzipFile(
                    filename=path.name, mode="wb", fileobj=sink, compresslevel=6, mtime=source.mtime
                )

            return _replace(path, lambda sink: _redact_stream(source, sink, compress), digests)
    with path.open("rb") as source:
        if b"\0" in source.read(_SNIFF_BYTES):
            return "unchanged"
        source.seek(0)
        return _replace(path, lambda sink: _redact_stream(source, sink), digests)


def _files(paths: Iterable[Path]) -> list[Path]:
    """Every regular file under ``paths`` once, by resolved path, skipping ``.git``."""
    found: dict[Path, None] = {}

    def add(candidate: Path) -> None:
        resolved = Path(os.path.realpath(candidate))
        if resolved.is_file():
            found.setdefault(resolved, None)

    for path in paths:
        if not path.is_dir():
            add(path)
            continue
        for root, dirs, names in os.walk(path):
            dirs[:] = sorted(name for name in dirs if name != ".git")
            for name in sorted(names):
                add(Path(root, name))
    return list(found)


def _scan_digests(lines: Iterable[bytes], digests: set[str]) -> None:
    for line in lines:
        digests.update(match.group(1).decode().lower() for match in _DIGEST_FIELD.finditer(line))


def _collect_digests(path: Path, digests: set[str]) -> None:
    try:
        if path.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    if member.isfile() and member.name.endswith(".json"):
                        _scan_digests(archive.extractfile(member), digests)
        elif path.suffix == ".json":
            with path.open("rb") as handle:
                _scan_digests(handle, digests)
    except (OSError, EOFError, tarfile.TarError, zlib.error) as error:
        print(f"::warning::could not read digests from {path}: {error}")


def _referenced_digests(files: Iterable[Path]) -> frozenset[str]:
    """Every 64-hex value of a JSON key ending in ``sha256`` in JSON files and archive members."""
    digests: set[str] = set()
    for path in files:
        _collect_digests(path, digests)
    return frozenset(digests)


def main(argv: list[str] | None = None) -> int:
    """Redact every file under the paths in ``argv`` and return 0."""
    parser = argparse.ArgumentParser(prog="python -m infx.results.redact", description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="files or directories; missing ones are ignored"
    )
    args = parser.parse_args(argv)
    files = _files(args.paths)
    digests = _referenced_digests(files)
    outcomes: Counter[str] = Counter()
    for path in files:
        try:
            outcome = redact_file(path, digests)
        except (OSError, EOFError, ValueError, tarfile.TarError, zlib.error) as error:
            print(f"::warning::could not redact {path}: {error}")
            continue
        outcomes[outcome] += 1
        if outcome == "guarded":
            print(
                f"::warning::left {path} unredacted because a JSON sha256 field records its digest"
            )
    print(
        f"Redaction scanned {len(files)} files, changed {outcomes['changed']}, "
        f"skipped {outcomes['guarded']} referenced by a sha256 digest"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
