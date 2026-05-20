#!/usr/bin/env python3
"""Update the Contributors section of docs/community/index.md from the GitHub API.

Run this script before a release or whenever the contributor list should be
refreshed:

    python scripts/generate_contributors.py

The script replaces the content between the sentinel comments
``<!-- BEGIN CONTRIBUTORS -->`` and ``<!-- END CONTRIBUTORS -->`` in
``docs/community/index.md`` and leaves everything else untouched.

Requires only the Python standard library. GitHub's unauthenticated rate limit
(60 req/hr) is sufficient for the small number of contributors in this repo.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path

REPO = "ibs-lab/cedalion"
INDEX_FILE = Path(__file__).parent.parent / "docs" / "community" / "index.md"

BEGIN_SENTINEL = "<!-- BEGIN CONTRIBUTORS -->"
END_SENTINEL = "<!-- END CONTRIBUTORS -->"

# Shown first in their own "Core Maintainers" section (in this fixed order).
MAINTAINERS = [
    {"login": "emiddell", "name": "Eike Middell"},
    {"login": "avolu", "name": "Alexander von Lühmann"},
    {"login": "dboas", "name": "David Boas"},
]

# Anonymous git commits whose author name maps to a GitHub login.
ANON_MERGE: dict[str, str] = {
    "Shakiba Moradi": "shakiba93",
}

# Merge duplicate accounts: keys are absorbed into the value login.
ACCOUNT_MERGE: dict[str, str] = {
    "isa-musisi-mcs": "isamusisi",
}

# Override the display name fetched from GitHub (for profiles without a real name set).
NAME_OVERRIDES: dict[str, str] = {
    "shakiba93": "Shakiba Moradi",
    "isamusisi": "Isa Musisi",
    "ahns97": "Sung Min Ahn",
}

_MAINTAINER_LOGINS = {m["login"] for m in MAINTAINERS}


def _gh_get(url: str) -> list | dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "cedalion-docs-builder/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def fetch_contributors() -> dict[str, int]:
    """Return {login: commit_count}, merging anonymous and duplicate entries."""
    data = _gh_get(
        f"https://api.github.com/repos/{REPO}/contributors?per_page=100&anon=true"
    )
    counts: dict[str, int] = {}
    for entry in data:
        if entry.get("type") == "Anonymous":
            name = entry.get("name", "")
            login = ANON_MERGE.get(name)
            if login:
                counts[login] = counts.get(login, 0) + entry["contributions"]
        else:
            login = entry["login"]
            # Redirect duplicate accounts to their canonical login.
            login = ACCOUNT_MERGE.get(login, login)
            counts[login] = counts.get(login, 0) + entry["contributions"]
    return counts


def fetch_display_name(login: str) -> str:
    """Return the display name for a login, preferring NAME_OVERRIDES."""
    if login in NAME_OVERRIDES:
        return NAME_OVERRIDES[login]
    try:
        profile = _gh_get(f"https://api.github.com/users/{login}")
        return profile.get("name") or login
    except Exception:
        return login


def _avatar_url(login: str, size: int = 100) -> str:
    return f"https://avatars.githubusercontent.com/{login}?s={size}"


def _card(
    login: str, name: str, commits: int | None, *, maintainer: bool = False
) -> str:
    cls = "contributor-card maintainer" if maintainer else "contributor-card"
    if commits is not None:
        noun = "commit" if commits == 1 else "commits"
        commits_html = f'    <div class="contributor-commits">{commits} {noun}</div>\n'
    else:
        commits_html = ""
    return (
        f'<div class="{cls}">\n'
        f'  <a href="https://github.com/{login}" target="_blank" rel="noopener">\n'
        f'    <img src="{_avatar_url(login)}" alt="{name}" loading="lazy"/>\n'
        f'    <div class="contributor-name">{name}</div>\n'
        f"{commits_html}"
        f"  </a>\n"
        f"</div>"
    )


def build_section(counts: dict[str, int]) -> str:
    """Build the auto-generated block that goes between the sentinel comments."""
    maintainer_cards = "\n".join(
        _card(m["login"], m["name"], counts.get(m["login"]), maintainer=True)
        for m in MAINTAINERS
    )

    remaining = sorted(
        (
            (login, cnt)
            for login, cnt in counts.items()
            if login not in _MAINTAINER_LOGINS
        ),
        key=lambda x: -x[1],
    )
    print(f"  Fetching display names for {len(remaining)} contributors...")
    contrib_cards = "\n".join(
        _card(login, fetch_display_name(login), cnt) for login, cnt in remaining
    )

    today = date.today().isoformat()
    return (
        f"% AUTO-GENERATED — do not edit by hand.\n"
        f"% Regenerate with: python scripts/generate_contributors.py\n"
        f"% Last updated: {today}\n"
        f"\n"
        f"### Core Maintainers\n"
        f"\n"
        f"````{{raw}} html\n"
        f'<div class="contributor-grid">\n'
        f"{maintainer_cards}\n"
        f"</div>\n"
        f"````\n"
        f"\n"
        f"### Code Contributors\n"
        f"\n"
        f"````{{raw}} html\n"
        f'<div class="contributor-grid">\n'
        f"{contrib_cards}\n"
        f"</div>\n"
        f"````\n"
    )


def main() -> None:
    print(f"Fetching contributor data from {REPO}...")
    counts = fetch_contributors()
    print(f"  Found {len(counts)} unique contributors")

    section = build_section(counts)

    text = INDEX_FILE.read_text(encoding="utf-8")
    start = text.index(BEGIN_SENTINEL) + len(BEGIN_SENTINEL)
    end = text.index(END_SENTINEL)
    new_text = text[:start] + "\n" + section + text[end:]
    INDEX_FILE.write_text(new_text, encoding="utf-8")
    print(f"Updated contributors section in {INDEX_FILE}")


if __name__ == "__main__":
    main()
