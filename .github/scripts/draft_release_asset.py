"""Find, list, or store assets of a DRAFT GitHub release by tag, stdlib only.

Draft releases have no git tag and are invisible to the by-tag endpoint, so the release
is located by scanning the release list. Downloads are left to curl: Python's redirect
handler forwards the Authorization header to the storage host, which rejects it.

usage: draft_release_asset.py --tag TAG (--find NAME | --list | --put PATH --name NAME)
Reads GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_API_URL from the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def api(url: str, *, method: str = "GET", data: bytes | None = None, content_type: str | None = None):
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if content_type:
        request.add_header("Content-Type", content_type)
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def find_release(tag: str) -> dict:
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    repo = os.environ["GITHUB_REPOSITORY"]
    for page in range(1, 6):
        releases = api(f"{base}/repos/{repo}/releases?per_page=100&page={page}")
        for release in releases:
            if release["tag_name"] == tag:
                return release
        if len(releases) < 100:
            break
    raise SystemExit(f"no release with tag {tag!r} (drafts included)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--find", metavar="NAME", help="print the asset's JSON (its `url` downloads it)")
    group.add_argument("--list", action="store_true")
    group.add_argument("--put", metavar="PATH", help="upload PATH, replacing an asset of the same name")
    parser.add_argument("--name", help="asset name for --put (default: the file name)")
    args = parser.parse_args()

    release = find_release(args.tag)
    assets = {asset["name"]: asset for asset in release["assets"]}
    if args.list:
        print(f"release id={release['id']} draft={release['draft']} tag={release['tag_name']}")
        for asset in release["assets"]:
            print(f"  id={asset['id']} name={asset['name']} size={asset['size']} state={asset['state']}")
        return 0
    if args.find:
        if args.find not in assets:
            raise SystemExit(f"asset {args.find!r} not on release {args.tag!r}")
        json.dump(assets[args.find], sys.stdout)
        return 0
    name = args.name or os.path.basename(args.put)
    if name in assets:
        api(assets[name]["url"], method="DELETE")
    upload_url = release["upload_url"].split("{", 1)[0] + "?" + urllib.parse.urlencode({"name": name})
    with open(args.put, "rb") as handle:
        data = handle.read()
    created = api(upload_url, method="POST", data=data, content_type="application/octet-stream")
    print(f"stored {name}: id={created['id']} size={created['size']} state={created['state']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as error:
        print(f"ERROR: {error.code} {error.reason} for {error.url}: {error.read()[:400]!r}", file=sys.stderr)
        sys.exit(1)
