#!/usr/bin/env python3
import json
import subprocess
import argparse


def get_last_version() -> str:
    """Return the version number of the last release."""
    json_string = (
        subprocess.run(
            ["gh", "release", "view", "--json", "tagName"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        .stdout.decode("utf8")
        .strip()
    )

    return json.loads(json_string)["tagName"]


def bump_major_number(version_number: str) -> str:
    """Return a copy of `version_number` with the patch number incremented."""
    major, minor, patch = version_number.lstrip("v").split(".")
    return f"v{int(major) + 1}.0.0"


def bump_minor_number(version_number: str) -> str:
    """Return a copy of `version_number` with the patch number incremented."""
    major, minor, patch = version_number.lstrip("v").split(".")
    return f"v{major}.{int(minor) + 1}.0"


def bump_patch_number(version_number: str) -> str:
    """Return a copy of `version_number` with the patch number incremented."""
    major, minor, patch = version_number.lstrip("v").split(".")
    return f"v{major}.{minor}.{int(patch) + 1}"


def create_new_release(args):
    """Create a new release on GitHub."""
    if args.version:
        new_version_number = args.version
    else:
        try:
            last_version_number = get_last_version()
        except subprocess.CalledProcessError as err:
            if err.stderr.decode("utf8").startswith("HTTP 404:"):
                # The project doesn't have any releases yet.
                new_version_number = "v0.1.0"
            else:
                raise
        else:
            if args.major:
                new_version_number = bump_major_number(last_version_number)
            if args.minor:
                new_version_number = bump_minor_number(last_version_number)
            if args.patch:
                new_version_number = bump_patch_number(last_version_number)

    subprocess.run(
        ["gh", "release", "create", "--generate-notes", new_version_number],
        check=True,
    )


if __name__ == "__main__":
    # Use argparse instead of dargparse because this script is used by the CI
    parser = argparse.ArgumentParser()
    parser.add_argument("--major", action="store_true")
    parser.add_argument("--minor", action="store_true")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--version", type=str, default=None)

    args = parser.parse_args()
    create_new_release(args)
