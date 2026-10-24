# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep license-sensitive records aligned with the resolved manifest."""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_LOCK = tomllib.loads((_ROOT / "dependency-manifest" / "uv.lock").read_text())
_LOCK_VERSIONS: dict[str, set[str]] = {}
for _package in _LOCK["package"]:
    _LOCK_VERSIONS.setdefault(_package["name"], set()).add(_package["version"])

_NOTICES = (_ROOT / "THIRD_PARTY_NOTICES.md").read_text()
_RECIPROCAL_SECTION = _NOTICES.split(
    "### Reciprocal licenses and bundled native libraries", 1
)[1].split("### `text-unidecode` license election", 1)[0]
_RECIPROCAL_VERSIONS = dict(
    re.findall(r"^\| `([^`]+)` ([^ /|]+)", _RECIPROCAL_SECTION, re.MULTILINE)
)

# Keep this denylist explicit: a dependency audit that identifies another
# license-sensitive package must add its notice and guard record together.
_LICENSE_SENSITIVE_PACKAGES = frozenset(
    {
        "certifi",
        "num2words",
        "opencv-python-headless",
        "pycountry",
        "pyzmq",
        "soundfile",
        "soxr",
        "text-unidecode",
        "tqdm",
    }
)

_LICENSE_RECORDS = {
    "certifi": (
        "2026.10.19",
        (("certifi-2026.10.19/LICENSE", "Mozilla Public License"),),
    ),
    "num2words": (
        "0.5.14",
        (("num2words-0.5.14/COPYING", "GNU Lesser General Public License"),),
    ),
    "opencv-python-headless": (
        "5.0.0.93",
        (
            ("opencv-python-headless-5.0.0.93/LICENSE.txt", "MIT License"),
            (
                "opencv-python-headless-5.0.0.93/LICENSE-3RD-PARTY.txt",
                "FFmpeg is redistributed within all opencv-python packages",
            ),
        ),
    ),
    "pycountry": (
        "26.2.16",
        (
            ("pycountry-26.2.16/LICENSE.txt", "GNU LESSER GENERAL PUBLIC LICENSE"),
            (
                "pycountry-26.2.16/COPYRIGHT.txt",
                "COPYRIGHT (c) 2008 - 2023, pycountry",
            ),
        ),
    ),
    "pyzmq": (
        "27.2.0",
        (
            ("pyzmq-27.2.0/LICENSE.zeromq", "Mozilla Public License Version 2.0"),
            ("pyzmq-27.2.0/LICENSE.libsodium", "ISC License"),
        ),
    ),
    "soundfile": (
        "0.13.1",
        (
            ("libsndfile-1.2.2/COPYING", "GNU LESSER GENERAL PUBLIC LICENSE"),
            ("soundfile-0.13.1/native-notices.md", "Lame by Mike Cheng"),
            ("soundfile-0.13.1/native-notices.md", "Mpg123 by Michael Hipp"),
        ),
    ),
    "soxr": (
        "1.0.0",
        (
            ("soxr-1.0.0/COPYING.LGPL", "GNU LESSER GENERAL PUBLIC LICENSE"),
            ("soxr-1.0.0/LICENSE-libsoxr", "SoX Resampler Library"),
            ("soxr-1.0.0/LICENSE-PFFFT", "FFTPACK license"),
        ),
    ),
    "tqdm": (
        "4.70.1",
        (("tqdm-4.70.1/LICENCE", "Mozilla Public Licence (MPL) v. 2.0"),),
    ),
}


def test_license_sensitive_dependency_denylist_is_guarded() -> None:
    """Keep every known license-sensitive lock entry under test."""
    assert _LICENSE_SENSITIVE_PACKAGES <= _LOCK_VERSIONS.keys()
    assert _LICENSE_SENSITIVE_PACKAGES == _LICENSE_RECORDS.keys() | {
        "text-unidecode"
    }


@pytest.mark.parametrize("package", sorted(_LICENSE_RECORDS))
def test_reciprocal_dependency_has_current_license_record(package: str) -> None:
    """Fail a dependency refresh that leaves its compliance record stale."""
    audited_version, license_files = _LICENSE_RECORDS[package]
    assert _RECIPROCAL_VERSIONS[package] == audited_version
    assert audited_version in _LOCK_VERSIONS[package]
    for relative_path, required_text in license_files:
        license_file = _ROOT / "third_party_licenses" / relative_path
        contents = license_file.read_text()
        assert contents.strip()
        assert required_text in contents


def test_text_unidecode_artistic_license_election_is_explicit() -> None:
    assert _LOCK_VERSIONS["text-unidecode"] == {"1.3"}
    license_file = _ROOT / "third_party_licenses/text-unidecode-1.3/LICENSE"
    contents = license_file.read_text()
    assert contents.strip()
    assert 'The "Artistic License"' in contents
    assert "elects the **Artistic-1.0-Perl** option" in _NOTICES
    assert "does not rely on the GPL grant" in _NOTICES


def test_bundled_license_index_is_complete_and_links_exist() -> None:
    section = _NOTICES.split("## Bundled license file index\n", 1)[1].split("\n## ", 1)[0]
    table_links = re.findall(r"^\|[^\n]*\]\(([^)]+)\)", section, re.MULTILINE)
    assert len(table_links) == len(set(table_links)), "Duplicate license index entries"
    expected = {
        path.relative_to(_ROOT).as_posix()
        for path in (_ROOT / "third_party_licenses").rglob("*")
        if path.is_file()
        and path.name != "license_notes.md"
        and (
            path.name.upper().startswith(("LICENSE", "LICENCE", "COPYING"))
            or path.name == "FTL.TXT"
        )
    }
    assert set(table_links) == expected, (
        f"Missing entries: {expected - set(table_links)}; "
        f"Unexpected entries: {set(table_links) - expected}"
    )
    for target in re.findall(r"\]\(([^)]+)\)", section):
        assert target.startswith("third_party_licenses/"), target
        assert (_ROOT / target).is_file(), f"Broken license index link: {target}"


# Preserve the complete license and bundled-library texts from audited wheels.
_NATIVE_SECURITY_REFRESH = {
    "pillow": (
        "13.0.0",
        (
            ("pillow-13.0.0/LICENSE",
             "dda12a98c1979cf3d94df1cff45d27a4cb3f04a60c76f76902ac54cac03ec0ce"),
        ),
    ),
    "onnxruntime": (
        "1.31.0",
        (
            ("onnxruntime-1.31.0/LICENSE",
             "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c"),
            ("onnxruntime-1.31.0/LICENSE.thirdparty",
             "143764b952fdb1a7c69ce653bfba74a7744d6a8a573bfb73e235fba356c83de3"),
        ),
    ),
    "pipecat-ai": (
        "1.14.0",
        (
            ("pipecat-ai-1.14.0/LICENSE",
             "e31a733ecb79cbffafd5474e964e5eca15e0a0876a42b1d7253b51b198eb61aa"),
        ),
    ),
}


@pytest.mark.parametrize("package", sorted(_NATIVE_SECURITY_REFRESH))
def test_voice_image_refresh_has_audited_artifact_notices(package: str) -> None:
    """Require a fresh artifact audit when these qualified versions change."""
    version, files = _NATIVE_SECURITY_REFRESH[package]
    assert _LOCK_VERSIONS[package] == {version}
    assert f"third_party_licenses/{package}-{version}/" in _NOTICES
    for relative_path, digest in files:
        contents = (_ROOT / "third_party_licenses" / relative_path).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == digest
