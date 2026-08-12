from pathlib import Path

from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = ROOT / "requirements.txt"


def _read_requirements(path: Path) -> list[str]:
    requirements: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Remove pip-only inline flags (e.g. "package -f <url>") not valid in install_requires.
        for flag in (" -f ", " --find-links ", " --index-url ", " --extra-index-url "):
            if flag in line:
                line = line.split(flag, 1)[0].strip()
        # Skip standalone pip options that are not valid requirement specifiers.
        if line.startswith("-"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="andglore",
    version="0.1.0",
    description="Author Name Disambiguation via Global and Refined Views (AND-GloRe)",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=_read_requirements(REQUIREMENTS_PATH),
)
