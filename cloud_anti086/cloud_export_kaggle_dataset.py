from __future__ import annotations

from pathlib import Path
import zipfile


def main() -> None:
    source = Path("artifacts/anti086")
    target = source / "anti086_kaggle_input.zip"
    names = [path for path in source.glob("*.json*")] + [path for path in source.glob("*.jsonl")]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(names)):
            archive.write(path, path.name)
    print(target)


if __name__ == "__main__":
    main()
