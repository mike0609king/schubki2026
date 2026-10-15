rm -rf ./schubki2026_telepathy.zip
zip -r schubki2026_telepathy.zip . -x "venv/*" "*/__pycache__/*" "*/.mypy_cache/*" "schubki2026_telepathy.zip"
