#!/bin/sh
set -eu
recsys obtain --offline --data-dir /data/raw
recsys normalize --anime-csv /data/raw/anime.csv --synopsis-csv /data/raw/anime_with_synopsis.csv --output-dir /data/work/normalized
recsys build --catalog /data/work/normalized/catalog.json --normalization-report /data/work/normalized/normalization-report.json --glove /data/raw/glove.6B.300d.txt --output-dir /data/work/build
result=$(recsys export --catalog /data/work/normalized/catalog.json --neighbors /data/work/build/neighbors.json --build-report /data/work/build/build-report.json --normalization-report /data/work/normalized/normalization-report.json --output-dir /data/bundles)
case "$result" in
    /data/bundles/sha256-[a-f0-9]*\ sha256:*\ records=*) ;;
    *) echo 'Unexpected bundle export result' >&2; exit 1 ;;
esac
bundle=${result%% *}
case "$bundle" in
    /data/bundles/sha256-????????????????????????????????????????????????????????????????) ;;
    *) echo 'Unexpected bundle path' >&2; exit 1 ;;
esac
recsys validate "$bundle"
# Export stages mode 0700. Publish only the validated JSON files for the distinct runtime UID.
chmod 755 "$bundle"
chmod 644 "$bundle/catalog.json" "$bundle/neighbors.json" "$bundle/manifest.json"
printf 'ARB_BUNDLE_DIR=%s\n' "${bundle#/data/bundles/}"
