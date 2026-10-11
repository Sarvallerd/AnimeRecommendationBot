#!/bin/sh
set -eu
recsys release prepare --release "${ARB_RELEASE:-current}" --data-dir /data
