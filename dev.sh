#!/usr/bin/env bash
set -e

lang="$1"

case "$lang" in
    python)
        nix develop .#python
        ;;
    rust)
	nix develop .#rust
	;;
    *)
        echo "Error: unknown env: '$lang'"
        exit 1
        ;;
esac
