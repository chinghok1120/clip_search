#!/usr/bin/env bash
#
# Deploy the PN code layer to the Jetson: sync pn/ -> PN:~/clip_search/.
#
# The artifact layer (venv, model weights, embeddings, bench data) is EXCLUDED via
# pn/.deployignore, so it is never overwritten and rsync --delete never touches it.
#
# Dry-run by default — shows exactly what would change and what --delete would remove.
# Pass --go to actually apply.
#
#   ./deploy.sh            # preview (dry-run)
#   ./deploy.sh --go       # apply
#   PN_HOST=user@host ./deploy.sh --go
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PN_HOST="${PN_HOST:-superrx@210.17.139.83}"
PN_DIR="${PN_DIR:-clip_search}"     # relative to the remote $HOME

FLAGS=(-az --delete --itemize-changes --exclude-from="$REPO/pn/.deployignore")
if [ "${1:-}" = "--go" ]; then
  echo ">> SYNCING  pn/ -> $PN_HOST:~/$PN_DIR/"
else
  echo ">> DRY-RUN (pass --go to apply):  pn/ -> $PN_HOST:~/$PN_DIR/"
  FLAGS+=(--dry-run)
fi

rsync "${FLAGS[@]}" "$REPO/pn/" "$PN_HOST:$PN_DIR/"

[ "${1:-}" = "--go" ] && echo ">> deployed. On the PN: cd ~/$PN_DIR && ./run_demo.sh" || echo ">> dry-run only."
