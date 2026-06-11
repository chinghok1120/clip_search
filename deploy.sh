#!/usr/bin/env bash
#
# Deploy the PN code layer: sync pn/ -> target location.
#
# The artifact layer (venv, model weights, embeddings, bench data) is EXCLUDED via
# pn/.deployignore, so it is never overwritten and rsync --delete never touches it.
#
# Dry-run by default — shows exactly what would change and what --delete would remove.
# Pass --go to actually apply.
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: ./deploy.sh [OPTIONS]

Deploy the PN code layer (pn/) to target location via rsync.

OPTIONS:
  --local         Local deploy mode (on Jetson itself, repo → deployment folder)
  --go            Actually apply changes (default is dry-run preview)
  -h, --help      Show this help message

REMOTE DEPLOY (from dev machine to Jetson):
  ./deploy.sh                      # Preview what would change
  ./deploy.sh --go                 # Apply deployment
  PN_HOST=user@host ./deploy.sh --go    # Deploy to custom host

LOCAL DEPLOY (on Jetson itself):
  ./deploy.sh --local              # Preview (creates separate deploy folder)
  ./deploy.sh --local --go         # Apply deployment
  PN_HOST=localhost ./deploy.sh --go    # Alternative syntax

ENVIRONMENT VARIABLES:
  PN_HOST         Target host (default: superrx@210.17.139.83)
                  Set to 'localhost' for local deploy
  PN_DIR          Target directory relative to $HOME (default: clip_search)

CONFLICT DETECTION:
  If git repo and deploy target are the same, target is automatically
  adjusted to ${PN_DIR}_deploy to prevent overwrite.

EXAMPLES:
  # Remote deploy from dev machine
  ./deploy.sh --go

  # Local deploy on Jetson (auto-creates ~/clip_search_deploy/)
  cd ~/clip_search && ./deploy.sh --local --go

  # Custom target
  PN_DIR=my_pn_folder ./deploy.sh --local --go

EXCLUDED (artifact layer, never synced):
  venv/, *.pth, *.engine, embeddings/, benchmark data
  See pn/.deployignore for full list

EOF
}

# Parse arguments
LOCAL_DEPLOY=false
APPLY=false
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      show_help
      exit 0
      ;;
    --local) LOCAL_DEPLOY=true ;;
    --go) APPLY=true ;;
  esac
done

REPO="$(cd "$(dirname "$0")" && pwd)"
PN_HOST="${PN_HOST:-superrx@210.17.139.83}"
PN_DIR="${PN_DIR:-clip_search}"     # relative to $HOME

# Detect local deploy mode
if [ "$PN_HOST" = "localhost" ] || [ "$LOCAL_DEPLOY" = true ]; then
  LOCAL_DEPLOY=true
  TARGET="$HOME/$PN_DIR"

  # SAFETY: prevent deploying into the same directory as the git repo
  if [ "$REPO" = "$TARGET" ]; then
    echo ">> CONFLICT: Git repo and deploy target are the same: $REPO"
    echo ">> Auto-adjusting deploy target to avoid overwrite..."
    PN_DIR="${PN_DIR}_deploy"
    TARGET="$HOME/$PN_DIR"
    echo ">> New target: $TARGET"
    echo ""
  fi

  DEPLOY_TYPE="LOCAL"
  # For rsync: local paths don't need host: prefix
  RSYNC_TARGET="$TARGET/"
else
  DEPLOY_TYPE="REMOTE"
  # For rsync: remote paths need host: prefix
  RSYNC_TARGET="$PN_HOST:$PN_DIR/"
  TARGET="$PN_HOST:~/$PN_DIR"  # For display only
fi

FLAGS=(-az --delete --itemize-changes --exclude-from="$REPO/pn/.deployignore")
if [ "$APPLY" = true ]; then
  echo ">> $DEPLOY_TYPE DEPLOY: pn/ -> $TARGET"
else
  echo ">> DRY-RUN (pass --go to apply): pn/ -> $TARGET"
  FLAGS+=(--dry-run)
fi

rsync "${FLAGS[@]}" "$REPO/pn/" "$RSYNC_TARGET"

if [ "$APPLY" = true ]; then
  echo ">> deployed. Next: cd ~/$PN_DIR && ./setup/setup_model.sh"
else
  echo ">> dry-run only."
fi
