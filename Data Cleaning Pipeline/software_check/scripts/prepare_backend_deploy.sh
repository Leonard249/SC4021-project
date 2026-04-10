#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEPLOY_DIR="${DEPLOY_DIR:-/tmp/software_check_deploy}"
INSTANCE_NAME="${INSTANCE_NAME:-instance-sc4021}"
PROJECT_ID="${PROJECT_ID:-project-9ec2f031-335c-4ab1-930}"
ZONE="${ZONE:-asia-south1-c}"
REMOTE_PATH="${REMOTE_PATH:-~}"
UPLOAD=0

usage() {
  cat <<'EOF'
Prepare a backend-only deploy bundle for the software_check app.

By default this script stages the backend bundle locally at:
  /tmp/software_check_deploy

Optional upload:
  --upload
    Upload the staged bundle to the GCP VM with gcloud compute scp.

Optional overrides:
  --deploy-dir /tmp/software_check_deploy
  --instance instance-sc4021
  --project project-9ec2f031-335c-4ab1-930
  --zone asia-south1-c
  --remote-path ~

Examples:
  bash scripts/prepare_backend_deploy.sh

  bash scripts/prepare_backend_deploy.sh --upload

  INSTANCE_NAME=my-vm ZONE=asia-south1-c \
    bash scripts/prepare_backend_deploy.sh --upload
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upload)
      UPLOAD=1
      shift
      ;;
    --deploy-dir)
      DEPLOY_DIR="$2"
      shift 2
      ;;
    --instance)
      INSTANCE_NAME="$2"
      shift 2
      ;;
    --project)
      PROJECT_ID="$2"
      shift 2
      ;;
    --zone)
      ZONE="$2"
      shift 2
      ;;
    --remote-path)
      REMOTE_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

required_paths=(
  "$ROOT_DIR/backend"
  "$ROOT_DIR/profiles.json"
  "$ROOT_DIR/selected_data.json"
  "$ROOT_DIR/db_bryan.json"
  "$ROOT_DIR/db_ananya.json"
  "$ROOT_DIR/db_ryan.json"
  "$ROOT_DIR/db_leonard.json"
  "$ROOT_DIR/scripts/build_reviewer_dbs.py"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$DEPLOY_DIR"

rm -rf \
  "$DEPLOY_DIR/backend" \
  "$DEPLOY_DIR/scripts"

rm -f \
  "$DEPLOY_DIR"/db_*.json \
  "$DEPLOY_DIR/profiles.json" \
  "$DEPLOY_DIR/selected_data.json"

cp -R "$ROOT_DIR/backend" "$DEPLOY_DIR/"
cp "$ROOT_DIR"/db_*.json "$DEPLOY_DIR/"
cp "$ROOT_DIR/profiles.json" "$DEPLOY_DIR/"
cp "$ROOT_DIR/selected_data.json" "$DEPLOY_DIR/"
cp -R "$ROOT_DIR/scripts" "$DEPLOY_DIR/"

find "$DEPLOY_DIR" \
  \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.DS_Store' \) \
  -exec rm -rf {} +

echo "Staged backend deploy bundle at:"
echo "  $DEPLOY_DIR"
echo
echo "Contents:"
find "$DEPLOY_DIR" -maxdepth 2 -mindepth 1 | sort

if [[ "$UPLOAD" -eq 1 ]]; then
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is not installed or not on PATH." >&2
    exit 1
  fi

  echo
  echo "Uploading bundle to ${INSTANCE_NAME}:${REMOTE_PATH}"
  gcloud compute scp --recurse \
    "$DEPLOY_DIR" \
    "${INSTANCE_NAME}:${REMOTE_PATH}" \
    --project="$PROJECT_ID" \
    --zone="$ZONE"

  echo
  echo "Upload complete."
  echo "On the VM, the bundle should be at:"
  echo "  ${REMOTE_PATH}/$(basename "$DEPLOY_DIR")"
fi
