#!/usr/bin/env bash
set -euo pipefail

CHECKPOINTS_JSON=$(uv run tinker -f json checkpoint list)

KEEP=$(echo "$CHECKPOINTS_JSON" | jq -r '
  [.checkpoints | sort_by(.time) | reverse | .[] | select(.checkpoint_id == "sampler_weights/final")][0].tinker_path
')

if [ -z "$KEEP" ] || [ "$KEEP" = "null" ]; then
  echo "ERROR: No sampler_weights/final checkpoint found. Aborting."
  exit 1
fi

TO_DELETE=$(echo "$CHECKPOINTS_JSON" | jq -r --arg keep "$KEEP" '
  .checkpoints[] | select(.tinker_path != $keep) | .tinker_path
')

if [ -z "$TO_DELETE" ]; then
  echo "Nothing to delete. Only the latest sampler checkpoint exists:"
  echo "  $KEEP"
  exit 0
fi

echo "KEEPING:"
echo "  $KEEP"
echo ""
echo "DELETING:"
echo "$TO_DELETE" | sed 's/^/  /'
echo ""
read -p "Proceed? (y/N) " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
  # Get unique run IDs from checkpoints to delete
  # (can't pass tinker:// paths as positional args due to CLI bug)
  RUN_IDS=$(echo "$TO_DELETE" | sed 's|tinker://||;s|/.*||' | sort -u)

  KEEP_RUN=$(echo "$KEEP" | sed 's|tinker://||;s|/.*||')

  for run_id in $RUN_IDS; do
    if [ "$run_id" = "$KEEP_RUN" ]; then
      uv run tinker checkpoint delete -y --run-id "$run_id" --type weights </dev/null
    else
      uv run tinker checkpoint delete -y --run-id "$run_id" </dev/null
    fi
  done

  echo "Done."
else
  echo "Aborted."
fi
