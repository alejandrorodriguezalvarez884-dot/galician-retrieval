#!/usr/bin/env bash
# Push the seed and SINAI-ablation kernels once the weekly GPU quota has reset, follow them,
# and download their small result files (not the model weights) to work/kaggle/<slug>/.
#
#   OWNER=<kaggle user> kaggle/launch_variance.sh   # from the repo root, after
#   python src/build_notebooks.py --variance-kernels kaggle/kernels
#
# MIN_HOURS (default 8): GPU hours that must be left before pushing (the three need ~6 h).
set -uo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
HERE="$ROOT/kaggle/kernels"
OWNER=${OWNER:?set OWNER to your Kaggle user name}
SLUGS="galician-nosinai galician-seed1 galician-seed2"
ts() { date "+%Y-%m-%d %H:%M"; }

[ "$(kaggle config view 2>/dev/null | awk '/username/ {print $3}')" = "$OWNER" ] || { echo "$(ts) CLI is not signed in as $OWNER"; exit 1; }

while :; do
    left=$(kaggle quota 2>/dev/null | awk '$1 == "GPU" {gsub("h", "", $3); print int($3)}')
    echo "$(ts) GPU hours left: ${left:-?}"
    [ -n "$left" ] && [ "$left" -ge "${MIN_HOURS:-8}" ] && break
    sleep 600
done

for s in $SLUGS; do
    until out=$(kaggle kernels push -p "$HERE/$s" --accelerator NvidiaTeslaT4 2>&1) && echo "$out" | grep -qi "successfully pushed"; do
        echo "$(ts) push $s failed, retrying in 10 min: $(echo "$out" | tail -1)"
        sleep 600
    done
    echo "$(ts) pushed $s: $(echo "$out" | tail -1)"
    sleep 60
done

end=$((SECONDS + 14 * 3600))
while [ "$SECONDS" -lt "$end" ]; do
    pending=0
    for s in $SLUGS; do
        st=$(kaggle kernels status "$OWNER/$s" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | head -1)
        echo "$(ts) $s ${st:-unknown}"
        case "$st" in
            *COMPLETE|*ERROR|*CANCEL*) [ -e "$ROOT/work/kaggle/$s/.fetched" ] || {
                mkdir -p "$ROOT/work/kaggle/$s"
                kaggle kernels output "$OWNER/$s" -p "$ROOT/work/kaggle/$s" \
                    --file-pattern '(results\.jsonl|runs/.*\.jsonl|train_summary\.json|\.log)$' > /dev/null 2>&1 \
                    && touch "$ROOT/work/kaggle/$s/.fetched" && echo "$(ts) fetched $s results"
                kaggle kernels logs "$OWNER/$s" > "$ROOT/work/kaggle/$s/kernel.log" 2>&1 || true
            } ;;
            *) pending=1 ;;
        esac
    done
    [ "$pending" = 0 ] && { echo "$(ts) all done"; exit 0; }
    sleep 1200
done
echo "$(ts) gave up waiting after 14 h"
exit 1
