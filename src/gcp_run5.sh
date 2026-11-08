#!/usr/bin/env bash
# Run 5 on a GCP L4 from the laptop.
#
#   gcp_run5.sh hunt      # try every zone with L4 quota every 15 min (HOURS, default 6), then launch + watch
#   gcp_run5.sh create    # one g2 VM (MACHINE, default g2-standard-8: 1 L4) in ZONE
#   gcp_run5.sh launch    # wait for SSH and the driver, push, start; deletes the VM if any of it fails
#   gcp_run5.sh push      # scripts + mined5/{train,dev}.jsonl (prep5.py output, ~440 MB)
#   gcp_run5.sh start     # run5_vm.sh in the background on the VM (~3 h)
#   gcp_run5.sh watch     # every 10 min: on DONE or FAILED (or after MAX_HOURS, default 5) fetch, then delete
#   gcp_run5.sh status    # DONE / FAILED / still running, and the last log lines
#   gcp_run5.sh fetch     # run5/final, train_summary.json, test_run5/, e_run5/ and logs
#   gcp_run5.sh delete    # deletes the VM and its disk (billing stops)
#
# The zone the VM got is kept in OUT/zone, so later commands find it. MINED5 (default
# work/kaggle/mined5) and OUT (default work/gcp/run5) are under the repo root.
set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
WIP="$ROOT/src"
SELF="$WIP/gcp_run5.sh"
PROJECT=${PROJECT:?set PROJECT to your GCP project id}
VM=${VM:-galician-run5}
MACHINE=${MACHINE:-g2-standard-8}
MINED5=${MINED5:-$ROOT/work/kaggle/mined5}
OUT=${OUT:-$ROOT/work/gcp/run5}
mkdir -p "$OUT"
ZONE=${ZONE:-$(cat "$OUT/zone" 2>/dev/null || echo europe-west4-a)}
G=(--project "$PROJECT" --zone "$ZONE")
SSH=(--quiet --strict-host-key-checking=no)
# Cheapest regions first; europe-west2/3/6 cost 10-20 % more.
ZONES="europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c us-central1-a us-central1-b
us-central1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b us-west1-c
europe-west3-a europe-west3-b europe-west2-a europe-west2-b europe-west6-b europe-west6-c"

ts() { date "+%Y-%m-%d %H:%M"; }
on_vm() { gcloud compute ssh "$VM" "${G[@]}" "${SSH[@]}" --command "$1"; }

create_vm() {  # zone machine
    gcloud compute instances create "$VM" --project "$PROJECT" --zone "$1" --machine-type "$2" \
        --maintenance-policy TERMINATE --provisioning-model STANDARD \
        --image-family common-cu129-ubuntu-2404-nvidia-580 --image-project deeplearning-platform-release \
        --boot-disk-size 100GB --boot-disk-type pd-balanced --metadata install-nvidia-driver=True
}

case "${1:-}" in
hunt)
    end=$((SECONDS + ${HOURS:-6} * 3600))
    while [ "$SECONDS" -lt "$end" ]; do
        for machine in g2-standard-8 g2-standard-4; do
            for z in $ZONES; do
                if create_vm "$z" "$machine" > "$OUT/create.log" 2>&1 ||
                    gcloud compute instances describe "$VM" --project "$PROJECT" --zone "$z" > /dev/null 2>&1; then
                    echo "$z" > "$OUT/zone"
                    echo "$(ts) created $machine in $z"
                    ZONE=$z "$SELF" launch && exec env ZONE="$z" "$SELF" watch
                    exit 1
                fi
                echo "$(ts) $machine $z: $(grep -o -E 'reason: [a-z_]+' "$OUT/create.log" | head -1)"
            done
        done
        sleep 900
    done
    echo "$(ts) no L4 found in ${HOURS:-6} h; nothing was created"
    exit 1
    ;;
create)
    create_vm "$ZONE" "$MACHINE" && echo "$ZONE" > "$OUT/zone"
    ;;
launch)
    trap 'echo "$(ts) launch failed, deleting the VM"; "$SELF" delete' ERR
    for i in $(seq 40); do
        on_vm "nvidia-smi -L" > "$OUT/launch.log" 2>&1 && break
        [ "$i" = 40 ] && false
        sleep 30
    done
    cat "$OUT/launch.log"
    "$SELF" push
    "$SELF" start
    echo "$(ts) started"
    ;;
push)
    on_vm "mkdir -p run5/mined5"
    gcloud compute scp "${G[@]}" "${SSH[@]}" "$WIP"/{run5_vm.sh,train.py,eval_dog.py,eval_extra.py,gen_queries.py} "$VM":run5/
    gcloud compute scp "${G[@]}" "${SSH[@]}" "$MINED5"/{train.jsonl,dev.jsonl} "$VM":run5/mined5/
    ;;
start)
    on_vm "cd run5 && (nohup bash run5_vm.sh > run.log 2>&1 < /dev/null &) ; sleep 5; cat run.log"
    ;;
watch)
    end=$((SECONDS + ${MAX_HOURS:-5} * 3600))
    while :; do
        s=$(on_vm "cd run5; if [ -e DONE ]; then echo DONE; elif [ -e FAILED ]; then echo FAILED; else echo RUNNING; fi" 2>/dev/null || echo NO_SSH)
        echo "$(ts) $s $(on_vm "grep -h '^{' run5/train.log 2>/dev/null | tail -1 | cut -c1-160" 2>/dev/null || true)"
        if [ "$s" = DONE ] || [ "$s" = FAILED ] || [ "$SECONDS" -ge "$end" ]; then
            "$SELF" fetch || true
            "$SELF" delete
            echo "$(ts) finished: $s; VM deleted"
            [ "$s" = DONE ]
            exit
        fi
        sleep 600
    done
    ;;
status)
    on_vm "cd run5; ls DONE FAILED 2>/dev/null; tail -3 run.log; grep -h '^{' train.log 2>/dev/null | tail -4; tail -2 t0.log e1.log 2>/dev/null"
    ;;
fetch)
    gcloud compute scp "${G[@]}" "${SSH[@]}" "$VM":run5/{run.log,train.log,t0.log,e1.log} "$OUT"/ || true
    gcloud compute scp --recurse "${G[@]}" "${SSH[@]}" "$VM":run5/run5 "$VM":run5/test_run5 "$VM":run5/e_run5 "$OUT"/
    ;;
delete)
    gcloud compute instances delete "$VM" "${G[@]}" --quiet
    ;;
*)
    sed -n 2,15p "$0"
    exit 1
    ;;
esac
