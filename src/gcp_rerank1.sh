#!/usr/bin/env bash
# Reranker run 1 on a GCP L4 from the laptop (gcp_run5.sh with the reranker's files).
#
#   gcp_rerank1.sh hunt      # try every zone with L4 quota every 15 min (HOURS, default 6), then launch + watch
#   gcp_rerank1.sh create    # one g2 VM (MACHINE, default g2-standard-8: 1 L4) in ZONE
#   gcp_rerank1.sh launch    # wait for SSH and the driver, push, start; deletes the VM if any of it fails
#   gcp_rerank1.sh push      # scripts + mined2/{train,dev}.jsonl (galician-train-2 output, ~690 MB)
#   gcp_rerank1.sh start     # rerank1_vm.sh in the background on the VM (~3 h)
#   gcp_rerank1.sh watch     # every 10 min: on DONE or FAILED (or after MAX_HOURS, default 5) fetch, then delete
#   gcp_rerank1.sh status    # DONE / FAILED / still running, and the last log lines
#   gcp_rerank1.sh fetch     # rerank1/ (final, train_summary.json), rr1/summary.json, test_rerank1/ and logs
#   gcp_rerank1.sh delete    # deletes the VM and its disk (billing stops)
#
# The zone the VM got is kept in OUT/zone, so later commands find it. MINED (default
# work/kaggle/train2/mined2) and OUT (default work/gcp/rerank1) are under the repo root.
set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
WIP="$ROOT/src"
SELF="$WIP/gcp_rerank1.sh"
PROJECT=${PROJECT:?set PROJECT to your GCP project id}
VM=${VM:-galician-rerank1}
# The job lives in the home of whoever launched it (gcloud logs in with the local user name).
# From another computer, set VM_USER to that name.
HOST=${VM_USER:+$VM_USER@}$VM
MACHINE=${MACHINE:-g2-standard-8}
MINED=${MINED:-$ROOT/work/kaggle/train2/mined2}
OUT=${OUT:-$ROOT/work/gcp/rerank1}
mkdir -p "$OUT"
ZONE=${ZONE:-$(cat "$OUT/zone" 2>/dev/null || echo europe-west4-a)}
G=(--project "$PROJECT" --zone "$ZONE")
SSH=(--quiet --strict-host-key-checking=no)
# Cheapest regions first; europe-west2/3/6 cost 10-20 % more.
ZONES="europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c us-central1-a us-central1-b
us-central1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b us-west1-c
europe-west3-a europe-west3-b europe-west2-a europe-west2-b europe-west6-b europe-west6-c"

ts() { date "+%Y-%m-%d %H:%M"; }
on_vm() { gcloud compute ssh "$HOST" "${G[@]}" "${SSH[@]}" --command "$1"; }

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
    on_vm "mkdir -p job/mined2"
    gcloud compute scp "${G[@]}" "${SSH[@]}" "$WIP"/{rerank1_vm.sh,prep_rerank.py,train_reranker.py,train.py,eval_dog.py} "$HOST":job/
    gcloud compute scp "${G[@]}" "${SSH[@]}" "$MINED"/{train.jsonl,dev.jsonl} "$HOST":job/mined2/
    ;;
start)
    on_vm "cd job && (nohup bash rerank1_vm.sh > run.log 2>&1 < /dev/null &) ; sleep 5; cat run.log"
    ;;
watch)
    end=$((SECONDS + ${MAX_HOURS:-5} * 3600))
    while :; do
        s=$(on_vm "cd job; if [ -e DONE ]; then echo DONE; elif [ -e FAILED ]; then echo FAILED; else echo RUNNING; fi" 2>/dev/null || echo NO_SSH)
        echo "$(ts) $s $(on_vm "grep -h '^{' job/prep.log job/train.log 2>/dev/null | tail -1 | cut -c1-160" 2>/dev/null || true)"
        if [ "$s" = DONE ] || [ "$s" = FAILED ] || [ "$SECONDS" -ge "$end" ]; then
            # Delete only after the results are on this machine; a failed fetch keeps the VM
            # (and its billing) so nothing is lost. On 2026-11-06 a fetch that PuTTY refused
            # was followed by the delete, and reranker run 1's model went with the disk.
            if "$SELF" fetch "$s"; then
                "$SELF" delete
                echo "$(ts) finished: $s; results in $OUT; VM deleted"
            else
                echo "$(ts) finished: $s; FETCH FAILED, VM KEPT (still billing): fix and run '$0 fetch', then '$0 delete'"
                exit 1
            fi
            [ "$s" = DONE ]
            exit
        fi
        sleep 600
    done
    ;;
status)
    on_vm "cd job; ls DONE FAILED 2>/dev/null; tail -3 run.log; grep -h '^{' prep.log train.log 2>/dev/null | tail -4; tail -2 t0.log 2>/dev/null"
    ;;
fetch)
    # One archive and one remote source: gcloud on Windows copies with PuTTY's pscp, which
    # refuses several remote sources in one call. Fails unless the model came back whole.
    # rr1/groups.jsonl holds the teacher scores (2 GPU-hours), so later variants can skip that stage.
    on_vm "cd job && rm -f out.tgz && tar czf out.tgz --ignore-failed-read run.log prep.log train.log t0.log rr1/summary.json \
        rr1/groups.jsonl rerank1/final rerank1/train_summary.json test_rerank1 2>&1 | grep -v 'Cannot stat' || true; ls -l out.tgz"
    gcloud compute scp "${G[@]}" "${SSH[@]}" "$HOST":job/out.tgz "$OUT"/out.tgz
    tar xzf "$OUT"/out.tgz -C "$OUT"
    if [ "${2:-DONE}" = DONE ]; then [ -s "$OUT"/rerank1/final/model.safetensors ] || { echo "model missing in $OUT"; false; }; fi
    ;;
delete)
    gcloud compute instances delete "$VM" "${G[@]}" --quiet
    ;;
*)
    sed -n 2,15p "$0"
    exit 1
    ;;
esac
