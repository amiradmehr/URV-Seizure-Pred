#!/bin/bash
#SBATCH --job-name=sz2-report
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/report-%j.out
#SBATCH --error=logs/report-%j.out
#
# Final stage, submitted with --dependency=afterany so it runs whether the
# pipeline succeeded or died partway. Its job is to leave one file that says
# what actually happened, so a finished (or broken) run is legible without
# reconstructing it from four separate logs.
#
# Deliberately does NOT use `set -e`: every check below must run even when an
# earlier one finds nothing.

REPO_ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
cd "${REPO_ROOT}" || exit 1

REPORT="${REPO_ROOT}/outputs/RUN_REPORT.txt"
mkdir -p "${REPO_ROOT}/outputs"

{
echo "=========================================================================="
echo "SeizeIT2 pipeline run report"
echo "Generated: $(date)"
echo "=========================================================================="
echo
echo "--- SLURM stage outcomes ---"
# Job IDs recorded by the submitting session.
if [ -f .pipeline_jobids ]; then
    for jobid in $(tr ' ' '\n' < .pipeline_jobids); do
        sacct -j "${jobid}" --noheader --format=JobID%14,JobName%14,State%20,Elapsed%12 2>/dev/null \
            | head -1
    done
else
    echo "(no .pipeline_jobids file; listing this user's recent jobs)"
    sacct -u "${USER}" --starttime=today \
        --format=JobID%14,JobName%14,State%20,Elapsed%12 2>/dev/null | head -30
fi

echo
echo "--- Dataset ---"
MANIFEST="data/interim/manifests/processed_shard_manifest.csv"
if [ -f "${MANIFEST}" ]; then
    echo "shards: $(( $(wc -l < "${MANIFEST}") - 1 ))"
    echo "raw       : $(du -sh data/raw 2>/dev/null | cut -f1)"
    echo "interim   : $(du -sh data/interim 2>/dev/null | cut -f1)"
    echo "processed : $(du -sh data/processed 2>/dev/null | cut -f1)"
else
    echo "MISSING: ${MANIFEST} — the build did not finish."
fi

VALIDATION_SUMMARY="data/interim/manifests/validation_summary.csv"
if [ -f "${VALIDATION_SUMMARY}" ]; then
    echo "dataset validation: PASSED (${VALIDATION_SUMMARY} written)"
else
    echo "dataset validation: DID NOT COMPLETE"
fi

echo
echo "--- Training ---"
TRAIN_METRICS="outputs/models/eegnet_mean_pool/metrics.json"
if [ -f "${TRAIN_METRICS}" ]; then
    python3 -c "
import json
m = json.load(open('${TRAIN_METRICS}'))
h = m.get('history', [])
print(f\"epochs completed : {len(h)}\")
print(f\"best val AP      : {m.get('best_validation_average_precision')}\")
if h:
    b = max(h, key=lambda e: e['validation_average_precision'])
    print(f\"best epoch       : {b['epoch']}\")
    print(f\"  train_loss     : {b['train_loss']:.4f}\")
    print(f\"  val_roc_auc    : {b['validation_roc_auc']:.4f}\")
    print(f\"  val_avg_prec   : {b['validation_average_precision']:.4f}\")
" 2>/dev/null || echo "(metrics.json present but unreadable)"
    ls -la outputs/models/eegnet_mean_pool/ 2>/dev/null
else
    echo "MISSING: ${TRAIN_METRICS} — training did not produce metrics."
fi

echo
echo "--- Evaluation ---"
for split in validation test; do
    METRICS="outputs/evaluation/${split}/metrics.json"
    echo "[${split}]"
    if [ -f "${METRICS}" ]; then
        python3 -c "
import json
m = json.load(open('${METRICS}'))
for k in ('patients','decisions','target_seizures','prevalence',
          'average_precision','roc_auc','brier_score'):
    print(f'  {k:18s}: {m.get(k)}')
print('  sensitivity under alarm budget:')
for b in m.get('alarm_budgets', []):
    if b.get('achievable'):
        print(f\"    <={b['false_alarm_budget_per_hour']:>4} /h : \"
              f\"seizure {b['seizure_sensitivity']:.3f} \"
              f\"({b['seizures_detected']}/{b['seizures_total']}), \"
              f\"decision {b['decision_sensitivity']:.3f}\")
    else:
        print(f\"    <={b['false_alarm_budget_per_hour']:>4} /h : not achievable\")
" 2>/dev/null || echo "  (metrics.json present but unreadable)"
        echo "  files: $(ls outputs/evaluation/${split}/ 2>/dev/null | tr '\n' ' ')"
    else
        echo "  not evaluated"
    fi
done

echo
echo "--- Storage ---"
echo "workspace: $(ws_list 2>/dev/null | grep -A2 'id: seizeit2' | tr '\n' ' ')"
df -h /scratch4 2>/dev/null | tail -1
echo
echo "Extend the scratch workspace before it expires:"
echo "    ws_allocate -x -F workspace -n seizeit2 -d 30"
echo "=========================================================================="
} 2>&1 | tee "${REPORT}"

echo
echo "Report written to ${REPORT}"
