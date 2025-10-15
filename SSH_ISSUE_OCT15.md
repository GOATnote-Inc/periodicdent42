# SSH Connection Issue: GPU Instance
## October 15, 2025 - 18:20 UTC

---

## Issue

**Symptom**: SSH connection to `cudadent42-l4-dev` failing with exit code 255  
**Instance Status**: RUNNING (verified via `gcloud compute instances describe`)  
**Duration**: ~10 minutes

---

## Diagnostics

```bash
$ gcloud compute instances describe cudadent42-l4-dev --zone=us-central1-a --format="value(status)"
RUNNING

$ gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a --command "echo test"
ERROR: (gcloud.compute.ssh) [/usr/bin/ssh] exited with return code [255].
```

---

## Possible Causes

1. **SSH daemon restart** (from earlier session)
2. **Firewall rule change** (unlikely, was working 30 min ago)
3. **SSH key mismatch** (project metadata vs instance)
4. **Network routing issue** (IAP tunnel or external IP)

---

## Workarounds Attempted

- ✅ Wait 5 seconds and retry → Still fails
- ✅ Verify instance is RUNNING → Confirmed
- ⏳ Restart instance → Not attempted (would lose session state)
- ⏳ Use IAP tunnel → Not attempted

---

## Recommended Actions

### Option A: Restart SSH Daemon (Preferred)
```bash
gcloud compute reset-windows-password cudadent42-l4-dev --zone=us-central1-a --user=kiteboard
# Then retry SSH
```

### Option B: Restart Instance (Last Resort)
```bash
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a
sleep 30
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
sleep 60
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a
```

### Option C: Use Serial Console
```bash
gcloud compute connect-to-serial-port cudadent42-l4-dev --zone=us-central1-a
# Manual debug
```

---

## Files Prepared (Ready for Deployment)

While SSH is unavailable, all files have been prepared locally:

- ✅ `scripts/run_gpu_validation.sh` (6-stage validation suite)
- ✅ `cudadent42/bench/tests/oracles/tile_oracle_v3.py` (sanitizer harness)
- ✅ `.github/workflows/guard_no_gpu_stop.yml` (CI protection)
- ✅ `scripts/bench_s512_tc_vs_sdpa.py` (stream variant)
- ✅ All documentation updated

**When SSH is restored**: Simply run `./scripts/run_gpu_validation.sh` to collect all artifacts.

---

## Impact Assessment

**Evidence Collection**: ⚠️ **BLOCKED** (cannot access GPU)  
**Code Quality**: ✅ **COMPLETE** (all scripts prepared)  
**CI Protection**: ✅ **ACTIVE** (guard workflow committed)  
**Cost**: ~$0.10/hour (instance still running)

---

## Next Steps

1. **Immediate**: Try Option A (reset SSH keys)
2. **If fails**: Use Option B (restart instance)
3. **Once SSH works**: Run `./scripts/run_gpu_validation.sh`
4. **Collect**: Review artifacts and commit

---

## Preventive Measures

To avoid this in future sessions:

```bash
# Add to GPU keepalive script
while true; do
  systemctl status sshd >/dev/null 2>&1 || systemctl restart sshd
  sleep 300
done
```

---

**Status**: ⚠️ **SSH BLOCKED** - Evidence collection paused  
**Time Lost**: ~10 minutes  
**Next Action**: Restart instance or wait for SSH recovery

**Date**: October 15, 2025 18:20 UTC  
**Session**: GPU validation deployment

