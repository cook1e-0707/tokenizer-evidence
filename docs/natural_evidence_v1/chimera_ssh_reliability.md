# Chimera SSH reliability note

This note records why hourly `natural_evidence_v1` checks sometimes fail on the
first `ssh chimera` attempt and how future automation should handle it.

## Observed failure

Several hourly runs initially failed with:

```text
ssh: Could not resolve hostname chimera.umb.edu: nodename nor servname provided, or not known
```

A later manual retry in the same session succeeded:

```text
chimerahead.umb.edu
2026-05-05T01:13:34Z
```

## Diagnosis

The local SSH alias is valid:

```text
Host chimera
    HostName chimera.umb.edu
    User guanjie.lin001
    IdentityFile ~/.ssh/chimera_ed25519
```

Current DNS resolution is also valid:

```text
chimera.umb.edu.      CNAME   chimerahead.umb.edu.
chimerahead.umb.edu.  A       158.121.247.54
```

The repeated failure mode is therefore not a bad SSH key, bad Slurm state, or
missing Chimera account. It is an intermittent local DNS/resolver cold-start
failure for `chimera.umb.edu` at the beginning of an hourly automation run. The
default SSH configuration makes this brittle because `ssh -G chimera` reports
`connectionattempts 1`, so a single transient DNS failure ends the check before
authentication is attempted.

## Required hourly preflight

Before marking Chimera access as failed, hourly automation must run a DNS warm-up
and retry SSH. Use a non-interactive command so the automation cannot hang:

```bash
dscacheutil -q host -a name chimera.umb.edu >/dev/null 2>&1 || true

for attempt in 1 2 3; do
  if ssh -o BatchMode=yes -o ConnectTimeout=10 chimera \
      'hostname; date -u "+%Y-%m-%dT%H:%M:%SZ"'; then
    break
  fi
  sleep 5
done
```

If all three attempts fail with the same DNS error, record
`CHIMERA_DNS_UNREACHABLE_AFTER_RETRY` and do not advance experiment gates. If any
attempt succeeds, continue with the normal remote Slurm/artifact checks:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 chimera \
  'squeue -u "$USER"; sacct -u "$USER" --format=JobID,JobName,State,Elapsed,ExitCode -S now-2days'
```

## Reporting rule

Do not report the first DNS failure as a scientific gate failure. It is an
operational preflight issue. Only after the retry block fails should the hourly
report mark Chimera job/artifact status as unverified.

