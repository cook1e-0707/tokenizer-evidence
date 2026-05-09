# Chimera SSH reliability note

This note records why hourly `natural_evidence_v1` checks repeatedly failed on
`ssh chimera` and how future automation should handle it.

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

From 2026-05-05T05:05Z through 2026-05-05T13:02Z, the failure became
persistent: the first `ssh chimera` attempt plus the documented DNS warm-up and
three non-interactive retries all failed while resolving `chimera.umb.edu`.

## Diagnosis

The original local SSH alias pointed at the CNAME endpoint:

```text
Host chimera
    HostName chimera.umb.edu
    User guanjie.lin001
    IdentityFile ~/.ssh/chimera_ed25519
```

DNS for that endpoint is a CNAME hop:

```text
chimera.umb.edu.      CNAME   chimerahead.umb.edu.
chimerahead.umb.edu.  A       158.121.247.54
```

The repeated failure mode is therefore not a bad SSH key, bad Slurm state, or
missing Chimera account. It is DNS brittleness on the `chimera.umb.edu` CNAME
path. Directly resolving and connecting to the canonical host
`chimerahead.umb.edu` succeeds, while the `chimera` alias previously forced every
automation check through the brittle CNAME. The default SSH configuration made
this worse because `ssh -G chimera` reported `connectionattempts 1` and no
`connecttimeout`, so a transient resolver failure ended the check before
authentication was attempted.

## Local fixes applied

On 2026-05-05T13:45Z, the local SSH alias was changed to target the canonical
host directly:

```text
Host chimera
    HostName chimerahead.umb.edu
    User guanjie.lin001
    IdentityFile ~/.ssh/chimera_ed25519
    IdentitiesOnly yes
    ConnectTimeout 10
    ConnectionAttempts 3
```

Verification after the fix:

```text
ssh -G chimera -> hostname chimerahead.umb.edu
ssh -o BatchMode=yes chimera 'hostname' -> chimerahead.umb.edu
```

On 2026-05-06T15:08Z, hourly reports still showed intermittent resolver failure
for the canonical hostname itself:

```text
ssh -G chimera -> hostname chimerahead.umb.edu
ssh chimera -> Could not resolve hostname chimerahead.umb.edu
```

Manual testing in the same worktree showed that SSH via the fixed Chimera
head-node IP succeeds, so the alias was updated to bypass DNS for connection
setup while retaining host-key checking against the canonical hostname:

```text
Host chimera
    HostName 158.121.247.54
    HostKeyAlias chimerahead.umb.edu
    User guanjie.lin001
    IdentityFile ~/.ssh/chimera_ed25519
    IdentitiesOnly yes
    ConnectTimeout 10
    ConnectionAttempts 3
```

Verification after the IP-pinned fix:

```text
ssh -G chimera -> hostname 158.121.247.54
ssh -G chimera -> hostkeyalias chimerahead.umb.edu
ssh -o BatchMode=yes chimera 'hostname' -> chimerahead.umb.edu
```

## Required hourly preflight

Before marking Chimera access as failed, hourly automation must first check that
the local alias either points at the IP-pinned host or, as a fallback, at the
canonical hostname:

```bash
ssh -G chimera | egrep '^(hostname 158\.121\.247\.54|hostname chimerahead\.umb\.edu)$'
```

If this check fails, repair the local SSH config before treating Chimera as
unreachable. If the alias is IP-pinned, skip DNS warm-up and retry SSH directly.
If the alias still uses `chimerahead.umb.edu`, run a DNS warm-up and retry SSH.
Use a non-interactive command so the automation cannot hang:

```bash
dscacheutil -q host -a name chimerahead.umb.edu >/dev/null 2>&1 || true

for attempt in 1 2 3; do
  if ssh -o BatchMode=yes -o ConnectTimeout=10 chimera \
      'hostname; date -u "+%Y-%m-%dT%H:%M:%SZ"'; then
    break
  fi
  sleep 5
done
```

If all three attempts fail with the same DNS error after the alias check passes,
record `CHIMERA_DNS_UNREACHABLE_AFTER_CANONICAL_RETRY` and do not advance
experiment gates. If any attempt succeeds, continue with the normal remote
Slurm/artifact checks:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 chimera \
  'squeue -u "$USER"; sacct -u "$USER" --format=JobID,JobName,State,Elapsed,ExitCode -S now-2days'
```

## Reporting rule

Do not report the first DNS failure as a scientific gate failure. It is an
operational preflight issue. Only after the retry block fails should the hourly
report mark Chimera job/artifact status as unverified.
