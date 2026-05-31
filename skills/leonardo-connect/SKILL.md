---
name: leonardo-connect
description: >-
  Connect to the CINECA Leonardo HPC cluster (SSH password auth via expect,
  credentials from .env) and run a connectivity + environment smoke test.
  Use when the user says "connect to leonardo", "ssh leonardo", "leonardo smoke
  test", "submit a job to leonardo", "run on the cluster", "check the cluster",
  or mentions CINECA / SLURM / boost_usr_prod / the s_tra_ncc reservation.
  Also covers pixi bootstrap, file transfer (scp), and SLURM job templates.
---

# Leonardo Connect (CINECA HPC)

SSH helper + smoke test for the Leonardo A100 cluster. Password auth is wrapped
with `expect`; credentials are read from `.env` (`LEONARDO_USERNAME`,
`LEONARDO_PASSWORD`). Never print the password.

## Quick start

```bash
LEO=~/.pi/agent/skills/leonardo-connect/scripts/leo.sh

$LEO smoke                       # connectivity + environment check
$LEO run "squeue --me"           # run any command on a login node (fresh SSH+auth each call)
$LEO run "sbatch job_gpu.sh"     # submit a job
$LEO shell                       # interactive shell (password auto-sent)
```

Login nodes (any works): `login01/02/05/07-ext.leonardo.cineca.it`.
Override with `LEONARDO_HOST=login05-ext.leonardo.cineca.it`.

> `leo.sh run` opens a NEW SSH + password every call (slow if you poll a lot). For repeated
> commands use the **persistent tmux session** below. For uploads use the **datamover** (scp
> to login nodes is blocked).

## Persistent session (recommended for repeated commands)

`leo_tmux.sh` keeps ONE authenticated SSH alive in a local tmux pane — log in once, then run
many commands with **no re-auth** (≈instant vs ~3s/call for `leo.sh run`).

```bash
LT=~/.pi/agent/skills/leonardo-connect/scripts/leo_tmux.sh
$LT start                        # create tmux session 'leo', ssh+auth once
$LT run "squeue --me; cd ~/zero-one && ls"   # run + capture output (no re-auth)
$LT jobs                         # on-demand: squeue --me + tail newest ~/zero-one/slurm-*.out
$LT watch                        # live self-refreshing monitor in window 'watch' (10s)
$LT peek                         # snapshot the watch window anytime
$LT attach                       # view live in your terminal (Ctrl-b d to detach)
$LT kill                         # tear down
```

Implementation notes: `run` uses `stty -echo` + unique BEG/END markers and `tmux
capture-pane` to return clean output; `watch` writes `$HOME/jobwatch.sh` (a `sleep`-loop, ~0
CPU so it survives the login-node 10-min limit) and runs it in its own authenticated window.
Markers are unique per call, so concurrent reads don't collide — but avoid running `$LT run`
and a manual command in window 0 at the same time.

## File transfer (scp via datamover — NOT login nodes)

scp to `login*-ext` is **blocked** (silent fail). You MUST use a **datamover** host and an
**ABSOLUTE** remote path (datamover cwd is not your $HOME):

```bash
# datamovers: data.leonardo.cineca.it  (alias) | dmover1..4.leonardo.cineca.it
HOMEABS=/leonardo/home/usertrain/<user>
# upload (expect wraps the password; never echo it):
expect -c 'set timeout 600; spawn scp -o StrictHostKeyChecking=accept-new \
  -o PubkeyAuthentication=no -o PreferredAuthentications=password \
  LOCAL <user>@dmover1.leonardo.cineca.it:'"$HOMEABS"'/REMOTE; \
  expect -re {(?i)password:} {send -- "$env(LEONARDO_PASSWORD)\r"; exp_continue} eof'
```

Practical pattern for many files: `tar -czf /tmp/x.tar.gz <files>` → scp the tarball to
`$HOMEABS/x.tar.gz` via dmover → `$LT run "cd ~/zero-one && tar -xzf ~/x.tar.gz"`.
Gotcha: the expect wrapper evaluates `$HOME` as a Tcl var — pass a literal absolute path or a
bare relative name (lands in $HOME), never the string `$HOME`.

## Common failure modes (learned)

- **scp to login node** silently fails → use a datamover + absolute path (above).
- **`ModuleNotFoundError: research/leonardo`** in jobs → set `PYTHONPATH=$HOME/zero-one:$HOME/zero-one/src` (repo root for `research`/`leonardo`, src for `uniqa`).
- **Compute-node OOM** with big LLM prompts → lower batch size (64GB A100 fits ~48 sessions at ~2.5k-token prompts, OOMs at 100) and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Job killed `DUE TO TIME LIMIT`** → raise `#SBATCH --time` and/or write results incrementally so partials survive.
- **`scancel` mid-training** loses the final adapter (saved only at the end) → use the last `checkpoint-N/` (it has `adapter_config.json` + `adapter_model.safetensors`).

## Verified environment (account <user>, group corsi)

| Item | Value |
|------|-------|
| `$HOME` | `/leonardo/home/usertrain/<user>` — 50 GB |
| `$SCRATCH` | `/leonardo_scratch/large/usertrain/<user>` — large, **use for big files** (purged after 40 days) |
| `$PUBLIC` | `/leonardo/pub/usertrain/<user>` — 50 GB, share between users |
| SLURM | 23.11.10 · partition `boost_usr_prod` |
| Reservation | `s_tra_ncc` · account `euhpc_d30_031` · ends **2026-05-31 12:00** |
| modules | Environment Modules 5.2.0 |
| pixi | not installed by default → `scripts/bootstrap_pixi.sh` |

## Hard rules (login nodes)

- **10-minute CPU limit** on login-node processes. For longer interactive work:
  `srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash`
- **Compute nodes have NO internet.** Download on login nodes, or set the proxy
  env vars inside the SLURM script (see `scripts/job_gpu.sh`). Proxy is
  low-bandwidth only and restarts every ~10 min (TCP drops).
- `$HOME` is 50 GB — keep datasets / checkpoints in `$SCRATCH`.

## SLURM templates

`scripts/job_gpu.sh` — parameterized 1/2/4-GPU job using the hackathon
reservation. Submit with `$LEO put` then `$LEO run "sbatch job_gpu.sh"`.

Useful commands: `sbatch job.sh` · `squeue --me` · `tail -c +0 -f slurm-<id>.out`
· `scancel <id>` · `srun --overlap --pty --jobid=<id> bash`.

## pixi bootstrap

`scripts/bootstrap_pixi.sh` installs pixi and inits a project. Run it once via
`$LEO run "$(cat scripts/bootstrap_pixi.sh)"` or upload + execute.

## Troubleshooting

- `__LEO_AUTH_FAIL__` → wrong creds in `.env`.
- `__LEO_NET_FAIL__` → host unreachable (try another login node / check VPN).
- Host-key / post-quantum warnings are benign (server-side).
