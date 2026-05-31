#!/usr/bin/env bash
# Leonardo (CINECA) SSH helper — password auth via expect, creds from .env.
#
# Usage:
#   leo.sh smoke                 # run connectivity + environment smoke test
#   leo.sh run "<remote cmds>"   # run arbitrary command(s) on a login node
#   leo.sh shell                 # open interactive shell (forwards to ssh)
#   leo.sh put <local> <remote>  # scp upload
#   leo.sh get <remote> <local>  # scp download
#
# Env / .env keys:
#   LEONARDO_USERNAME, LEONARDO_PASSWORD   (required)
#   LEONARDO_HOST                          (optional, default login01-ext...)
#
# Login nodes (any works): login01/02/05/07-ext.leonardo.cineca.it
set -uo pipefail

# ── locate + load .env (search cwd, then repo root) ───────────────────────────
_find_env() {
  local d="$PWD"
  for _ in 1 2 3 4 5 6; do
    [ -f "$d/.env" ] && { echo "$d/.env"; return 0; }
    d="$(dirname "$d")"; [ "$d" = "/" ] && break
  done
  # fallback: known project location
  [ -f "$HOME/Desktop/hobby-dev/zero-one/.env" ] && { echo "$HOME/Desktop/hobby-dev/zero-one/.env"; return 0; }
  return 1
}

if [ -z "${LEONARDO_USERNAME:-}" ] || [ -z "${LEONARDO_PASSWORD:-}" ]; then
  ENV_FILE="$(_find_env || true)"
  if [ -n "${ENV_FILE:-}" ]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
  fi
fi

: "${LEONARDO_USERNAME:?set LEONARDO_USERNAME in .env}"
: "${LEONARDO_PASSWORD:?set LEONARDO_PASSWORD in .env}"
HOST="${LEONARDO_HOST:-login01-ext.leonardo.cineca.it}"

SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o NumberOfPasswordPrompts=1 -o PreferredAuthentications=password -o PubkeyAuthentication=no"

# ── expect wrapper: run a remote command string, stream output ────────────────
_remote() {
  local remote_cmd="$1"
  LEO_USER="$LEONARDO_USERNAME" LEO_PASS="$LEONARDO_PASSWORD" \
  LEO_HOST="$HOST" LEO_SSHOPTS="$SSH_OPTS" LEO_CMD="$remote_cmd" \
  expect -f - <<'EXP'
log_user 1
set timeout 60
set user  $env(LEO_USER)
set pass  $env(LEO_PASS)
set host  $env(LEO_HOST)
set cmd   $env(LEO_CMD)
set opts  [split $env(LEO_SSHOPTS) " "]
spawn ssh {*}$opts $user@$host $cmd
expect {
  -re {(?i)password:} { send -- "$pass\r"; exp_continue }
  -re {(?i)passcode:} { send -- "$pass\r"; exp_continue }
  -re {(?i)permission denied} { puts "\n__LEO_AUTH_FAIL__"; exit 3 }
  -re {(?i)could not resolve|connection refused|timed out} { puts "\n__LEO_NET_FAIL__"; exit 4 }
  eof
}
catch wait result
exit [lindex $result 3]
EXP
}

cmd="${1:-smoke}"; shift || true

case "$cmd" in
  run)
    [ $# -ge 1 ] || { echo "usage: leo.sh run \"<remote cmd>\""; exit 2; }
    _remote "$*"
    ;;

  shell)
    echo "Opening interactive shell on $HOST (password auto-sent)..."
    LEO_USER="$LEONARDO_USERNAME" LEO_PASS="$LEONARDO_PASSWORD" LEO_HOST="$HOST" LEO_SSHOPTS="$SSH_OPTS" \
    expect -f - <<'EXP'
set timeout 30
set opts [split $env(LEO_SSHOPTS) " "]
spawn ssh {*}$opts $env(LEO_USER)@$env(LEO_HOST)
expect { -re {(?i)password:} { send -- "$env(LEO_PASS)\r" } eof }
interact
EXP
    ;;

  put)
    [ $# -eq 2 ] || { echo "usage: leo.sh put <local> <remote>"; exit 2; }
    LEO_PASS="$LEONARDO_PASSWORD" expect -f - "$1" "$LEONARDO_USERNAME@$HOST:$2" <<'EXP'
set timeout 600
eval spawn scp -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no [lindex $argv 0] [lindex $argv 1]
expect { -re {(?i)password:} { send -- "$env(LEO_PASS)\r"; exp_continue } eof }
catch wait r; exit [lindex $r 3]
EXP
    ;;

  get)
    [ $# -eq 2 ] || { echo "usage: leo.sh get <remote> <local>"; exit 2; }
    LEO_PASS="$LEONARDO_PASSWORD" expect -f - "$LEONARDO_USERNAME@$HOST:$1" "$2" <<'EXP'
set timeout 600
eval spawn scp -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no [lindex $argv 0] [lindex $argv 1]
expect { -re {(?i)password:} { send -- "$env(LEO_PASS)\r"; exp_continue } eof }
catch wait r; exit [lindex $r 3]
EXP
    ;;

  smoke)
    echo "=== Leonardo smoke test → $HOST (user: $LEONARDO_USERNAME) ==="
    REMOTE='
      echo "--- identity ---";   hostname; whoami; id -gn;
      echo "--- storage ---";    echo "HOME=$HOME"; echo "SCRATCH=${SCRATCH:-<unset>}"; echo "PUBLIC=${PUBLIC:-<unset>}";
      [ -n "${SCRATCH:-}" ] && ls -ld "$SCRATCH" 2>/dev/null;
      echo "--- quota (home) ---"; ( cineca_project 2>/dev/null || true ); df -h "$HOME" 2>/dev/null | tail -1;
      echo "--- modules ---";    module --version 2>&1 | head -1 || echo "no modules";
      echo "--- pixi ---";       command -v pixi >/dev/null && pixi --version || echo "pixi NOT installed (run: curl -fsSL https://pixi.sh/install.sh | bash)";
      echo "--- slurm ---";      command -v sinfo >/dev/null && echo "slurm OK: $(sinfo --version 2>/dev/null)" || echo "no slurm";
      echo "--- reservation s_tra_ncc ---"; scontrol show res s_tra_ncc 2>/dev/null | head -6 || echo "reservation not visible (may be time-bound)";
      echo "--- my queue ---";   squeue --me 2>/dev/null || echo "squeue unavailable";
      echo "--- DONE ---"
    '
    _remote "$REMOTE"
    rc=$?
    echo
    if [ $rc -eq 0 ]; then echo "✅ SMOKE TEST PASSED (login + environment reachable)"; 
    elif [ $rc -eq 3 ]; then echo "❌ AUTH FAILED — check LEONARDO_USERNAME/PASSWORD in .env";
    elif [ $rc -eq 4 ]; then echo "❌ NETWORK FAILED — host unreachable / VPN?";
    else echo "⚠️  exited rc=$rc — inspect output above"; fi
    exit $rc
    ;;

  *)
    echo "usage: leo.sh {smoke|run \"<cmd>\"|shell|put <l> <r>|get <r> <l>}"; exit 2
    ;;
esac
