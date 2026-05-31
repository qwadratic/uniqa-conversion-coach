#!/usr/bin/env bash
# Persistent Leonardo session in a LOCAL tmux pane — log in ONCE, then run many commands
# with no re-auth. Much faster than leo.sh (which opens a fresh SSH+password per command).
#
#   leo_tmux.sh start              # create tmux session 'leo', ssh+auth, ready for commands
#   leo_tmux.sh run "squeue --me"  # run a command, capture+print its output (no re-auth)
#   leo_tmux.sh jobs               # live squeue --me + tail of the newest slurm-*.out
#   leo_tmux.sh watch              # launch a self-refreshing job monitor in window 'watch'
#   leo_tmux.sh peek               # snapshot the watch window
#   leo_tmux.sh attach             # attach the tmux session in your terminal (Ctrl-b d to leave)
#   leo_tmux.sh kill               # tear down the session
#
# scp note: scp to login nodes is BLOCKED — use a datamover host + ABSOLUTE remote path.
# See SKILL.md "File transfer (scp via datamover)". This helper is for commands, not uploads.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESS="${LEO_TMUX_SESSION:-leo}"
RUNDIR="${LEO_RUNDIR:-\$HOME/zero-one}"   # REMOTE project dir for jobs/tail (\$HOME = remote)

_load_env() {
  local d="$PWD"
  for _ in 1 2 3 4 5 6; do
    [ -f "$d/.env" ] && { set -a; . "$d/.env"; set +a; return 0; }
    d="$(dirname "$d")"; [ "$d" = "/" ] && break
  done
  [ -f "$HOME/Desktop/hobby-dev/zero-one/.env" ] && { set -a; . "$HOME/Desktop/hobby-dev/zero-one/.env"; set +a; }
}
[ -z "${LEONARDO_PASSWORD:-}" ] && _load_env
: "${LEONARDO_USERNAME:?set LEONARDO_USERNAME in .env}"
: "${LEONARDO_PASSWORD:?set LEONARDO_PASSWORD in .env}"
HOST="${LEONARDO_HOST:-login01-ext.leonardo.cineca.it}"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o NumberOfPasswordPrompts=1 -o PreferredAuthentications=password -o PubkeyAuthentication=no"

_started() { tmux has-session -t "$SESS" 2>/dev/null; }

cmd="${1:-help}"; shift || true
case "$cmd" in
  start)
    if _started; then echo "[$SESS] already running"; exit 0; fi
    command -v tmux >/dev/null || { echo "tmux not installed (brew install tmux)"; exit 2; }
    LEO_USER="$LEONARDO_USERNAME" LEO_PASS="$LEONARDO_PASSWORD" LEO_HOST="$HOST" LEO_SSHOPTS="$SSH_OPTS" \
      tmux new-session -d -s "$SESS" -x 220 -y 60 "expect -f '$HERE/login.exp'"
    sleep 8
    # quiet, predictable shell: no input echo, minimal prompt, no pager (remote \$HOME)
    tmux send-keys -t "$SESS" "stty -echo; export PS1='' PROMPT_COMMAND='' PAGER=cat GIT_PAGER=cat; cd \"$RUNDIR\" 2>/dev/null; clear" Enter
    sleep 1
    echo "[$SESS] up. Test:  $0 run 'hostname; squeue --me'"
    ;;

  run)
    _started || { echo "[$SESS] not running — '$0 start' first"; exit 2; }
    local_cmd="$*"
    B="__B$$_${RANDOM}${RANDOM}"; E="__E$$_${RANDOM}${RANDOM}"
    tmux send-keys -t "$SESS" "echo $B; { $local_cmd ; } 2>&1; echo $E" Enter
    cap=""
    for _ in $(seq 1 "${LEO_TIMEOUT:-240}"); do
      cap="$(tmux capture-pane -t "$SESS" -p -S -8000 2>/dev/null)"
      printf '%s' "$cap" | grep -q "$E" && break
      sleep 1
    done
    printf '%s\n' "$cap" | awk -v b="$B" -v e="$E" '$0~e{f=0} f; $0~b{f=1}'
    ;;

  jobs)
    "$0" run "squeue --me -o '%.11i %.13j %.2t %.11M %.6D %R'; echo; f=\$(ls -t $RUNDIR/slurm-*.out 2>/dev/null | head -1); echo \"== tail \$f ==\"; tail -n 8 \"\$f\" 2>/dev/null | tr '\r' '\n' | grep -vE '^[[:space:]]*\$' | tail -6"
    ;;

  watch)
    _started || { echo "[$SESS] not running — '$0 start' first"; exit 2; }
    tmux kill-window -t "$SESS:watch" 2>/dev/null || true
    # 1) write a plain monitor script on Leonardo via the (reliable) run channel
    "$0" run "cat > \$HOME/jobwatch.sh <<'JW'
while true; do clear; date '+%F %H:%M:%S'; echo '== squeue --me =='; squeue --me -o '%.11i %.14j %.2t %.11M %.6D %R'; echo; f=\$(ls -t \$HOME/zero-one/slurm-*.out 2>/dev/null | head -1); echo \"== \$f ==\"; tail -n 8 \"\$f\" 2>/dev/null | tr '\r' '\n' | grep -vE '^[[:space:]]*\$' | tail -6; sleep 10; done
JW
echo wrote jobwatch.sh" >/dev/null
    # 2) own authenticated SSH window that just runs it (sleep loop = ~0 CPU)
    LEO_USER="$LEONARDO_USERNAME" LEO_PASS="$LEONARDO_PASSWORD" LEO_HOST="$HOST" LEO_SSHOPTS="$SSH_OPTS" \
      tmux new-window -t "$SESS" -n watch "expect -f '$HERE/jobwatch.exp'"
    echo "[$SESS] watch window launched. Snapshot:  $0 peek   |  live:  $0 attach"
    ;;

  peek)
    tmux capture-pane -t "$SESS:watch" -p 2>/dev/null || tmux capture-pane -t "$SESS" -p
    ;;

  send)   _started && tmux send-keys -t "$SESS" "$*" Enter ;;
  attach) exec tmux attach -t "$SESS" ;;
  kill)   tmux kill-session -t "$SESS" 2>/dev/null && echo "[$SESS] killed" || echo "[$SESS] not running" ;;
  status) _started && echo "[$SESS] running" || echo "[$SESS] not running" ;;
  *) sed -n '2,20p' "${BASH_SOURCE[0]}" ;;
esac
