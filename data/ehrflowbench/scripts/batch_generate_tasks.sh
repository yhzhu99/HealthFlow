#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/generate_tasks.py"
OUTPUT_ROOT="$PROJECT_ROOT/data/ehrflowbench/processed/papers/generated_tasks"
RUNS_ROOT="$PROJECT_ROOT/data/ehrflowbench/processed/papers/generation_runs"
LATEST_RUN_FILE="$RUNS_ROOT/latest_run.txt"

usage() {
  cat <<'EOF'
Usage:
  batch_generate_tasks.sh START_ID END_ID
  batch_generate_tasks.sh status [RUN_ID]
  batch_generate_tasks.sh tail [RUN_ID]
  batch_generate_tasks.sh stop [RUN_ID]
  batch_generate_tasks.sh list

Examples:
  batch_generate_tasks.sh 1 100
  batch_generate_tasks.sh status
  batch_generate_tasks.sh tail 20260401_235500_1_100
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

is_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

ensure_layout() {
  mkdir -p "$OUTPUT_ROOT" "$RUNS_ROOT"
}

load_progress() {
  local run_dir="$1"
  local progress_file="$run_dir/progress.env"
  if [[ ! -f "$progress_file" ]]; then
    echo "error: progress file not found for run: $run_dir" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$progress_file"
}

resolve_run_dir() {
  local requested_run_id="${1:-}"
  ensure_layout
  if [[ -z "$requested_run_id" ]]; then
    if [[ ! -f "$LATEST_RUN_FILE" ]]; then
      echo "error: no recorded runs yet" >&2
      exit 1
    fi
    requested_run_id="$(<"$LATEST_RUN_FILE")"
  fi
  local run_dir="$RUNS_ROOT/$requested_run_id"
  if [[ ! -d "$run_dir" ]]; then
    echo "error: run directory not found: $run_dir" >&2
    exit 1
  fi
  printf '%s\n' "$run_dir"
}

write_progress_file() {
  local progress_file="$RUN_DIR/progress.env"
  local tmp_file="$progress_file.tmp"
  {
    printf 'RUN_ID=%q\n' "$RUN_ID"
    printf 'RANGE_START=%q\n' "$RANGE_START"
    printf 'RANGE_END=%q\n' "$RANGE_END"
    printf 'TOTAL=%q\n' "$TOTAL"
    printf 'COMPLETED=%q\n' "$COMPLETED"
    printf 'SUCCEEDED=%q\n' "$SUCCEEDED"
    printf 'FAILED=%q\n' "$FAILED"
    printf 'CURRENT_ID=%q\n' "$CURRENT_ID"
    printf 'STATE=%q\n' "$STATE"
    printf 'STARTED_AT=%q\n' "$STARTED_AT"
    printf 'FINISHED_AT=%q\n' "$FINISHED_AT"
    printf 'LAST_ERROR=%q\n' "$LAST_ERROR"
    printf 'OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
    printf 'RUN_DIR=%q\n' "$RUN_DIR"
  } >"$tmp_file"
  mv "$tmp_file" "$progress_file"
}

print_status() {
  local run_dir
  run_dir="$(resolve_run_dir "${1:-}")"
  load_progress "$run_dir"

  local pid=""
  local alive="no"
  if [[ -f "$run_dir/pid" ]]; then
    pid="$(<"$run_dir/pid")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      alive="yes"
    fi
  fi

  local percent=0
  if (( TOTAL > 0 )); then
    percent=$(( COMPLETED * 100 / TOTAL ))
  fi

  echo "run_id: $RUN_ID"
  echo "state: $STATE"
  echo "pid: ${pid:-n/a} (alive: $alive)"
  echo "range: $RANGE_START-$RANGE_END"
  echo "progress: $COMPLETED/$TOTAL (${percent}%)"
  echo "success: $SUCCEEDED"
  echo "failed: $FAILED"
  echo "current_paper: ${CURRENT_ID:-n/a}"
  echo "started_at: $STARTED_AT"
  echo "finished_at: ${FINISHED_AT:-n/a}"
  echo "output_root: $OUTPUT_ROOT"
  echo "run_dir: $RUN_DIR"
  echo "batch_log: $run_dir/batch.log"
  echo "failed_ids: $run_dir/failed_ids.txt"
  echo "success_ids: $run_dir/success_ids.txt"
  if [[ -n "$LAST_ERROR" ]]; then
    echo "last_error: $LAST_ERROR"
  fi
}

tail_run() {
  local run_dir
  run_dir="$(resolve_run_dir "${1:-}")"
  local log_file="$run_dir/batch.log"
  if [[ ! -f "$log_file" ]]; then
    echo "error: batch log not found: $log_file" >&2
    exit 1
  fi
  tail -n 40 -f "$log_file"
}

stop_run() {
  local run_dir
  run_dir="$(resolve_run_dir "${1:-}")"
  local pid_file="$run_dir/pid"
  if [[ ! -f "$pid_file" ]]; then
    echo "error: pid file not found: $pid_file" >&2
    exit 1
  fi
  local pid
  pid="$(<"$pid_file")"
  if [[ -z "$pid" ]]; then
    echo "error: pid file is empty: $pid_file" >&2
    exit 1
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "sent stop signal to run $(basename "$run_dir") (pid $pid)"
  else
    echo "process is not running for run $(basename "$run_dir") (pid $pid)"
  fi
}

list_runs() {
  ensure_layout
  if ! find "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    echo "no runs found"
    return
  fi

  find "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | while IFS= read -r run_dir; do
    if [[ -f "$run_dir/progress.env" ]]; then
      load_progress "$run_dir"
      echo "$RUN_ID  state=$STATE  progress=$COMPLETED/$TOTAL  success=$SUCCEEDED  failed=$FAILED"
    else
      echo "$(basename "$run_dir")  state=unknown"
    fi
  done
}

run_generator_for_paper() {
  local paper_id="$1"
  uv run python "$PYTHON_SCRIPT" \
    --paper-id "$paper_id" \
    --output-dir "$OUTPUT_ROOT" \
    --overwrite
}

worker_main() {
  RANGE_START="$1"
  RANGE_END="$2"
  RUN_DIR="$3"
  RUN_ID="$4"
  TOTAL=$(( RANGE_END - RANGE_START + 1 ))
  COMPLETED=0
  SUCCEEDED=0
  FAILED=0
  CURRENT_ID=""
  STATE="running"
  STARTED_AT="$(timestamp)"
  FINISHED_AT=""
  LAST_ERROR=""

  mkdir -p "$RUN_DIR/papers"
  : >"$RUN_DIR/success_ids.txt"
  : >"$RUN_DIR/failed_ids.txt"
  write_progress_file

  exec >>"$RUN_DIR/batch.log" 2>&1

  echo "[$(timestamp)] run $RUN_ID started for paper ids $RANGE_START-$RANGE_END"
  echo "[$(timestamp)] generated tasks will be written under $OUTPUT_ROOT"

  on_stop() {
    STATE="stopped"
    FINISHED_AT="$(timestamp)"
    LAST_ERROR="run stopped by signal"
    write_progress_file
    echo "[$(timestamp)] run $RUN_ID stopped"
    exit 130
  }

  trap on_stop INT TERM

  local paper_id=""
  local paper_log=""
  for (( paper_id = RANGE_START; paper_id <= RANGE_END; paper_id++ )); do
    CURRENT_ID="$paper_id"
    LAST_ERROR=""
    write_progress_file

    paper_log="$RUN_DIR/papers/${paper_id}.log"
    echo "[$(timestamp)] paper $paper_id started"

    if run_generator_for_paper "$paper_id" >"$paper_log" 2>&1; then
      SUCCEEDED=$(( SUCCEEDED + 1 ))
      printf '%s\n' "$paper_id" >>"$RUN_DIR/success_ids.txt"
      echo "[$(timestamp)] paper $paper_id finished successfully"
    else
      FAILED=$(( FAILED + 1 ))
      printf '%s\n' "$paper_id" >>"$RUN_DIR/failed_ids.txt"
      LAST_ERROR="paper $paper_id failed, see $paper_log"
      echo "[$(timestamp)] paper $paper_id failed, see $paper_log"
    fi

    COMPLETED=$(( COMPLETED + 1 ))
    write_progress_file
  done

  CURRENT_ID=""
  STATE="completed"
  FINISHED_AT="$(timestamp)"
  write_progress_file
  echo "[$(timestamp)] run $RUN_ID completed: success=$SUCCEEDED failed=$FAILED total=$TOTAL"
}

start_run() {
  local start_id="$1"
  local end_id="$2"

  if ! is_integer "$start_id" || ! is_integer "$end_id"; then
    echo "error: START_ID and END_ID must be positive integers" >&2
    usage >&2
    exit 1
  fi
  if (( start_id > end_id )); then
    echo "error: START_ID must be less than or equal to END_ID" >&2
    exit 1
  fi

  ensure_layout

  local stamp
  stamp="$(date '+%Y%m%d_%H%M%S')"
  local run_id="${stamp}_${start_id}_${end_id}"
  local run_dir="$RUNS_ROOT/$run_id"
  mkdir -p "$run_dir/papers"
  : >"$run_dir/batch.log"
  : >"$run_dir/success_ids.txt"
  : >"$run_dir/failed_ids.txt"

  RUN_ID="$run_id"
  RANGE_START="$start_id"
  RANGE_END="$end_id"
  TOTAL=$(( RANGE_END - RANGE_START + 1 ))
  COMPLETED=0
  SUCCEEDED=0
  FAILED=0
  CURRENT_ID=""
  STATE="starting"
  STARTED_AT="$(timestamp)"
  FINISHED_AT=""
  LAST_ERROR=""
  RUN_DIR="$run_dir"
  write_progress_file

  printf '%s\n' "$run_id" >"$LATEST_RUN_FILE"

  nohup env bash "$SELF_PATH" __worker "$start_id" "$end_id" "$run_dir" "$run_id" </dev/null >/dev/null 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" >"$run_dir/pid"

  echo "started run: $run_id"
  echo "pid: $pid"
  echo "range: $start_id-$end_id"
  echo "status: $SELF_PATH status $run_id"
  echo "tail: $SELF_PATH tail $run_id"
  echo "stop: $SELF_PATH stop $run_id"
}

main() {
  if (( $# == 2 )) && is_integer "$1" && is_integer "$2"; then
    start_run "$1" "$2"
    return
  fi

  if (( $# < 1 )); then
    usage >&2
    exit 1
  fi

  case "$1" in
    status)
      print_status "${2:-}"
      ;;
    tail)
      tail_run "${2:-}"
      ;;
    stop)
      stop_run "${2:-}"
      ;;
    list)
      list_runs
      ;;
    __worker)
      if (( $# != 5 )); then
        echo "error: invalid internal worker invocation" >&2
        exit 1
      fi
      worker_main "$2" "$3" "$4" "$5"
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
