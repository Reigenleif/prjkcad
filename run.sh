#!/usr/bin/env bash
set -e

# Resolve directory of this script, following symlinks if any
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# Resolve canonical project root (directory containing gui/ and utils/)
if [ -d "$SCRIPT_DIR/gui" ] && [ -d "$SCRIPT_DIR/utils" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/../gui" ] && [ -d "$SCRIPT_DIR/../utils" ]; then
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd )"
else
    PROJECT_ROOT="/home/alief/prjkcad"
fi

cd "$PROJECT_ROOT"

# Resolve Python from conda env 'n' if available
if [ -f "/home/alief/miniconda3/envs/n/bin/python" ]; then
    PYTHON_BIN="/home/alief/miniconda3/envs/n/bin/python"
elif command -v conda &> /dev/null && conda env list | grep -q "^n "; then
    PYTHON_BIN="conda run --no-capture-output -n n python"
else
    PYTHON_BIN="python"
fi

# Clean up any lingering dev server processes on ports 8000 and 5173
fuser -k 8000/tcp >/dev/null 2>&1 || true
fuser -k 5173/tcp >/dev/null 2>&1 || true

echo "=================================================="
echo " Starting MINI CAD (FE & BE Dev Servers)"
echo "=================================================="
echo " Project:  $PROJECT_ROOT"
echo " Backend:  http://localhost:8000 (Docs: http://localhost:8000/docs)"
echo " Frontend: http://localhost:5173"
echo " Python:   $PYTHON_BIN"
echo "=================================================="

cleanup() {
    echo ""
    echo "Shutting down servers..."
    if [ -n "$BE_PID" ]; then
        kill "$BE_PID" 2>/dev/null || true
    fi
    if [ -n "$FE_PID" ]; then
        kill "$FE_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "All servers stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
$PYTHON_BIN -m uvicorn gui.backend.main:app --host 0.0.0.0 --port 8000 --reload &
BE_PID=$!
echo "[Backend started with PID $BE_PID]"

npm --prefix "$PROJECT_ROOT/gui/frontend" run dev &
FE_PID=$!
echo "[Frontend started with PID $FE_PID]"

wait -n
