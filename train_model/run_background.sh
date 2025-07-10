#!/bin/bash

PID_FILE="training.pid"

start() {
    if [ -f "$PID_FILE" ]; then
        echo "Training is already running. PID: $(cat $PID_FILE)"
        exit 1
    fi

    nohup python3 -u train.py > training.log 2>&1 &
    echo $! > "$PID_FILE"
    echo "Training started in the background with PID $(cat $PID_FILE). Log is being written to training.log"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Training is not running."
        exit 1
    fi

    PID=$(cat "$PID_FILE")
    echo "Stopping training process with PID $PID"
    kill "$PID"
    rm "$PID_FILE"
    echo "Training stopped."
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
esac
