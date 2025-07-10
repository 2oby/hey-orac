#!/bin/bash

# Startup script for Hey Orac - Option 3: Web Backend as a Service
# Runs both the web backend and main wake word detection in the same container

set -e  # Exit on any error

echo "🚀 Starting Hey Orac services (Option 3: Web Backend as Service)..."

# Function to cleanup background processes on exit
cleanup() {
    echo "🛑 Shutting down services..."
    if [ ! -z "$WEB_PID" ]; then
        echo "🛑 Stopping web backend (PID: $WEB_PID)..."
        kill $WEB_PID 2>/dev/null || true
    fi
    if [ ! -z "$MAIN_PID" ]; then
        echo "🛑 Stopping main process (PID: $MAIN_PID)..."
        kill $MAIN_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start web backend in background
echo "🌐 Starting web backend on port 7171..."
/app/venv/bin/python src/web_backend.py &
WEB_PID=$!

# Wait a moment for web backend to start
sleep 3

# Check if web backend started successfully
if ! kill -0 $WEB_PID 2>/dev/null; then
    echo "❌ Web backend failed to start"
    exit 1
fi

echo "✅ Web backend started successfully (PID: $WEB_PID)"

# Start main wake word detection in background
echo "🎤 Starting wake word detection..."
/app/venv/bin/python src/main.py --startup-test-model third_party/openwakeword/custom_models/Hay--compUta_v_lrg.onnx --startup-test-duration 15 &
MAIN_PID=$!

# Wait a moment for main process to start
sleep 2

# Check if main process started successfully
if ! kill -0 $MAIN_PID 2>/dev/null; then
    echo "❌ Main process failed to start"
    cleanup
    exit 1
fi

echo "✅ Main process started successfully (PID: $MAIN_PID)"
echo "🎉 All services running! Web interface available at http://localhost:7171"

# Wait for either process to exit
while kill -0 $WEB_PID 2>/dev/null && kill -0 $MAIN_PID 2>/dev/null; do
    sleep 1
done

# If we get here, one of the processes has exited
echo "⚠️  One of the services has stopped"
cleanup 