#!/bin/bash

# Clean up background processes on Ctrl+C / Exit
cleanup() {
    echo -e "\nStopping server..."
    kill "$FRONTEND_PID" 2>/dev/null
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

echo "Starting Local HTTP Web Server on port 8080..."
python -m http.server 8080 &
FRONTEND_PID=$!

echo "=================================================="
echo "🚀 Explainable GAT Face Dashboard is running!"
echo "   - Local URL: http://localhost:8080"
echo "=================================================="
echo "Press Ctrl+C to stop the server."

# Wait for processes
wait "$FRONTEND_PID"
