#!/bin/bash

# Run Streamlit UI for Selective Steering

echo "Starting Selective Steering Chat UI..."
echo "=================================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit not found. Installing..."
    pip install streamlit
fi

# Run the app
streamlit run ui/app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#667eea" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f0f2f6" \
    --theme.textColor "#262730"