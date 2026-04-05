"""
API Server for ISOPro Simulations

This module runs the Flask API server for the Core Simulation API.
"""

import os
from dotenv import load_dotenv
from isopro.api import app

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get the port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Run the Flask application
    app.run(host="0.0.0.0", port=port)