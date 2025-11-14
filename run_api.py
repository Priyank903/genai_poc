"""
Startup script for FastAPI backend
"""

import uvicorn
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Run FastAPI server
    uvicorn.run(
        "src.fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
