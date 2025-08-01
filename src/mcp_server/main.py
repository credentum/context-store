#!/usr/bin/env python3
"""
MCP Server for context-store
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Context Store MCP Server")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)