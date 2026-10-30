"""Start the local API + browser dashboard at http://127.0.0.1:8000."""
import uvicorn

if __name__ == "__main__":
    print("\nFraudShield demo: open http://127.0.0.1:8000 in your browser\n")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
