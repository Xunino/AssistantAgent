import uvicorn
from dotenv import load_dotenv

def main():
    """Main function to run the FastAPI application"""
    load_dotenv()  # Load environment variables from .env file
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        workers=1,    # Number of worker processes
        log_level="info"
    )

if __name__ == "__main__":
    main()
