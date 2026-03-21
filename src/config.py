from dotenv import load_dotenv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


def check_keys():
    print("GOOGLE:", bool(GOOGLE_API_KEY))
    print("TAVILY:", bool(TAVILY_API_KEY))
    print("NVIDIA:", bool(NVIDIA_API_KEY))


if __name__ == "__main__":
    check_keys()