from dotenv import load_dotenv
import os
from pathlib import Path

# 1. Load the .env file before accessing os.getenv
# (explicit path is extra-safe if your entry script lives in src/)
load_dotenv

# 2. NOW read the values
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")
DATA_PATH = os.getenv("DATA_PATH")
DEVICE = os.getenv("DEVICE")

print("API key loaded?", OPENAI_API_KEY)
print("Model path:", MODEL_PATH)
print("Data path:", DATA_PATH)
print("Device:", DEVICE)