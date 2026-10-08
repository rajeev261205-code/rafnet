import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Aegis Voice SOC"
    API_V1_STR: str = "/api/v1"
    
    # Supabase
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    # ML Config
    MODEL_PATH: str = "experiments/checkpoints/gateway2.pth"
    DEVICE: str = os.getenv("RAFNET_DEVICE", "cpu")
    
    class Config:
        env_file = ".env"

settings = Settings()
