import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseClient:
    def __init__(self):
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            self.client: Optional[Client] = None
            print("Supabase credentials missing. Running in mock mode.")
        else:
            self.client = create_client(url, key)

    def save_analysis(self, data: dict):
        if not self.client:
            print(f"Mock saving analysis: {data['audio_name']}")
            return
        self.client.table("analysis_results").insert(data).execute()

    def get_history(self, user_id: str):
        if not self.client:
            return []
        return self.client.table("analysis_results").select("*").eq("user_id", user_id).execute().data

supabase = SupabaseClient()
