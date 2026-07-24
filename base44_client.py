"""Base44 API client for streaming live price ticks to TradeStream AI."""
from __future__ import annotations
import os
import httpx

BASE44_ENDPOINT = "https://trade-qualified-pulse-live.base44.app/functions/log_price_tick"

class Base44Client:
    """Posts price ticks to base44 TradeStream AI dashboard."""
    
    def __init__(self):
        self.api_key = os.getenv("BASE44_AGENT_API_KEY", "")
        self.enabled = bool(self.api_key)
    
    async def post_tick(self, symbol: str, price: float) -> bool:
        """
        Post a single price tick to base44.
        Returns True if successful, False otherwise (never raises).
        """
        if not self.enabled:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    BASE44_ENDPOINT,
                    json={"symbol": symbol, "price": price},
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code in (200, 201, 204)
        except Exception as exc:
            print(f"[base44] post_tick failed: {exc}", flush=True)
            return False
