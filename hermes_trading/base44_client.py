import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)


class Base44Client:
    """Client for posting price ticks to Base44 AI site builder."""

    def __init__(self, agent_api_key: str):
        """
        Initialize Base44Client.

        Args:
            agent_api_key: The API key for authentication with Base44
        """
        self.agent_api_key = agent_api_key
        self.endpoint = "https://trade-qualified-pulse-live.base44.app/functions/log_price_tick"

    async def post_price_tick(self, symbol: str, price: float, ts: float | None = None) -> bool:
        """
        Post a single price tick to Base44.

        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            price: Current price
            ts: Optional timestamp (defaults to server time)

        Returns:
            True if successful, False otherwise
        """
        payload = {"symbol": symbol, "price": price}
        if ts is not None:
            payload["ts"] = ts

        return await self._post(payload)

    async def post_price_ticks(self, ticks: list[dict]) -> bool:
        """
        Post multiple price ticks in a single request.

        Args:
            ticks: List of tick dicts with 'symbol', 'price', and optional 'ts' keys

        Returns:
            True if successful, False otherwise
        """
        payload = {"ticks": ticks}
        return await self._post(payload)

    async def _post(self, payload: dict) -> bool:
        """
        Internal method to POST to the Base44 endpoint with proper auth.

        Args:
            payload: The JSON payload to send

        Returns:
            True if POST succeeded (2xx status), False otherwise
        """
        headers = {"Authorization": f"Bearer {self.agent_api_key}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    if response.status >= 200 and response.status < 300:
                        logger.debug(f"Base44 POST successful: {response.status}")
                        return True
                    else:
                        # non-2xx (e.g. auth) — debug only so it can't spam the deploy logs
                        logger.debug(f"Base44 POST failed with status {response.status}")
                        return False
        except asyncio.TimeoutError:
            logger.debug("Base44 POST timeout")
            return False
        except Exception as e:
            logger.debug(f"Base44 POST error: {e}")
            return False
