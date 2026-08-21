import logging
import os

from tavily import TavilyClient as TavilySDKClient


logger = logging.getLogger(__name__)


class TavilyClient:

    def __init__(self):

        logger.info("Initializing Tavily client")

        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:

            logger.error(
                "TAVILY_API_KEY IS NOT SET"
            )

            raise ValueError(
                "TAVILY_API_KEY is not configured"
            )

        logger.info(
            "TAVILY_API_KEY is configured"
        )

        self.client = TavilySDKClient(
            api_key=api_key
        )

        logger.info(
            "Tavily client initialized"
        )

    def extract_quotes(self, url: str) -> list[str]:

        logger.info(
            "Tavily extraction started: %s",
            url,
        )

        try:

            response = self.client.extract(
                urls=[url]
            )

            logger.info(
                "Tavily request completed"
            )

            logger.debug(
                "Tavily response type: %s",
                type(response).__name__,
            )

            logger.debug(
                "Tavily response: %s",
                response,
            )

            results = response.get(
                "results",
                []
            )

            logger.info(
                "Tavily result count: %d",
                len(results),
            )

            quotes = []

            for result in results:

                content = result.get(
                    "raw_content",
                    ""
                )

                if content:

                    quotes.append(content)

            logger.info(
                "Extracted %d quote candidates",
                len(quotes),
            )

            return quotes

        except Exception:

            logger.exception(
                "TAVILY EXTRACTION FAILED: %s",
                url,
            )

            raise