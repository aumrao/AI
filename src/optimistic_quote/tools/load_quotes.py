import logging

from agents import function_tool

from src.optimistic_quote.tools.load_url import _load_url
from src.optimistic_quote.clients.tavily_client import TavilyClient


logger = logging.getLogger(__name__)


@function_tool
def load_quotes_from_url() -> list[str]:

    logger.info("========== load_quotes_from_url START ==========")

    try:

        urls = _load_url()

        logger.info(
            "load_url returned %d URLs",
            len(urls),
        )

        tavily_client = TavilyClient()

        all_quotes = []

        for url in urls:

            logger.info(
                "Processing URL: %s",
                url,
            )

            try:

                quotes = tavily_client.extract_quotes(url)

                logger.info(
                    "Tavily returned %d items for URL",
                    len(quotes),
                )

                all_quotes.extend(quotes)

            except Exception:

                logger.exception(
                    "FAILED TO EXTRACT QUOTES FROM: %s",
                    url,
                )

        logger.info(
            "Total quotes collected: %d",
            len(all_quotes),
        )

        if not all_quotes:

            logger.warning(
                "ZERO QUOTES WERE COLLECTED"
            )

        logger.info("========== load_quotes_from_url END ==========")

        return all_quotes

    except Exception:

        logger.exception(
            "load_quotes_from_url FAILED"
        )

        raise