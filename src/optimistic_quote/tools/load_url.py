import logging
from pathlib import Path

from agents import function_tool


logger = logging.getLogger(__name__)


def _load_url() -> list[str]:
    """
    Normal Python function used internally by other tools.
    """

    logger.info("========== load_url START ==========")

    file_path = Path(__file__).resolve().parents[3] / "quotes_urls.txt"

    logger.info(
        "Looking for URL file: %s",
        file_path.absolute(),
    )

    if not file_path.exists():

        logger.error(
            "URL file does not exist: %s",
            file_path.absolute(),
        )

        raise FileNotFoundError(
            f"URL file not found: {file_path.absolute()}"
        )

    urls = [
        line.strip()
        for line in file_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    logger.info(
        "Number of URLs loaded: %d",
        len(urls),
    )

    for url in urls:

        logger.info(
            "Configured URL: %s",
            url,
        )

    logger.info("========== load_url END ==========")

    return urls


@function_tool
def load_url() -> list[str]:
    """
    OpenAI Agent tool for loading configured quote URLs.
    """

    return _load_url()