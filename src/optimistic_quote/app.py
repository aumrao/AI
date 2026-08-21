import asyncio
from dotenv import load_dotenv

load_dotenv(".env", override=True)

from optimistic_quote.agents.optimistic_quote_agent import (
    run_optimistic_quote_agent,
)


async def main():
    result = await run_optimistic_quote_agent()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())