from agents import Agent, Runner
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI
import os

from optimistic_quote.tools.load_url import load_url
from optimistic_quote.tools.load_quotes import load_quotes_from_url
from optimistic_quote.tools.quote_optimizer import quote_optimizer
from optimistic_quote.tools.push_notification import push_notification
from optimistic_quote.tools.send_email_notification import send_email
openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

model = OpenAIResponsesModel(
    model="gpt-5-mini",
    openai_client=openai_client,
)

optimistic_quote_agent = Agent(
    name="OptimisticQuote",

    instructions="""
    You are an Optimistic Quote Agent.

    Your job is to find an optimistic and meaningful quote
    from the configured quote sources and send it to the user's
    smartphone.

    Follow this process:

    1. Use load_quotes_from_url to load quotes from the configured URLs.
    2. Use quote_optimizer to analyze all loaded quotes and select
       the most optimistic quote.
    3. Use push_notification to send the selected quote to the user.
    4. 3. Use send_email to send the selected quote to the email.

    Do not invent quotes.
    Only use quotes returned by the quote tools.

    Return a concise confirmation after the notification is sent.
    """,

    tools=[
        load_url,
        load_quotes_from_url,
        quote_optimizer,
        push_notification,
        send_email,
    ],
)


async def run_optimistic_quote_agent():

    result = await Runner.run(
        optimistic_quote_agent,
        "Find the most optimistic quote and send it to me."
    )

    return result.final_output