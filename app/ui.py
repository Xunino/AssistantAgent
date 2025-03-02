from langchain_core.messages import HumanMessage
from app.services.graph import AssistantGraph
from utils.llm import openai_llm
from rich.console import Console
from rich.text import Text
import logging

logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("primp").setLevel(logging.CRITICAL)


# logger.remove()

# Initialize rich console for beautiful output
console = Console()

graph = AssistantGraph(llm=openai_llm)

# Pretty visualize the graph
console.rule("Graph Visualization")
console.print(graph._visualize(), style="bold blue")


async def test_stream(inputs: dict):
    try:
        async for response in graph.astream_with_checkpointer(inputs, thread_id="test"):
            # Use rich for colorized and formatted streaming
            console.print(
                Text(response, style="bold cyan"),
                end="",
            )
        console.print()  # Newline after streaming
    except Exception as e:
        # Pretty print errors
        console.print(f"[bold red]Error occurred:[/bold red] {e}")


# Run the async function
if __name__ == "__main__":
    import asyncio

    while True:
        # Styled input prompt
        console.rule("User Input")
        user_question = console.input("[bold blue]User:[/bold blue] ")
        inputs = {
            "user_question": user_question,
            "messages": [HumanMessage(content=user_question)],
        }
        asyncio.run(test_stream(inputs))
