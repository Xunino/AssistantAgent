from loguru import logger
from typing import Dict
from crawl4ai import AsyncWebCrawler
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
from duckduckgo_search import DDGS

from app.services.graph.config import GraphState
from app.services.graph.brain.prompts.google_search import ENHANCE_GG_SEARCH_PROMPT


class GoogleSearch(Runnable):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def get_detailed_website(self, url: str) -> str:
        """Fetch detailed website content and return as markdown."""
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown

    def get_links(self, query: str) -> str:
        links = DDGS().text(query, max_results=1)
        if links:
            return links[0]["href"]
        else:
            return ""

    def get_human_message(self, inputs: GraphState):
        return [msg for msg in inputs.messages if isinstance(msg, HumanMessage)]

    def invoke(self, inputs: GraphState) -> Dict:
        """Invoke the google search module."""
        raise NotImplementedError("Method not implemented")

    async def ainvoke(self, inputs: GraphState) -> Dict:
        """Invoke the google search module asynchronously."""
        prompt = (
            [SystemMessage(content=ENHANCE_GG_SEARCH_PROMPT)]
            + self.get_human_message(inputs)
            + [HumanMessage(content=inputs.user_question)]
        )
        response = await self.llm.ainvoke(prompt)
        logger.info(f"Google search response: {response.content}")
        links = self.get_links(response.content)
        detailed_website = await self.get_detailed_website(links)

        return {"google_search_result": str(detailed_website)}
