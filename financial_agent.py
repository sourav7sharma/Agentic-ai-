from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Web search agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,

)

# Financial agent
Finance_agent=Agent(
    name="Finance Agent",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            technical_indicators=True, 
            company_news=True
        )],
    instructions=["Use tables to display the data"],
    show_tools_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    team=[web_search_agent, Finance_agent],
    model=Groq(id="llama3-70b-8192"),
    instructions=["Always include sources","Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recommendation and share latest news for AMZN", stream=True)