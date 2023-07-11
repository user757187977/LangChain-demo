import os

from langchain import OpenAI, PromptTemplate, LLMChain, SerpAPIWrapper, LLMMathChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool

llm = OpenAI(temperature=0)

def one():
    """
    基础版：输入整个语义
    :return:
    """
    text = "对于一家生产彩色袜子的公司来说，什么是好的公司名称？"
    result = llm(text)
    print(result)


def two():
    """
    prompt版本：提前定义好语义，输入关键词即可
    :return:
    """
    prompt = PromptTemplate(input_variables=["product"],
                            template="对于一家生产{product}的公司来说，什么是好的公司名称？")

    chain = LLMChain(llm=llm, prompt=prompt)

    # 这样我们只需要输入产品名称即可
    result = chain.run("杨梅")
    print(result)


def three():
    """
    agents版：动态代理，实现自动选择计算工具
    :return:
    """
    # SerpApi是一个付费提供搜索结果API的第三方服务提供商。它允许用户通过简单的API调用访问各种搜索引擎的搜索结果，包括Google、Bing、Yahoo、Yandex等。
    # llm-math是langchain里面的能做数学计算的模块
    # tools = load_tools(["serpapi", "llm-math"], llm=llm)

    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]

    # 初始化tools，models 和使用的agent
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # 输出结果
    result = agent.run("你是人工智能吗？海湾战争距离现在多少年了? 这个数字的三次方式多少?")
    print(result)


if __name__ == '__main__':
    three()