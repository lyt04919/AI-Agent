import streamlit as st
import os
from dotenv import load_dotenv
from langchain.tools import tool
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="小米Token申请演示 - 智能客服MVP", layout="wide")

st.title("🤖 智能客服多Agent协作系统 (MVP演示)")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("配置中心")
    api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    api_base = st.text_input("Base URL", value=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    model = st.text_input("Model Name", value="gpt-4o")
    if st.button("更新配置"):
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = api_base
        st.success("配置已保存")

# Mock Tools
@tool("KnowledgeBase")
def kb_tool(query: str) -> str:
    """查询公司政策"""
    return "政策：7天内无理由退货，8-30天内退货需扣除10%手续费。"

@tool("OrderSystem")
def order_tool(order_id: str) -> str:
    """查询订单状态"""
    return f"订单 {order_id} 已签收15天。"

# UI 交互
user_msg = st.text_area("用户输入:", "我想退掉订单 ORDER8888，我已经收到半个月了，还可以退吗？")

if st.button("开始协作处理"):
    if not api_key:
        st.warning("请在侧边栏填入 API Key")
    else:
        llm = ChatOpenAI(model=model, temperature=0.1)
        
        # Agents
        analyst = Agent(role='意图分析师', goal='拆解需求', backstory='擅长逻辑拆解', llm=llm)
        researcher = Agent(role='政策专家', goal='查政策', backstory='精通规则', tools=[kb_tool], llm=llm)
        operator = Agent(role='系统专员', goal='查订单', backstory='严谨', tools=[order_tool], llm=llm)
        writer = Agent(role='金牌客服', goal='写回复', backstory='专业礼貌', llm=llm)

        # Tasks
        tasks = [
            Task(description=f"分析: {user_msg}", agent=analyst, expected_output="分析报告"),
            Task(description="查政策", agent=researcher, expected_output="规则细节"),
            Task(description="查数据", agent=operator, expected_output="订单事实"),
            Task(description="写最终回复", agent=writer, expected_output="客服话术")
        ]

        with st.spinner("多 Agent 正在深度推理中..."):
            crew = Crew(agents=[analyst, researcher, operator, writer], tasks=tasks, process=Process.sequential)
            final_result = crew.kickoff()
            st.success("处理完成!")
            st.markdown("### 最终回复：")
            st.info(final_result)
