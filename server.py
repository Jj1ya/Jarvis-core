import os
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uvicorn
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# ==========================================
# 1. Tools Definition (도구 정의)
# ==========================================

@tool
def web_search(query: str) -> str:
    """Searches the internet for up-to-date information or news."""
    try:
        search = DuckDuckGoSearchRun()
        return f"[Web Search Result]\n{search.invoke(query)}"
    except Exception as e:
        return f"[Error] Web search failed: {e}"

@tool
def read_local_file(file_path: str) -> str:
    """
    Reads the content of a local file on the Mac.
    Useful for reviewing C language code or smart-freight-ai architecture files.
    """
    try:
        # [Security] Path Traversal (경로 조작 해킹) 원천 차단
        base_dir = Path.home() / "Desktop" / "portfolio"
        target_path = (base_dir / file_path).resolve()
        
        if not str(target_path).startswith(str(base_dir)):
            return "[Error] Security Violation: Attempted to access files outside the project directory."
            
        if not target_path.exists():
            return f"[Error] File not found at {target_path}"
            
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"[File Content of {target_path.name}]\n{content}"
    except Exception as e:
        return f"[Error] Failed to read file: {e}"

# 추후 Vector DB(Chroma) 연동을 위한 자리 표시자 (Placeholder)
@tool
def search_memory(query: str) -> str:
    """Searches the local Vector DB for past memories or project context."""
    return "[Memory] (Vector DB integration is pending. Placeholder response.)"

# ==========================================
# 2. Engine & System Initialization (엔진 초기화)
# ==========================================
print("[System] Jarvis Core Integrated API Server Booting...")

llm = ChatOllama(model="llama3.1", temperature=0)
tools = [web_search, read_local_file, search_memory]
jarvis_engine = llm.bind_tools(tools)
tools_map = {t.name: t for t in tools}

app = FastAPI(
    title="Jarvis Core API",
    description="Autonomous Agent Backend for smart-freight-ai",
    version="2.0.0"
)

# ==========================================
# 3. State Management (상태 관리 / 세션 메모리)
# ==========================================
# 사용자별 대화 맥락(Context)을 기억하기 위한 인메모리 DB
session_db = {}

def get_session_history(session_id: str):
    if session_id not in session_db:
        session_db[session_id] = [
            SystemMessage(content="""
            You are 'Jarvis Core', an elite AI architecture reviewer and autonomous agent.
            You are assisting Jiwoong with the 'smart-freight-ai' project and reviewing C programming code.
            Always answer in Korean concisely.
            If asked to read code, use the `read_local_file` tool.
            If asked about recent events, use the `web_search` tool.
            CRITICAL RULE: The user is listening via voice (TTS). DO NOT use any markdown formatting like **, *, #, or bullet points. Output plain spoken conversational text only.
            """)
        ]
    return session_db[session_id]

# ==========================================
# 4. API Endpoints (클라이언트 통신부)
# ==========================================
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user" # 모바일에서 식별자를 넘겨줄 수 있도록 설계

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        print(f"\n[API Request | Session: {req.session_id}] Received: {req.message}")
        
        # 1. 세션 불러오기 및 유저 메시지 추가
        chat_history = get_session_history(req.session_id)
        chat_history.append(HumanMessage(content=req.message))
        
        # 2. 1차 추론 (도구를 쓸지 말지 결정)
        response = jarvis_engine.invoke(chat_history)
        chat_history.append(response)
        
        # 3. 도구 실행 라우팅 (Tool Routing)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                print(f"[Agent Action] Executing tool: {tool_name}")
                
                if tool_name.lower() not in tools_map:
                    continue
                
                selected_tool = tools_map[tool_name.lower()]
                tool_output = selected_tool.invoke(tool_call["args"])
                chat_history.append(ToolMessage(
                    tool_call_id=tool_call["id"], 
                    name=tool_name, 
                    content=str(tool_output)
                ))
            
            # 4. 도구 결과값을 바탕으로 최종 답변 생성 (2차 추론)
            final_response = jarvis_engine.invoke(chat_history)
            chat_history.append(final_response)
            
            print(f"[API Response] {final_response.content[:50]}...")
            return ChatResponse(reply=final_response.content, session_id=req.session_id)
            
        else:
            print(f"[API Response] {response.content[:50]}...")
            return ChatResponse(reply=response.content, session_id=req.session_id)
            
    except Exception as e:
        print(f"[API Error] {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)