import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

# ==========================================
# 1. 도구 정의 (Tool Definitions)
# ==========================================
@tool
def get_current_time(timezone_query: str = "CST") -> str:
    """Returns the current date and time."""
    zone_mapping = {
        "KST": "Asia/Seoul", "SEOUL": "Asia/Seoul", "KOREA": "Asia/Seoul", "한국": "Asia/Seoul",
        "CST": "America/Chicago", "TEXAS": "America/Chicago", "CHICAGO": "America/Chicago"
    }
    
    query_upper = timezone_query.upper()
    target_zone_str = "America/Chicago"
    
    for key, val in zone_mapping.items():
        if key in query_upper:
            target_zone_str = val
            break
            
    try:
        target_zone = ZoneInfo(target_zone_str)
        now = datetime.now(target_zone)
        return (f"[System Data] 요청하신 지역의 정확한 시스템 시간은 "
                f"{now.strftime('%Y년 %m월 %d일 %H시 %M분')} 입니다. "
                f"절대 이 시간을 임의로 더하거나 빼지 말고, 그대로 출력하십시오.")
    except Exception as e:
        return f"Error computing time: {e}"

@tool
def read_local_file(file_path: str) -> str:
    """
    Reads the content of a local file on the Mac.
    Use this when the user asks you to review, analyze, or explain a code file or document.
    """
    try:
        expanded_path = os.path.expanduser(file_path)
        with open(expanded_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"[File Content of {file_path}]\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"

# [추가됨] Vector DB를 뒤져서 과거의 지식을 꺼내오는 도구
@tool
def search_memory(query: str) -> str:
    """
    Searches the local Vector Database (Jarvis long-term memory) for information relevant to the query.
    Use this tool when the user asks about past context, personal rules, project details (e.g., smart-freight-ai), or specific vehicle records.
    """
    try:
        # 로컬 DB 연결 (nomic-embed-text 사용)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        db = Chroma(persist_directory="./jarvis_memory", embedding_function=embeddings)
        
        # 코사인 유사도 기반 검색 (가장 관련성 높은 3개의 조각 추출)
        docs = db.similarity_search(query, k=3)
        
        if not docs:
            return "[System Data] DB에서 관련된 기억을 찾을 수 없습니다."
        
        retrieved_context = "[Retrieved Memory from Vector DB]\n"
        for i, doc in enumerate(docs):
            retrieved_context += f"Data {i+1}: {doc.page_content}\n"
            
        return retrieved_context
    except Exception as e:
        return f"[Error] 메모리 검색 시스템 장애: {e}"

@tool
def web_search(query: str) -> str:
    """
    Searches the internet (DuckDuckGo) for up-to-date information, news, or technical documentation.
    Use this tool ONLY when the user asks for recent events, global facts, or current external information not found in local memory.
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.invoke(query)
        return f"[Web Search Result]\n{result}"
    except Exception as e:
        return f"[Error] Web search failed: {e}"

# ==========================================
# 2. 엔진 초기화 및 바인딩 (Engine Setup)
# ==========================================
def initialize_jarvis():
    try:
        print("[System] Jarvis Core 엔진 부팅 중... (Agentic Mode + RAG Memory)")
        llm = ChatOllama(model="llama3.1", temperature=0)
        
        # [수정됨] 도구 배열에 search_memory 추가
        tools = [get_current_time, read_local_file, search_memory, web_search]
        llm_with_tools = llm.bind_tools(tools)
        
        return llm_with_tools, tools
    except Exception as e:
        print(f"[Error] 엔진 초기화 실패: {e}")
        sys.exit(1)

# ==========================================
# 3. 메인 파이프라인 (Main Pipeline)
# ==========================================
def main():
    jarvis_engine, tools = initialize_jarvis()
    tools_map = {tool.name: tool for tool in tools}
    
    chat_history = [
        SystemMessage(content="""
        You are 'Jarvis Core', an elite AI architecture reviewer and strict technical interviewer.
        
        [User Context]
        - User Name: 지웅 (Jiwoong)
        - Location: Carrollton, Texas (Timezone: CST)
        - Current Focus: Developing 'smart-freight-ai' and studying C programming.
        
        [Core Directives & Rules]
        1. [Language] ALWAYS answer in Korean concisely, but MUST use precise English terminology for engineering concepts (e.g., Memory Leak, Pointer Aliasing, Time Complexity).
        2. [Tool Routing] 
           - For past rules, vehicle info, or project context -> USE 'search_memory'.
           - For real-time news or external facts -> USE 'web_search'.
           - For code analysis -> USE 'read_local_file'.
        3. [Interviewer Mode] When the user provides C code or algorithm designs:
           - NEVER just summarize. You MUST aggressively hunt for vulnerabilities.
           - Check strictly for Memory Leaks (malloc/free pairs), Dangling Pointers, and Buffer Overflows.
           - Analyze the Big-O Time/Space Complexity.
           - ALWAYS end your response with a sharp, challenging follow-up question (꼬리 질문) like a strict Silicon Valley interviewer.
        4. [Tool Restriction] YOU MUST ONLY USE THE PROVIDED TOOLS. NEVER invent or hallucinate tool names like 'weather_api'. If you need weather info, use 'web_search'.
        5. [Output Format] NEVER output raw JSON to the user. All JSON tool calls must be processed silently by the system.
        6. [Zero-Hallucination Policy] If a tool returns an error message (e.g., "[Error]"), YOU MUST NOT invent or guess the answer. You MUST explicitly tell the user that the tool failed and provide the exact error reason.
        """)
    ]
    
    print("\n[System] Jarvis Agent 세션이 시작되었습니다. (종료: 'quit')")
    
    while True:
        user_input = input("\n[User] ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        chat_history.append(HumanMessage(content=user_input))
        
        try:
            # 1차 추론: 라우팅 결정
            response = jarvis_engine.invoke(chat_history)
            chat_history.append(response)
            
            # 도구 사용 요청이 있는 경우
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    print(f"[System] ⚡️ Jarvis가 도구({tool_name})를 실행하려 합니다...")
                    
                    # [보안 검증 단계: HITL]
                    if tool_name == "delete_database_table":
                        approval = input(f"[경고] 삭제 요청 승인? (y/n): ")
                        if approval.lower() != 'y':
                            chat_history.append(ToolMessage(tool_call_id=tool_call["id"], name=tool_name, content="User denied."))
                            continue
                    
                    # [방어적 프로그래밍] 도구 실행 및 에러 캐치
                    try:
                        # 1. KeyError 방어: 모델이 이상한 도구를 부르면 강제 예외 처리
                        if tool_name.lower() not in tools_map:
                            raise KeyError(f"'{tool_name}' 도구는 존재하지 않습니다.")
                            
                        selected_tool = tools_map[tool_name.lower()]
                        tool_output = selected_tool.invoke(tool_call["args"])
                        
                        # [가시성 확보] 도구가 가져온 실제 Raw Data를 터미널에 일부 출력 (디버깅용)
                        print(f"[System Debug] 🔍 도구 반환값(미리보기): {str(tool_output)[:150]}...")
                        
                    except KeyError as e:
                        tool_output = f"[System Fallback] {e} 사용할 수 있는 도구만 사용하세요."
                        print(f"[Error] 모델이 잘못된 도구를 호출했습니다: {tool_name}")
                    except Exception as e:
                        tool_output = f"[System Fallback] 도구 실행 실패: {str(e)}"
                        print(f"[Error] 도구 실행 중 에러: {e}")
                    
                    # 결과를 메모리에 저장 (Observation)
                    chat_history.append(ToolMessage(tool_call_id=tool_call["id"], name=tool_name, content=str(tool_output)))
                
                # 2차 추론: 도구 결과를 바탕으로 최종 답변 생성
                final_response = jarvis_engine.invoke(chat_history)
                chat_history.append(final_response)
                
                # [빈 문자열 방어] LLM이 침묵할 경우의 안전장치
                if not final_response.content.strip():
                    print("[Jarvis] (데이터를 확인했으나 답변 생성 중 오류가 발생했습니다. 도구의 Raw Data를 확인하십시오.)")
                else:
                    print(f"[Jarvis] {final_response.content}")
                
            # 도구 사용 요청이 없는 일반 대화
            else:
                print(f"[Jarvis] {response.content}")
                
        except Exception as e:
            print(f"[Critical Error] Agent 파이프라인 붕괴: {e}")

if __name__ == "__main__":
    main()