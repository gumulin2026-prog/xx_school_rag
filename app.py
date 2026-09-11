# app.py
# 基于FastAPI搭建智能问答系统Web服务

import os
import json
import uuid
import time
import re
import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel

# 导入你的项目模块
from base.config import Config
from base.logger import logger
from core.mysql_client import MySQLClient
from core.rag_system import RAGSystem

# ============================================================
# 1. FastAPI 应用初始化
# ============================================================
app = FastAPI(title="XX大学迎新问答助手 API", description="集成 MySQL + BM25 + Milvus 的迎新智能问答系统")

# CORS 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录
os.makedirs('static', exist_ok=True)
app.mount('/static', StaticFiles(directory='static'), name='static')

# ============================================================
# 2. 全局初始化
# ============================================================
config = Config()
mysql_client = MySQLClient()
rag_system = RAGSystem()

# 确保数据库表存在
mysql_client.create_table()

# ============================================================
# 3. 问候语匹配
# ============================================================
GREETING_PATTERNS = [
    {"pattern": r"^(你好|您好|hi|hello)", "response": "你好！我是 XX 大学迎新助手，很高兴为你服务！"},
    {"pattern": r"^(你是谁|你叫什么)", "response": "我是 XX 大学迎新助手，一个基于 RAG 技术的问答助手，可以解答报到、住宿、选课等新生问题。"},
    {"pattern": r"^(在吗|在不在)", "response": "我在！随时为你解答新生相关的问题。"},
]

def check_greeting(query: str) -> Optional[str]:
    for pattern in GREETING_PATTERNS:
        if re.match(pattern["pattern"], query.strip(), re.IGNORECASE):
            return pattern["response"]
    return None

# ============================================================
# 4. Pydantic 模型
# ============================================================
class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float

# ============================================================
# 5. HTTP 接口
# ============================================================

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/sources")
async def get_sources():
    return {"sources": config.VALID_SOURCES}

@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    mysql_client.cursor.execute(
        "INSERT INTO sessions (session_id) VALUES (%s)", (session_id,)
    )
    mysql_client.connection.commit()
    return {"session_id": session_id}

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        history = mysql_client.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史失败: {str(e)}")

@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    try:
        mysql_client.clear_session_history(session_id)
        return {"status": "success", "message": "历史已清除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除历史失败: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    # 检查问候语
    greeting = check_greeting(request.query)
    if greeting:
        mysql_client.update_session_history(session_id, request.query, greeting)
        return QueryResponse(
            answer=greeting,
            is_streaming=False,
            session_id=session_id,
            processing_time=time.time() - start_time
        )

    # 先尝试 BM25
    answer, need_rag = rag_system.bm25_search.search(request.query)

    if answer:
        # BM25 命中，直接返回，并保存到对话历史（否则刷新页面这轮问答会丢失，会话列表的消息数也不准）
        mysql_client.update_session_history(session_id, request.query, answer)
        return QueryResponse(
            answer=answer,
            is_streaming=False,
            session_id=session_id,
            processing_time=time.time() - start_time
        )

    # BM25 未命中，提示使用 WebSocket
    return QueryResponse(
        answer="请使用 WebSocket 接口获取流式响应",
        is_streaming=True,
        session_id=session_id,
        processing_time=time.time() - start_time
    )

@app.get("/api/sessions")
async def get_sessions():
    """获取所有历史会话列表"""
    try:
        sql = """
            SELECT
                s.session_id,
                s.created_at,
                COUNT(c.id) as message_count,
                (SELECT question FROM conversations c2
                 WHERE c2.session_id = s.session_id
                 ORDER BY timestamp DESC LIMIT 1) as last_question
            FROM sessions s
            LEFT JOIN conversations c ON c.session_id = s.session_id
            GROUP BY s.session_id, s.created_at
            ORDER BY s.created_at DESC
            LIMIT 50
        """
        mysql_client.cursor.execute(sql)
        results = mysql_client.cursor.fetchall()
        sessions = []
        for row in results:
            sessions.append({
                "session_id": row[0],
                "created_at": row[1].strftime("%Y-%m-%d %H:%M") if row[1] else "",
                "message_count": row[2],
                "last_question": row[3] if row[3] else "空对话"
            })
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        return {"sessions": []}


@app.get("/api/session/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取指定会话的所有消息"""
    try:
        history = mysql_client.get_session_history(session_id, limit=100)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        logger.error(f"获取会话消息失败: {e}")
        return {"session_id": session_id, "messages": []}


@app.post("/api/session/{session_id}/switch")
async def switch_session(session_id: str):
    """切换当前会话（仅验证会话是否存在）"""
    try:
        # 检查会话是否存在（查 sessions 注册表，而不是 conversations——新会话还没消息时后者查不到）
        sql = "SELECT 1 FROM sessions WHERE session_id = %s LIMIT 1"
        mysql_client.cursor.execute(sql, (session_id,))
        if mysql_client.cursor.fetchone():
            return {"success": True, "session_id": session_id}
        else:
            return {"success": False, "message": "会话不存在"}
    except Exception as e:
        logger.error(f"切换会话失败: {e}")
        return {"success": False, "message": str(e)}

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    try:
        mysql_client.cursor.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
        mysql_client.cursor.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        mysql_client.connection.commit()
        logger.info(f"会话 {session_id} 已删除")
        return {"success": True, "message": "会话已删除"}
    except Exception as e:
        logger.error(f"删除会话失败: {e}")
        return {"success": False, "message": str(e)}

# ============================================================
# 6. WebSocket 流式接口
# ============================================================

@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)

            query = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()

            # 发送开始信号
            await websocket.send_json({"type": "start", "session_id": session_id})

            # 检查问候语
            greeting = check_greeting(query)
            if greeting:
                await websocket.send_json({"type": "token", "token": greeting, "session_id": session_id})
                await websocket.send_json({
                    "type": "end",
                    "session_id": session_id,
                    "is_complete": True,
                    "processing_time": time.time() - start_time
                })
                continue

            # 获取历史
            history = mysql_client.get_session_history(session_id)

            # 流式生成答案
            full_answer = ""
            for token in rag_system.generate_answer(
                query=query,
                source_filter=source_filter,
                history=history,
                stream=True
            ):
                if token:
                    full_answer += token
                    await websocket.send_json({"type": "token", "token": token, "session_id": session_id})
                await asyncio.sleep(0.01)

            # 保存对话历史
            if full_answer and session_id:
                try:
                    mysql_client.update_session_history(session_id, query, full_answer)
                except Exception as e:
                    logger.error(f"保存历史失败: {e}")

            # 发送结束信号
            await websocket.send_json({
                "type": "end",
                "session_id": session_id,
                "is_complete": True,
                "processing_time": time.time() - start_time
            })

    except WebSocketDisconnect:
        logger.info("WebSocket 断开连接")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass
        finally:
            await websocket.close()

# ============================================================
# 7. 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(app, host=host, port=port, reload=False)