"""鲸忆 v1 —— 给 AstrBot 一套真正的长期记忆。

三层结构：
1. turns   —— 每一轮对话落 sqlite，jieba 分词 + FTS5 索引，可秒级检索
2. digests —— 每天把前一天压缩成一段日记，让记忆能撑过几个月
3. inject  —— 每次 LLM 请求前按当前话题翻旧事，作为临时内容块注入

设计约束：
- 不改 system_prompt、不改用户原始消息，避免破坏上下文缓存前缀
- 按 scope（群 / 私聊）隔离，不在 A 群看到 B 群的记忆
- 纯本地 sqlite，零外部依赖，不上传任何数据
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import jieba
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

try:  # 官方推荐的内容块类型
    from astrbot.core.agent.message import TextPart
except Exception:  # pragma: no cover
    TextPart = None  # type: ignore[assignment]

PLUGIN_ID = "astrbot_plugin_whale_memory"
TAG = "[whale_memory]"

# 检索时忽略的虚词，避免「的了吗」这种把一切都匹配上
STOPWORDS = {
    "我们", "你们", "他们", "什么", "怎么", "这个", "那个", "就是", "不是",
    "可以", "现在", "今天", "明天", "昨天", "一下", "还是", "这样", "那样",
    "然后", "因为", "所以", "但是", "如果", "已经", "知道", "觉得", "应该",
}


@register(
    PLUGIN_ID,
    "大肥鱼",
    "长期记忆：对话落库、每日日记、按需检索注入",
    "1.0.0",
)
class WhaleMemory(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self._base = (
            Path(__file__).resolve().parents[2] / "plugin_data" / PLUGIN_ID
        )
        self._db_path = self._base / "memory.db"
        self._conn: sqlite3.Connection | None = None
        self._task: asyncio.Task | None = None
        self._last_turn_key: dict[str, tuple[str, float]] = {}
        self._digest_lock = asyncio.Lock()

    # ------------------------------------------------------------------ 基础
    def _cfg(self, key: str, default=None):
        try:
            val = self.config.get(key)
        except Exception:
            return default
        return default if val is None else val

    def _debug(self, msg: str) -> None:
        if self._cfg("debug", False):
            logger.info(f"{TAG} {msg}")

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._base.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema(conn)
            self._conn = conn
        return self._conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL NOT NULL,
                day         TEXT NOT NULL,
                scope       TEXT NOT NULL,
                sender_id   TEXT,
                sender_name TEXT,
                role        TEXT NOT NULL,
                text        TEXT NOT NULL,
                tok         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_turns_scope_ts ON turns(scope, ts);
            CREATE INDEX IF NOT EXISTS idx_turns_day ON turns(day);

            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                tok,
                content='turns',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 0'
            );

            CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
                INSERT INTO turns_fts(rowid, tok) VALUES (new.id, new.tok);
            END;
            CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
                INSERT INTO turns_fts(turns_fts, rowid, tok)
                VALUES ('delete', old.id, old.tok);
            END;

            CREATE TABLE IF NOT EXISTS digests (
                day        TEXT NOT NULL,
                scope      TEXT NOT NULL,
                summary    TEXT NOT NULL,
                msg_count  INTEGER DEFAULT 0,
                created_ts REAL NOT NULL,
                PRIMARY KEY (day, scope)
            );
            """
        )
        conn.commit()

    # ------------------------------------------------------------------ 分词
    @staticmethod
    def _tokenize(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        out: list[str] = []
        for w in jieba.cut_for_search(text):
            w = w.strip()
            if not w:
                continue
            if not re.search(r"[\w\u4e00-\u9fff]", w):
                continue
            if len(w) == 1 and "\u4e00" <= w <= "\u9fff":
                continue  # 单字中文噪声太大
            if w in STOPWORDS:
                continue
            out.append(w)
        # 中文二元组补充：jieba 认不出的专名（四会、老王、显卡）靠这一层兜底
        for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            for i in range(len(run) - 1):
                bg = run[i : i + 2]
                if bg not in STOPWORDS:
                    out.append(bg)
        # 去重保序
        seen: set[str] = set()
        uniq = [w for w in out if not (w in seen or seen.add(w))]
        return " ".join(uniq)

    # ------------------------------------------------------------------ 写入
    def _scope_of(self, event: AstrMessageEvent) -> str:
        try:
            if event.is_private_chat():
                return f"friend:{event.get_sender_id()}"
            gid = event.get_group_id()
            if gid:
                return f"group:{gid}"
        except Exception:
            pass
        return f"umo:{getattr(event, 'unified_msg_origin', 'unknown')}"

    def _add_turn(
        self,
        *,
        scope: str,
        role: str,
        text: str,
        sender_id: str = "",
        sender_name: str = "",
    ) -> None:
        text = " ".join((text or "").split()).strip()
        if not text:
            return
        if len(text) > 1200:
            text = text[:1200] + "…"
        now = time.time()
        day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        tok = self._tokenize(text)
        try:
            cur = self.conn
            cur.execute(
                "INSERT INTO turns (ts, day, scope, sender_id, sender_name, role, text, tok)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (now, day, scope, sender_id, sender_name, role, text, tok),
            )
            cur.commit()
            self._debug(f"记录 {scope} {role} {text[:30]}")
        except Exception as exc:
            logger.warning(f"{TAG} 写入失败: {exc}")

    # ------------------------------------------------------------------ 检索
    def _search(self, scope: str, query: str, limit: int) -> list[sqlite3.Row]:
        tok_str = self._tokenize(query)
        tokens = tok_str.split()
        if not tokens:
            return []
        # 抽稀长查询，最多取 12 个词，避免布尔查询爆炸
        tokens = tokens[:12]
        expr = " OR ".join(f'"{t}"' for t in tokens)
        try:
            rows = self.conn.execute(
                """
                SELECT t.*, bm25(turns_fts) AS rank
                FROM turns_fts
                JOIN turns t ON t.id = turns_fts.rowid
                WHERE turns_fts MATCH ? AND t.scope = ? AND t.role IN ('user','assistant')
                ORDER BY rank LIMIT ?
                """,
                (expr, scope, max(limit * 4, 12)),
            ).fetchall()
        except Exception as exc:
            logger.warning(f"{TAG} 检索失败: {exc}")
            return []
        if not rows:
            return []
        # 时间衰减：越新越靠前，但别让旧事永远翻不出来
        now = time.time()

        def score(r: sqlite3.Row) -> float:
            age_days = max((now - r["ts"]) / 86400.0, 0.0)
            recency = 1.0 / (1.0 + age_days / 30.0)
            base = 1.0 / (1.0 + abs(r["rank"]))
            return base * 0.75 + recency * 0.25

        rows = sorted(rows, key=score, reverse=True)
        return rows[:limit]

    def _digests_for(self, scope: str, query: str, limit: int) -> list[sqlite3.Row]:
        tokens = self._tokenize(query).split()[:8]
        if not tokens:
            return []
        like = " OR ".join(["summary LIKE ?"] * len(tokens))
        params = [f"%{t}%" for t in tokens] + [scope, limit]
        try:
            return self.conn.execute(
                f"SELECT * FROM digests WHERE ({like}) AND scope = ? "
                "ORDER BY day DESC LIMIT ?",
                params,
            ).fetchall()
        except Exception as exc:
            logger.warning(f"{TAG} 日记检索失败: {exc}")
            return []

    def _build_injection(self, scope: str, query: str) -> str:
        count = int(self._cfg("inject_count", 5) or 5)
        max_chars = int(self._cfg("inject_max_chars", 800) or 800)
        lines: list[str] = []

        for d in self._digests_for(scope, query, 1):
            summary = " ".join(str(d["summary"]).split())
            if summary:
                lines.append(f"[{d['day']} 的日记] {summary[:220]}")

        for r in self._search(scope, query, count):
            ts = datetime.fromtimestamp(r["ts"]).strftime("%m-%d %H:%M")
            who = "她" if r["role"] == "assistant" else (r["sender_name"] or "群友")
            text = " ".join(str(r["text"]).split())
            if len(text) > 110:
                text = text[:110] + "…"
            lines.append(f"- {ts} {who}：{text}")

        if not lines:
            return ""

        body = "\n".join(lines)
        if len(body) > max_chars:
            body = body[:max_chars] + "\n…（后略）"

        header = str(
            self._cfg(
                "inject_header",
                "【关于过去的记忆】以下是从长期记忆里翻出的旧事，和当前话题可能有关。"
                "用得上就自然带上，用不上就别硬提，也不要补充记忆里没有的细节。\n{}",
            )
        )
        if "{}" in header:
            return header.replace("{}", body)
        return f"{header}\n{body}"

    @staticmethod
    def _append_part(req: ProviderRequest, text: str) -> bool:
        if not text.strip() or TextPart is None:
            return False
        try:
            part = TextPart(text=text)
            mark = getattr(part, "mark_as_temp", None)
            if callable(mark):
                part = mark()
            req.extra_user_content_parts.append(part)
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning(f"{TAG} 注入失败: {exc}")
            return False

    # ------------------------------------------------------------------ 钩子
    @filter.on_llm_request()
    async def on_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if not self._cfg("enable", True):
            return
        scope = self._scope_of(event)
        try:
            user_text = event.get_message_str() or ""
        except Exception:
            user_text = getattr(req, "prompt", "") or ""

        # ---- 记录用户这一轮（同一请求重复触发时去重）
        if self._cfg("record_enable", True) and user_text.strip():
            key = hashlib.md5(user_text.encode("utf-8", "ignore")).hexdigest()
            last = self._last_turn_key.get(scope)
            now = time.time()
            if not (last and last[0] == key and now - last[1] < 15):
                self._last_turn_key[scope] = (key, now)
                try:
                    self._add_turn(
                        scope=scope,
                        role="user",
                        text=user_text,
                        sender_id=str(event.get_sender_id() or ""),
                        sender_name=str(event.get_sender_name() or ""),
                    )
                except Exception as exc:
                    logger.warning(f"{TAG} 记录用户消息失败: {exc}")

        # ---- 注入旧事
        if not self._cfg("inject_enable", True):
            return
        min_len = int(self._cfg("min_query_len", 4) or 4)
        if len(user_text.strip()) < min_len:
            return
        try:
            block = self._build_injection(scope, user_text)
        except Exception as exc:
            logger.warning(f"{TAG} 组装记忆失败: {exc}")
            return
        if block and self._append_part(req, block):
            self._debug(f"注入 {len(block)} 字")

    @filter.on_llm_response()
    async def on_response(self, event: AstrMessageEvent, response) -> None:
        if not self._cfg("enable", True) or not self._cfg("record_enable", True):
            return
        text = ""
        for attr in ("completion_text", "_completion_text"):
            val = getattr(response, attr, None)
            if isinstance(val, str) and val.strip():
                text = val
                break
        if not text:
            chain = getattr(response, "result_chain", None)
            if chain is not None:
                try:
                    text = chain.get_plain_text()
                except Exception:
                    text = ""
        if not text.strip():
            return
        try:
            self._add_turn(scope=self._scope_of(event), role="assistant", text=text)
        except Exception as exc:
            logger.warning(f"{TAG} 记录回复失败: {exc}")

    # ------------------------------------------------------------------ 日记
    async def _made_digest(self, day: str, scope: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM digests WHERE day=? AND scope=?", (day, scope)
        ).fetchone()
        return row is not None

    async def _write_digest(self, day: str, scope: str, umo: str) -> None:
        rows = self.conn.execute(
            "SELECT sender_name, role, text FROM turns "
            "WHERE day=? AND scope=? ORDER BY ts LIMIT 400",
            (day, scope),
        ).fetchall()
        if len(rows) < 6:
            return

        chunks: list[str] = []
        for r in rows:
            who = "她" if r["role"] == "assistant" else (r["sender_name"] or "群友")
            chunks.append(f"{who}：{' '.join(str(r['text']).split())[:160]}")
        transcript = "\n".join(chunks)[:6000]

        prompt = (
            "下面是一天里一段聊天记录。请用第三人称写一段 150-250 字的日记式摘要，"
            "只保留以后还可能有用的事实：谁是谁、说过要做的事、偏好、近况、约定、情绪变化。"
            "不要写客套话，不要评价，不要编造记录里没有的内容。直接输出摘要正文。\n\n"
            f"{transcript}"
        )
        try:
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt="你是一个只做事实压缩的记录员，禁止发挥。",
            )
            summary = (getattr(resp, "completion_text", "") or "").strip()
        except Exception as exc:
            logger.warning(f"{TAG} 生成日记失败 {day}/{scope}: {exc}")
            return
        if not summary:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO digests (day, scope, summary, msg_count, created_ts)"
            " VALUES (?,?,?,?,?)",
            (day, scope, summary[:1200], len(rows), time.time()),
        )
        self.conn.commit()
        logger.info(f"{TAG} 日记完成 {day} {scope} ({len(rows)} 条)")

    async def _digest_loop(self) -> None:
        await asyncio.sleep(60)
        while True:
            try:
                if self._cfg("digest_enable", True):
                    await self._maybe_digest()
                    self._cleanup()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"{TAG} 日记循环异常: {exc}")
            await asyncio.sleep(1800)

    async def _maybe_digest(self) -> None:
        hour = int(self._cfg("digest_hour", 3) or 3)
        if datetime.now().hour < hour:
            return
        async with self._digest_lock:
            today = datetime.now().date()
            target = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            scopes = self.conn.execute(
                "SELECT scope, COUNT(*) c, MAX(ts) m FROM turns WHERE day=? GROUP BY scope",
                (target,),
            ).fetchall()
            budget = int(self._cfg("digest_max_per_run", 3) or 3)
            for s in scopes:
                if budget <= 0:
                    break
                if await self._made_digest(target, s["scope"]):
                    continue
                umo = ""
                try:
                    row = self.conn.execute(
                        "SELECT scope FROM turns WHERE scope=? LIMIT 1", (s["scope"],)
                    ).fetchone()
                    umo = str(row["scope"]) if row else ""
                except Exception:
                    pass
                await self._write_digest(target, s["scope"], umo)
                budget -= 1

    def _cleanup(self) -> None:
        days = int(self._cfg("retention_days", 180) or 0)
        if days <= 0:
            return
        cutoff = time.time() - days * 86400
        try:
            cur = self.conn.execute("DELETE FROM turns WHERE ts < ?", (cutoff,))
            self.conn.commit()
            if cur.rowcount:
                logger.info(f"{TAG} 清理 {cur.rowcount} 条过期对话")
        except Exception as exc:
            logger.warning(f"{TAG} 清理失败: {exc}")

    # ------------------------------------------------------------------ 指令
    @filter.command("记")
    async def cmd_remember(self, event: AstrMessageEvent):
        text = (event.get_message_str() or "").strip()
        text = re.sub(r"^[/／]?记\s*", "", text).strip()
        if not text:
            yield event.plain_result("记什么？说清楚点。")
            return
        self._add_turn(
            scope=self._scope_of(event),
            role="user",
            text=f"（大哥让她记住：{text}）",
            sender_id=str(event.get_sender_id() or ""),
            sender_name=str(event.get_sender_name() or ""),
        )
        yield event.plain_result(f"记住了：{text[:60]}")

    @filter.command("回忆")
    async def cmd_recall(self, event: AstrMessageEvent):
        query = re.sub(r"^[/／]?回忆\s*", "", (event.get_message_str() or "")).strip()
        if not query:
            yield event.plain_result("想回忆什么？给个关键词。")
            return
        scope = self._scope_of(event)
        hits = self._search(scope, query, 5)
        digs = self._digests_for(scope, query, 2)
        if not hits and not digs:
            yield event.plain_result("这件事我没印象。")
            return
        lines: list[str] = []
        for d in digs:
            lines.append(f"【{d['day']}】{str(d['summary'])[:150]}")
        for r in hits:
            ts = datetime.fromtimestamp(r["ts"]).strftime("%m-%d %H:%M")
            who = "我" if r["role"] == "assistant" else (r["sender_name"] or "群友")
            lines.append(f"{ts} {who}：{str(r['text'])[:80]}")
        yield event.plain_result("\n".join(lines[:8]))

    @filter.command("记忆状态")
    async def cmd_status(self, event: AstrMessageEvent):
        scope = self._scope_of(event)
        try:
            total = self.conn.execute("SELECT COUNT(*) c FROM turns").fetchone()["c"]
            mine = self.conn.execute(
                "SELECT COUNT(*) c FROM turns WHERE scope=?", (scope,)
            ).fetchone()["c"]
            dg = self.conn.execute(
                "SELECT COUNT(*) c FROM digests WHERE scope=?", (scope,)
            ).fetchone()["c"]
            oldest = self.conn.execute(
                "SELECT MIN(ts) m FROM turns WHERE scope=?", (scope,)
            ).fetchone()["m"]
        except Exception as exc:
            yield event.plain_result(f"读不出来：{exc}")
            return
        since = (
            datetime.fromtimestamp(oldest).strftime("%Y-%m-%d") if oldest else "还没有"
        )
        yield event.plain_result(
            f"记忆库：全站 {total} 条，这里 {mine} 条，日记 {dg} 篇，最早 {since}。"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("忘记")
    async def cmd_forget(self, event: AstrMessageEvent):
        kw = re.sub(r"^[/／]?忘记\s*", "", (event.get_message_str() or "")).strip()
        if not kw:
            yield event.plain_result("给个关键词，我删掉含这个词的记忆。")
            return
        scope = self._scope_of(event)
        try:
            cur = self.conn.execute(
                "DELETE FROM turns WHERE scope=? AND text LIKE ?", (scope, f"%{kw}%")
            )
            self.conn.commit()
        except Exception as exc:
            yield event.plain_result(f"删失败：{exc}")
            return
        yield event.plain_result(f"删了 {cur.rowcount} 条含「{kw}」的记忆。")

    # ------------------------------------------------------------------ 生命周期
    async def initialize(self) -> None:
        try:
            _ = self.conn  # 建库
        except Exception as exc:
            logger.error(f"{TAG} 初始化数据库失败: {exc}")
            return
        if self._cfg("digest_enable", True):
            self._task = asyncio.create_task(self._digest_loop())
        logger.info(f"{TAG} 已加载，库文件 {self._db_path}")

    async def terminate(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
        if self._conn:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass
            self._conn = None
