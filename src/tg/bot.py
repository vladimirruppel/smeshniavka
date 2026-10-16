import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import UpdateType, ChatType
from aiogram.filters import Command

from src.ops.config import AppConfig
from src.ops.log import logger, shadow_logger
from src.db.manager import DatabaseManager
from src.db.contracts import TgSender
from src.gateway.client import GeminiGateway
from src.context.assemble import ContextAssembler, scrub_pii
from src.context.prompts import PromptBuilder, sanitize_nick
from src.persona.reply import PingDetector, AuthorSpamLimiter, TgSendQueue
from src.persona.proactive import ProactiveEngine
from src.persona.mood import detect_mood
from src.persona.sanitizer import (
    has_comma_glue,
    is_repetitive,
    replace_un_aliases,
    has_persona_leak,
    clean_media_reply_boilerplate,
    validate_persona_reply,
)
from src.gateway.schemas import PersonaTextReply
from src.dossier.worker import DossierWorker
from src.media.analyzer import MediaAnalyzer, contains_forbidden_score, SCORE_PATTERN, build_step2_media_json
from src.research.search import ResearchModule, build_media_search_query
from src.tg.lanes import LanePool, LaneWorker, LaneJob, route_lane, LANE_CHAT, LANE_TRACK, LANE_MEDIA, LANE_SYS
from src.antijob.detector import ProvocationDetector
from src.style.learner import StyleLearner
from src.tg.reactions import ReactionsTracker
from src.tg.stubs import (
    format_sticker_stub,
    format_animation_stub,
    format_poll_stub,
    format_dice_stub,
    extract_clean_urls,
)
from dataclasses import dataclass
from src.tg.filters import RuntimeChatResolver, check_media_caps
from src.ops.alerts import AlertManager


def to_iso_date(dt_or_ts: Any) -> str:
    if isinstance(dt_or_ts, (int, float)):
        return datetime.fromtimestamp(dt_or_ts, tz=timezone.utc).isoformat()
    if isinstance(dt_or_ts, datetime):
        return dt_or_ts.astimezone(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExtractedMedia:
    media_kind: Optional[str] = None
    file_id: Optional[str] = None
    file_unique_id: Optional[str] = None
    file_size: int = 0
    duration: Optional[int] = None
    caption: Optional[str] = None
    text_stub: Optional[str] = None


def extract_media_info(message: types.Message) -> ExtractedMedia:
    media_kind = None
    file_unique_id = None
    file_id = None
    file_size = 0
    duration = None
    text_stub = None

    if message.sticker:
        media_kind = "sticker"
        text_stub = format_sticker_stub(message.sticker.emoji, message.sticker.set_name)
        if (getattr(message.sticker, "is_animated", False) or getattr(message.sticker, "is_video", False)) and getattr(message.sticker, "thumbnail", None):
            file_unique_id = message.sticker.thumbnail.file_unique_id
            file_id = message.sticker.thumbnail.file_id
            file_size = getattr(message.sticker.thumbnail, "file_size", 0) or 0
        else:
            file_unique_id = message.sticker.file_unique_id
            file_id = message.sticker.file_id
            file_size = getattr(message.sticker, "file_size", 0) or 0
    elif message.animation:
        media_kind = "animation"
        file_unique_id = message.animation.file_unique_id
        file_id = message.animation.file_id
        file_size = message.animation.file_size or 0
        duration = message.animation.duration
        text_stub = format_animation_stub(message.caption)
    elif message.voice:
        media_kind = "voice"
        file_unique_id = message.voice.file_unique_id
        file_id = message.voice.file_id
        file_size = message.voice.file_size or 0
        duration = message.voice.duration
        text_stub = "[войс]"
    elif message.video_note:
        media_kind = "video_note"
        file_unique_id = message.video_note.file_unique_id
        file_id = message.video_note.file_id
        file_size = message.video_note.file_size or 0
        duration = message.video_note.duration
        text_stub = "[кружочек]"
    elif message.video:
        media_kind = "video"
        file_unique_id = message.video.file_unique_id
        file_id = message.video.file_id
        file_size = message.video.file_size or 0
        duration = message.video.duration
        text_stub = "[видео]"
    elif message.photo:
        media_kind = "photo"
        photo_obj = message.photo[-1]
        file_unique_id = photo_obj.file_unique_id
        file_id = photo_obj.file_id
        file_size = photo_obj.file_size or 0
        text_stub = "[фото]"
    elif message.audio:
        media_kind = "audio"
        file_unique_id = message.audio.file_unique_id
        file_id = message.audio.file_id
        file_size = message.audio.file_size or 0
        duration = message.audio.duration
        meta_parts = []
        if message.audio.performer:
            meta_parts.append(message.audio.performer)
        meta_str = ' - '.join(meta_parts)
        if meta_str:
            meta_str = scrub_pii(meta_str)
        audio_info = f" ({meta_str})" if meta_str else ""
        text_stub = f"[аудио{audio_info}]"
    elif message.poll:
        media_kind = "poll"
        opts = [o.text for o in message.poll.options]
        text_stub = format_poll_stub(message.poll.question, opts)
    elif message.dice:
        media_kind = "dice"
        text_stub = format_dice_stub(message.dice.emoji, message.dice.value)
    elif message.contact:
        media_kind = "contact"
        text_stub = "[контакт]"
    elif message.location or message.venue:
        media_kind = "location"
        text_stub = "[геопозиция]"

    return ExtractedMedia(
        media_kind=media_kind,
        file_id=file_id,
        file_unique_id=file_unique_id,
        file_size=file_size,
        duration=duration,
        caption=message.caption,
        text_stub=text_stub,
    )


class RealTgSender(TgSender):
    """Real Telegram sender using aiogram Bot instance."""

    def __init__(self, bot: Bot):
        self.bot = bot

    async def send_message(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
        parse_mode: Optional[str] = "HTML",
        disable_web_page_preview: bool = True
    ) -> Optional[int]:
        res = await self.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_to_message_id=reply_to_message_id,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview
        )
        return res.message_id

    async def leave_chat(self, chat_id: int) -> bool:
        return await self.bot.leave_chat(chat_id)

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        await self.bot.send_chat_action(chat_id=chat_id, action=action)


class FunnyBotApp:
    """Main Telegram Bot Application coordinating all subsystems."""

    def __init__(
        self,
        config: AppConfig,
        db_manager: DatabaseManager,
        bot: Optional[Bot] = None
    ):
        self.config = config
        self.bot_cfg = config.bot
        self.db = db_manager

        self.bot = bot or Bot(token=config.tg_token)
        self.dp = Dispatcher()
        self.sender = RealTgSender(self.bot)

        self.chat_resolver = RuntimeChatResolver(self.bot_cfg.ALLOWED_CHAT_ID)
        self.alert_mgr = AlertManager(self.sender, self.bot_cfg.OWNER_IDS)
        self.gateway = GeminiGateway(config)
        self.prompts = PromptBuilder()
        self.assembler = ContextAssembler(self.bot_cfg, self.db, self.prompts)
        self.send_queue = TgSendQueue(self.sender, self.bot_cfg)
        self.proactive = ProactiveEngine(self.bot_cfg)
        self.author_limiter = AuthorSpamLimiter(self.bot_cfg, config.persona)
        self.dossier_worker = DossierWorker(self.bot_cfg, self.db, self.gateway, self.prompts)
        self.media_analyzer = MediaAnalyzer(self.bot_cfg, self.db, self.gateway, self.prompts)
        self.research = ResearchModule(self.bot_cfg, self.db, self.gateway, self.prompts)
        self.style_learner = StyleLearner(self.bot_cfg, self.db, self.gateway)
        self.send_queue.on_sent = self._on_bot_sent
        self.reactions = ReactionsTracker(
            laugh_emojis=set(getattr(self.bot_cfg, "REACT_LAUGH_EMOJIS", None) or {"😁", "😂", "🤣", "😆", "😹", "🤭"}),
            threshold=int(getattr(self.bot_cfg, "REACT_LAUGH_THRESHOLD", 4) or 4),
        )

        self.ping_detector: Optional[PingDetector] = None
        self.bot_id = 0
        self.bot_username = ""
        self.start_time = time.time()
        self.muted_until: float = 0.0

        # Воркеры: chat (общение) / track x2 (треки) / media (остальное) / sys (фон).
        # У трековых воркеров разные имена → разные тёплые сессии моста.
        qmax = int(getattr(self.bot_cfg, "WORKER_QUEUE_MAX", 30) or 30)
        self.lanes = LanePool(
            chat=LaneWorker(LANE_CHAT, max_queue=qmax),
            track=LaneWorker(f"{LANE_TRACK}1", max_queue=qmax),
            track2=LaneWorker(f"{LANE_TRACK}2", max_queue=qmax),
            media=LaneWorker(LANE_MEDIA, max_queue=qmax),
            sys=LaneWorker(LANE_SYS, max_queue=qmax),
        )

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        self.dp.message.register(self.handle_message)
        self.dp.edited_message.register(self.handle_edited_message)
        self.dp.my_chat_member.register(self.handle_my_chat_member)
        self.dp.message_reaction.register(self.handle_reaction)

    async def _on_bot_sent(
        self, chat_id: int, message_id: int, text: str, reply_to: Optional[int]
    ) -> None:
        """Пишем свои исходящие в БД: модель видит свои реплики в истории,
        а смеяки на них резолвят текст для прайм-кандидатов. Best-effort."""
        try:
            await self.db.save_message({
                "chat_id": chat_id,
                "message_id": message_id,
                "user_id": self.bot_id,
                "alias": f"u{self.bot_id}" if self.bot_id else "u0",
                "nick": "Дик",
                "text": text,
                "reply_to": reply_to,
                "is_bot": 1,
                "created_at": to_iso_date(time.time()),
            })
        except Exception as e:
            logger.warning("Failed to save outgoing message %s/%s: %s", chat_id, message_id, e)

    async def handle_reaction(self, event: Any) -> None:
        """Смеяки: 3+ distinct — сразу в банк, 1–2 — в очередь на отбор.

        В банк кладём ответ вместе с контекстом (на что отвечали), чтобы
        пример нёс ситуацию. Чистку переполненного банка делает sys-воркер.
        """
        try:
            chat = event.chat
            if not self.chat_resolver.is_allowed(chat.id):
                return
            emojis: set[str] = set()
            for r in (getattr(event, "new_reaction", None) or []):
                if getattr(r, "type", None) == "emoji" and getattr(r, "emoji", None):
                    emojis.add(r.emoji)
            user = getattr(event, "user", None)
            signal = self.reactions.handle(
                chat.id, event.message_id,
                getattr(user, "id", None), emojis,
            )
            if not signal:
                return
            row = await self.db.get_message(chat.id, event.message_id)
            if not row or not (row.get("text") or "").strip():
                return
            # Контекст: на что был ответ (текст цели реплая, если есть)
            ctx = None
            try:
                target_id = row.get("reply_to")
                if target_id:
                    target = await self.db.get_message(chat.id, target_id)
                    if target and (target.get("text") or "").strip():
                        ctx = target["text"][:200]
            except Exception:
                ctx = None
            laughs = self.reactions.laugh_count(chat.id, event.message_id)
            added = await self.style_learner.collect_reacted(
                chat.id, row.get("user_id"), row.get("text"),
                direct=(signal == "bank"), ctx=ctx, laughs=laughs,
            )
            if added:
                job = LaneJob(
                    lane=LANE_SYS,
                    label=f"style-cap#{event.message_id}",
                    coro_factory=lambda: self.style_learner.ensure_capacity(),
                    enqueued_at=time.time(),
                )
                await self.lanes.get(LANE_SYS).submit(job)
        except Exception as e:
            logger.warning("Reaction handling failed: %s", e)

    async def initialize(self) -> None:
        me = await self.bot.get_me()
        self.bot_id = me.id
        self.bot_username = me.username or "funny_bot"
        self.ping_detector = PingDetector(self.bot_username, self.bot_id, self.bot_cfg)
        self.send_queue.start()
        self.lanes.start_all()
        logger.info("Bot initialized: @%s (id=%d) lanes=%s", self.bot_username, self.bot_id, self.lanes.sizes())

    async def shutdown(self) -> None:
        await self.lanes.stop_all()
        await self.send_queue.stop()

    def _lane_session(self, chat_id: int, lane: str) -> str:
        """Отдельная тёплая сессия моста на воркер: chat/track1/track2/media не вытесняют друг друга."""
        return f"tg_{chat_id}:{lane}"

    async def _typing_keepalive(self, chat_id: int, stop_event: asyncio.Event) -> None:
        """Человеческое «печатает…» на всё время долгой генерации, а не один раз."""
        interval = float(getattr(self.bot_cfg, "TYPING_KEEPALIVE_S", 4.0) or 4.0)
        while not stop_event.is_set():
            try:
                await self.sender.send_chat_action(chat_id, "typing")
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def handle_edited_message(self, message: types.Message) -> None:
        """TB-03: edit обновляет текст в БД, без ответов и без P1."""
        chat = message.chat
        if not self.chat_resolver.is_allowed(chat.id):
            return
        text = message.text or message.caption or ""
        try:
            await self.db.update_message_text(
                chat.id,
                message.message_id,
                text,
                to_iso_date(message.edit_date if message.edit_date is not None else message.date),
            )
        except Exception as e:
            logger.warning("Failed to update edited message %s/%s: %s", chat.id, message.message_id, e)

    async def handle_my_chat_member(self, event: types.ChatMemberUpdated) -> None:
        """I3/C6: чужой чат со статусом member/administrator — немедленный leaveChat."""
        chat = event.chat
        try:
            new_status = event.new_chat_member.status
        except Exception:
            return
        if self.chat_resolver.is_allowed(chat.id):
            return
        if str(new_status) not in ("member", "administrator", "creator"):
            return
        inviting_id = None
        try:
            inviting_id = event.from_user.id if event.from_user else None
        except Exception:
            pass
        logger.warning(
            "Bot added to foreign chat %s (%s) by %s. Leaving...",
            chat.id, getattr(chat, "title", ""), inviting_id,
        )
        await self.db.add_foreign_chat(chat.id, getattr(chat, "title", ""))
        try:
            await self.bot.leave_chat(chat.id)
        except Exception:
            pass
        await self.alert_mgr.send_alert(
            "FOREIGN_CHAT_LEFT",
            f"Бот добавлен в посторонний чат {chat.id} и покинул его.",
        )

    async def handle_message(self, message: types.Message) -> None:
        chat = message.chat
        user = message.from_user
        text = message.text or message.caption or ""

        # 1. Private chat handling
        if chat.type == ChatType.PRIVATE:
            if user and user.id in self.bot_cfg.OWNER_IDS:
                if text.strip().startswith("/start"):
                    await message.answer("Дик на связи. Алерты будут приходить сюда.")
                elif text.strip().startswith("/status"):
                    summary = self.gateway.get_key_status_summary()
                    await message.answer(f"Статус ключей: {json.dumps(summary, indent=2)}")
            return  # Non-owner DMs ignored per I3

        # 2. Check for chat migration (group -> supergroup)
        if message.migrate_to_chat_id:
            new_id = message.migrate_to_chat_id
            old_id = chat.id
            logger.info("Chat migration detected: %s -> %s", old_id, new_id)
            self.chat_resolver.set_allowed_chat_id(new_id)
            await self.db.migrate_chat_id(old_id, new_id)
            await self.alert_mgr.send_alert("CHAT_MIGRATION", f"Чат мигрировал из {old_id} в {new_id}")
            return

        # 3. Foreign chat filtering per I3
        if not self.chat_resolver.is_allowed(chat.id):
            logger.warning("Message from foreign chat %s (%s). Leaving chat...", chat.id, chat.title)
            await self.db.add_foreign_chat(chat.id, chat.title)
            try:
                await self.bot.leave_chat(chat.id)
            except Exception:
                pass
            await self.alert_mgr.send_alert(
                "FOREIGN_CHAT_LEFT",
                f"Бот был добавлен в посторонний чат {chat.id} ({chat.title}) и покинул его."
            )
            return

        # 4. Check stale update
        msg_age = time.time() - message.date.timestamp()
        is_stale = msg_age > self.bot_cfg.STALE_UPDATE_S

        # 5. Extract sender kind, media kind, stubs
        media_ext = extract_media_info(message)
        media_kind = media_ext.media_kind
        file_unique_id = media_ext.file_unique_id
        file_id = media_ext.file_id
        file_size = media_ext.file_size
        duration = media_ext.duration

        if media_ext.text_stub:
            text = text or media_ext.text_stub

        # Check media caps BEFORE downloading
        media_allowed = True
        if media_kind and file_size > 0:
            media_allowed, cap_reason = check_media_caps(
                media_kind, file_size, duration, self.config.limits
            )
            if not media_allowed:
                logger.info("Media rejected before download: %s", cap_reason)

        # 6. Extract URLs
        urls = extract_clean_urls(text)
        urls_json = json.dumps(urls) if urls else None

        user_id = user.id if user else None
        nick = sanitize_nick(user.full_name if user else "Anon", max_len=32)
        alias = f"u{user_id}" if user_id else "u0"

        # Reply info
        reply_to_mid = message.reply_to_message.message_id if message.reply_to_message else None
        reply_to_uid = (
            message.reply_to_message.from_user.id
            if message.reply_to_message and message.reply_to_message.from_user
            else None
        )

        msg_dict = {
            "chat_id": chat.id,
            "message_id": message.message_id,
            "user_id": user_id,
            "alias": alias,
            "nick": nick,
            "text": text,
            "urls": urls_json,
            "media_kind": media_kind,
            "file_unique_id": file_unique_id,
            "reply_to": reply_to_mid,
            "reply_to_user_id": reply_to_uid,
            "created_at": to_iso_date(message.date),
        }

        # Check /logs command early: must never be saved to database or pollute context
        clean_text = (text or "").strip()
        if clean_text.startswith("/logs") or clean_text.startswith("/logs@"):
            await self._handle_logs_command(message, msg_dict)
            return

        # Save to database
        await self.db.save_message(msg_dict)
        logger.info(
            "📩 [msg #%d] %s (%s): %s",
            message.message_id, nick, alias, text[:80] if text else f"[{media_kind or 'text'}]"
        )

        # If update is stale, do not generate responses
        if is_stale:
            return

        # 7. Check bot commands
        cmd_handled = await self._handle_commands(message, msg_dict)
        if cmd_handled:
            return

        # 8. Check Ping
        assert self.ping_detector is not None
        is_ping = self.ping_detector.is_ping(msg_dict)

        if is_ping and user_id:
            # Check author spam limit
            allowed, excuse = self.author_limiter.check_author(user_id)
            if not allowed:
                if excuse:
                    await self.send_queue.enqueue(
                        chat_id=chat.id,
                        text=excuse,
                        reply_to_message_id=message.message_id,
                        priority="P0"
                    )
                return

            # Роутинг по типу вложения (с учётом реплая), ingress не ждёт.
            # Быстрый sync-пик: если реплай на медиа — это media/track лейн, иначе chat.
            # Сессия моста — по конкретному воркеру (track1/track2 не делят контекст).
            route_kind = media_kind
            if not route_kind and message.reply_to_message:
                try:
                    route_kind = extract_media_info(message.reply_to_message).media_kind
                except Exception:
                    route_kind = None
            lane = route_lane(route_kind)
            worker = self.lanes.get(lane)
            session_tag = worker.name
            job = LaneJob(
                lane=lane,
                label=f"ping#{message.message_id}:{session_tag}",
                coro_factory=lambda m=message, md=dict(msg_dict), mk=media_kind, fi=file_id, fu=file_unique_id, ln=lane, st=session_tag: self._process_ping(m, md, mk, fi, fu, lane=ln, session_tag=st),
                enqueued_at=time.time(),
            )
            await worker.submit(job)
            self.ping_detector.record_bot_reply(user_id)
            self.proactive.on_ping_reply_sent()
        else:
            self.ping_detector.record_other_message()
            # 9. Check Proactive if not muted — тоже в chat-воркер, не в ingress
            now = time.time()
            if now > self.muted_until:
                should_eval = self.proactive.record_incoming_message(user_id)
                if should_eval:
                    can_trigger, reject_reason = self.proactive.check_proactive_caps()
                    if can_trigger:
                        job = LaneJob(
                            lane=LANE_CHAT,
                            label=f"proactive#{message.message_id}",
                            coro_factory=lambda m=message, md=dict(msg_dict): self._process_proactive(m, md),
                            enqueued_at=time.time(),
                        )
                        await self.lanes.get(LANE_CHAT).submit(job)

        # 10. Досье — в sys-воркер фоном, горячий путь не ждёт
        # (иначе каждый пинг плодит задачи и душит loop)
        try:
            await self.lanes.get(LANE_SYS).submit(LaneJob(
                lane=LANE_SYS,
                label=f"dossier#{message.message_id}",
                coro_factory=lambda chat_id=chat.id, mid=message.message_id: self._dossier_bg(chat_id, mid),
                enqueued_at=time.time(),
            ))
        except Exception as e:
            logger.warning("Dossier submit failed: %s", e)

        # 11. Живой банк стиля — сбор фразы сразу (дешёвые файловые операции),
        # промоушн батча — в sys-воркер (там сетевой вызов фильтра).
        # Чужих ботов не учим (иначе потащим чужие шаблоны), команды и медиа-стабы режет префильтр.
        _from_bot = False
        try:
            _from_bot = bool(getattr(message.from_user, "is_bot", False))
        except Exception:
            pass
        if user_id and not _from_bot and text and not media_kind:
            async def _style_bg(
                chat_id: int = chat.id,
                uid: int = user_id,
                t: str = text,
                reply_to: Optional[int] = msg_dict.get("reply_to"),
            ):
                try:
                    ctx = None
                    if reply_to:
                        try:
                            target = await self.db.get_message(chat_id, reply_to)
                            if target and (target.get("text") or "").strip():
                                ctx = target["text"][:200]
                        except Exception:
                            ctx = None
                    if await self.style_learner.collect_from_message(chat_id, uid, t, ctx=ctx):
                        await self.lanes.get(LANE_SYS).submit(LaneJob(
                            lane=LANE_SYS,
                            label="style-promote",
                            coro_factory=lambda: self._style_promote_bg(),
                            enqueued_at=time.time(),
                        ))
                except Exception as e:
                    logger.warning("Background style learn failed: %s", e)

            asyncio.create_task(_style_bg())

    async def _dossier_bg(self, chat_id: int, mid: int) -> None:
        try:
            await self.dossier_worker.process_window(chat_id, mid)
        except Exception as e:
            logger.warning("Background dossier failed: %s", e)

    async def _style_promote_bg(self) -> None:
        try:
            if await self.style_learner.maybe_promote():
                await self.style_learner.ensure_capacity()
        except Exception as e:
            logger.warning("Background style promote failed: %s", e)

    async def _is_admin(self, chat_id: int, user_id: Optional[int]) -> bool:
        """True if user is in OWNER_IDS or is a Telegram chat administrator/creator."""
        if not user_id:
            return False
        if user_id in self.bot_cfg.OWNER_IDS:
            return True
        try:
            member = await self.bot.get_chat_member(chat_id, user_id)
            return member.status in ("creator", "administrator")
        except Exception:
            return False

    async def _handle_commands(self, message: types.Message, msg_dict: dict[str, Any]) -> bool:
        text = (msg_dict.get("text") or "").strip()
        user_id = msg_dict.get("user_id")

        if text.startswith("/alive"):
            uptime_min = (time.time() - self.start_time) / 60.0
            key_summary = self.gateway.get_key_status_summary()
            shadow_tag = " [SHADOW]" if self.bot_cfg.SHADOW_MODE else " [LIVE]"
            bridge_status = "✅ OK" if key_summary.get("bridge_status") == "ok" else "❌ DOWN"
            bridge_url = key_summary.get("bridge_url", self.bot_cfg.BRIDGE_URL)
            status_text = (
                f"🤖 <b>Дик</b>{shadow_tag}\n"
                f"Аптайм: {uptime_min:.1f} мин\n"
                f"Модель: <code>{self.bot_cfg.MODEL_ID}</code>\n"
                f"Мост Antigravity: {bridge_status} (<code>{bridge_url}</code>)\n"
                f"Запросов к мосту: {key_summary.get('total_rpd_used', 0)}"
            )
            await message.reply(status_text, parse_mode="HTML")
            return True

        elif text.startswith("/reset"):
            if not await self._is_admin(message.chat.id, user_id):
                await message.reply("Команда доступна только админам.")
                return True
            ok_any = False
            for lane in (LANE_CHAT, f"{LANE_TRACK}1", f"{LANE_TRACK}2", LANE_MEDIA):
                try:
                    if await self.gateway.reset_session(self._lane_session(message.chat.id, lane)):
                        ok_any = True
                except Exception:
                    pass
            # Совместимость со старыми сессиями без суффикса лейна
            for legacy in (LANE_TRACK, f"tg_{message.chat.id}"):
                try:
                    if await self.gateway.reset_session(
                        legacy if legacy.startswith("tg_") else self._lane_session(message.chat.id, legacy)
                    ):
                        ok_any = True
                except Exception:
                    pass
            if ok_any:
                await message.reply("Сессии контекста диалога на мосту Antigravity сброшены (chat/track1/track2/media).")
            else:
                await message.reply("Сессии не найдены или уже были очищены.")
            return True

        elif text.startswith("/me"):
            alias = msg_dict.get("alias")
            await message.reply(f"Твой псевдоним: <code>{alias}</code>\nID: <code>{user_id}</code>")
            return True

        elif text.startswith("/who"):
            await self._process_who(message, msg_dict, text)
            return True

        elif text.startswith("/forget me") and user_id:
            await self.db.set_opt_out(message.chat.id, user_id)
            await self.db.delete_dossier(message.chat.id, user_id)
            await message.reply("Досье удалено, режим opt-out включён. Я тебя забыл.")
            return True

        elif text.startswith("/remember me") and user_id:
            await self.db.remove_opt_out(message.chat.id, user_id)
            await message.reply("Режим opt-out снят. Досье снова ведётся.")
            return True

        elif text.startswith("/mute"):
            if user_id not in self.bot_cfg.OWNER_IDS:
                return False
            # e.g. /mute 10м or /mute 1h
            match = re.search(r"/mute\s+(\d+)\s*([мhhdд]?)", text)
            duration_s = 600  # Default 10 min
            if match:
                val = int(match.group(1))
                unit = match.group(2).lower()
                if unit in ("h", "ч"):
                    duration_s = val * 3600
                elif unit in ("d", "д"):
                    duration_s = val * 86400
                else:
                    duration_s = val * 60

            self.muted_until = time.time() + duration_s
            mins = duration_s // 60
            await message.reply(f"🔇 Проактив заглушен на {mins} мин.")
            return True

        elif text.startswith("/refresh"):
            if user_id not in self.bot_cfg.OWNER_IDS:
                return False
            await self.dossier_worker.process_window(message.chat.id, message.message_id)
            await message.reply("🔄 Цикл досье и саммари запущен вручную.")
            return True

        elif text.startswith("/logs"):
            await self._handle_logs_command(message, msg_dict)
            return True

        return False

    async def _handle_logs_command(self, message: types.Message, msg_dict: dict[str, Any]) -> None:
        """Outputs last n log lines (default 15, max 50). Admin-only, excluded from context."""
        user_id = msg_dict.get("user_id")
        chat_id = message.chat.id
        if not await self._is_admin(chat_id, user_id):
            await message.reply("Команда доступна только админам.")
            return

        text = (msg_dict.get("text") or "").strip()
        match = re.search(r"^/logs(?:@\w+)?(?:\s+(\d+))?", text)
        n = 15
        if match and match.group(1):
            try:
                n = int(match.group(1))
            except ValueError:
                n = 15
        n = max(1, min(50, n))

        log_path = Path("logs/bot.log")
        if not log_path.exists():
            await message.reply("Лог-файл пуст или ещё не создан.")
            return

        try:
            from collections import deque
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [line.rstrip("\r\n") for line in deque(f, maxlen=n)]
        except Exception as e:
            logger.error("Failed to read log file %s: %s", log_path, e)
            await message.reply(f"Ошибка при чтении логов: {e}")
            return

        if not lines:
            await message.reply("Логи пусты.")
            return

        import html
        escaped_lines = [html.escape(l) for l in lines]
        full_text = "\n".join(escaped_lines)

        if len(full_text) <= 3800:
            try:
                await message.reply(f"<pre><code class=\"language-log\">{full_text}</code></pre>", parse_mode="HTML")
            except Exception:
                await message.reply("\n".join(lines)[:4000])
        else:
            chunks: list[str] = []
            cur_chunk: list[str] = []
            cur_len = 0
            for l in escaped_lines:
                if cur_len + len(l) + 1 > 3800:
                    if cur_chunk:
                        chunks.append("\n".join(cur_chunk))
                        cur_chunk = []
                        cur_len = 0
                cur_chunk.append(l)
                cur_len += len(l) + 1
            if cur_chunk:
                chunks.append("\n".join(cur_chunk))

            for chunk in chunks:
                try:
                    await message.reply(f"<pre><code class=\"language-log\">{chunk}</code></pre>", parse_mode="HTML")
                except Exception:
                    await message.reply(chunk[:4000])

    async def _process_who(
        self, message: types.Message, msg_dict: dict[str, Any], text: str
    ) -> None:
        """R12/TB-05: /who — роаст/сводка по досье (mode_who, P0)."""
        import random

        chat_id = message.chat.id
        target_arg = text[len("/who"):].strip().lstrip("@").lower()
        target_uid: Optional[int] = None
        target_alias = "all"

        # Цель: реплай > алиас uN > ник из last60 > все
        if message.reply_to_message and message.reply_to_message.from_user:
            target_uid = message.reply_to_message.from_user.id
            target_alias = f"u{target_uid}"
        elif target_arg:
            last_msgs = await self.db.get_last_messages(chat_id, limit=60)
            for m in last_msgs:
                if (m.get("alias") or "").lower() == target_arg or (
                    m.get("nick") or ""
                ).lower().startswith(target_arg):
                    target_uid = m.get("user_id")
                    target_alias = m.get("alias") or f"u{target_uid}"
                    break

        # opt_out цели — без API, статика
        if target_uid and await self.db.is_opt_out(chat_id, target_uid):
            excuse = (
                random.choice(self.config.persona.opt_out_replies)
                if self.config.persona.opt_out_replies
                else "Про него ничего нет."
            )
            await self.send_queue.enqueue(chat_id, excuse, message.message_id, priority="P0")
            return

        who_msg = dict(msg_dict)
        if target_uid:
            who_msg["reply_to_user_id"] = target_uid
        system_instruction, contents = await self.assembler.assemble(
            chat_id=chat_id,
            current_msg=who_msg,
            mode="who",
            request_attrs=f'target="{target_alias}"',
        )
        session_id = self._lane_session(chat_id, LANE_CHAT)
        res = await self.gateway.generate(
            prompt_name="persona_who",
            contents=contents,
            system_instruction=system_instruction,
            priority="P0",
            response_schema=PersonaTextReply,
            estimated_tokens=400,
            session_id=session_id,
            lane=LANE_CHAT,
        )
        candidate = res.reply.strip() if isinstance(res, PersonaTextReply) else (
            res.strip() if isinstance(res, str) and res.strip() else ""
        )
        is_valid, _ = validate_persona_reply(candidate, is_media=False)
        if not candidate or not is_valid:
            # Одна repair-попытка тем же приоритетом (I11), затем молчи
            res2 = await self.gateway.generate(
                prompt_name="persona_who",
                contents=contents,
                system_instruction=system_instruction,
                priority="P0",
                response_schema=PersonaTextReply,
                estimated_tokens=400,
                session_id=session_id,
                lane=LANE_CHAT,
            )
            candidate2 = res2.reply.strip() if isinstance(res2, PersonaTextReply) else (
                res2.strip() if isinstance(res2, str) and res2.strip() else ""
            )
            ok2, _ = validate_persona_reply(candidate2, is_media=False)
            if not candidate2 or not ok2:
                logger.warning("Outgoing guard: /who dropped after repair, not sending raw.")
                return
            candidate = candidate2
        final_text = await self._resolve_aliases_to_text(chat_id, candidate)
        await self.send_queue.enqueue(
            chat_id=chat_id, text=final_text,
            reply_to_message_id=message.message_id, priority="P0",
        )

    async def _auto_process_background_media(
        self,
        chat_id: int,
        media_kind: str,
        file_id: str,
        file_unique_id: str,
        caption: Optional[str] = None
    ) -> None:
        """Background P2 auto-analysis of media files per R7 (MEDIA_AUTO_ANALYZE_P)."""
        import random
        # Check cache first
        cached = await self.db.get_media_cache(file_unique_id)
        if cached:
            return

        prob_map = getattr(self.bot_cfg, "MEDIA_AUTO_ANALYZE_P", {}) or {}
        if hasattr(prob_map, "model_dump"):
            prob_map = prob_map.model_dump()
        p = prob_map.get(media_kind, 0.0) if isinstance(prob_map, dict) else getattr(prob_map, media_kind, 0.0)
        if random.random() > p:
            return

        download_path = Path(f"/tmp/smesh-{file_unique_id}")
        try:
            tg_file = await self.bot.get_file(file_id)
            if tg_file.file_path:
                await self.bot.download_file(tg_file.file_path, destination=download_path)
                await self.media_analyzer.analyze_media_step1(
                    file_path=download_path,
                    kind=media_kind,
                    file_unique_id=file_unique_id,
                    caption=caption,
                    priority="P2",
                    lane=("track" if media_kind == "audio" else "media"),
                )
                logger.info("Background P2 analyzed and cached media %s (%s)", file_unique_id, media_kind)
        except Exception as e:
            logger.warning("Background media analysis failed for %s: %s", file_unique_id, e)
        finally:
            download_path.unlink(missing_ok=True)

    async def _process_ping(
        self,
        message: types.Message,
        msg_dict: dict[str, Any],
        media_kind: Optional[str],
        file_id: Optional[str],
        file_unique_id: Optional[str],
        lane: str = LANE_CHAT,
        session_tag: Optional[str] = None,
    ) -> None:
        """Воркер-обработчик пинга. Живой флоу: typing всё время + контекст, а не дефолт-оценка."""
        chat_id = message.chat.id
        media_cache_item = None
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(self._typing_keepalive(chat_id, stop_typing))
        try:
            return await self._process_ping_inner(
                message, msg_dict, media_kind, file_id, file_unique_id, lane, session_tag
            )
        finally:
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

    async def _process_ping_inner(
        self,
        message: types.Message,
        msg_dict: dict[str, Any],
        media_kind: Optional[str],
        file_id: Optional[str],
        file_unique_id: Optional[str],
        lane: str = LANE_CHAT,
        session_tag: Optional[str] = None,
    ) -> None:
        chat_id = message.chat.id
        media_cache_item = None

        # 1. Resolve media: check current message, and if absent, check reply_to_message!
        target_kind = media_kind
        target_file_id = file_id
        target_file_unique_id = file_unique_id
        target_caption = message.caption

        if not target_file_unique_id and message.reply_to_message:
            reply_media = extract_media_info(message.reply_to_message)
            if reply_media.file_unique_id:
                target_kind = reply_media.media_kind
                target_file_id = reply_media.file_id
                target_file_unique_id = reply_media.file_unique_id
                target_caption = reply_media.caption

        # 1b. Чей реплай: снапшот цели из апдейта (покрывает via_bot и старое).
        # Стабильные alias/ник доберём из БД в gather, текст — из апдейта как fallback.
        reply_snapshot: Optional[dict[str, Any]] = None
        reply_to_id = msg_dict.get("reply_to")
        if reply_to_id and message.reply_to_message:
            rmsg = message.reply_to_message
            ruser = getattr(rmsg, "from_user", None)
            ruid = getattr(ruser, "id", None) if ruser else None
            if ruid is None:
                ruid = msg_dict.get("reply_to_user_id")
            rnick = sanitize_nick(
                getattr(ruser, "full_name", None) or getattr(getattr(rmsg, "sender_chat", None), "title", None) or "Unknown",
                max_len=32,
            )
            rtext = (getattr(rmsg, "text", None) or getattr(rmsg, "caption", None) or "")
            if not rtext:
                try:
                    rtext = extract_media_info(rmsg).text_stub or ""
                except Exception:
                    rtext = ""
            reply_snapshot = {
                "message_id": reply_to_id,
                "user_id": ruid,
                "alias": f"u{ruid}" if ruid is not None else "u0",
                "nick": rnick,
                "text": rtext,
            }
        # Уточняем лейн по реальной цели (пинг мог быть текстом, а цель — трек/мем)
        if target_kind == "audio":
            lane = LANE_TRACK
        elif target_kind:
            lane = LANE_MEDIA if lane == LANE_CHAT and target_kind != "audio" else lane
        # Сессия — по конкретному воркеру (track1/track2 не делят контекст).
        # Если лейн уточнился уже внутри воркера — fallback на общую сессию лейна.
        tag = session_tag or lane
        if lane == LANE_TRACK and not tag.startswith(LANE_TRACK):
            tag = LANE_TRACK
        elif lane != LANE_TRACK and tag != lane and tag not in (LANE_CHAT, LANE_MEDIA):
            tag = lane
        session_id = self._lane_session(chat_id, tag)
        media_lane = "track" if (target_kind == "audio") else ("media" if target_kind else lane)

        # 2. Медиа и история — параллельно: пока качается файл, уже читаем контекст
        text = (msg_dict.get("text") or "").strip()
        show_links = self.research.should_show_links(text)
        # Провокация в адрес бота: реплай ему + оскорбление/подначивание/разводка.
        # Обычные подъёбы между участниками — не провокация, там дефолт-режим.
        t_low = text.lower()
        _is_reply_to_bot = bool(msg_dict.get("reply_to_user_id") == self.bot_id and self.bot_id)
        _mentions = (f"@{self.bot_username}".lower() in t_low) if getattr(self, "bot_username", "") else False
        if not _mentions:
            for _a in (self.bot_cfg.BOT_ALIASES or []):
                # по границам слова: «дик» внутри «дикий» — не упоминание
                if _a and re.search(r"(?<!\w)" + re.escape(_a.lower()) + r"(?!\w)", t_low):
                    _mentions = True
                    break
        provoked = ProvocationDetector.is_provoked(
            text, is_reply_to_bot=_is_reply_to_bot, mentions_bot=_mentions
        )
        want_decide = (
            self.research.should_force_search(text)
            or self.research.should_soft_search(text)
            or bool(target_file_unique_id)
        )

        async def _load_media():
            if not target_file_unique_id:
                return None
            cached = await self.db.get_media_cache(target_file_unique_id)
            if cached:
                return cached
            if not (target_file_id and target_kind):
                return None
            download_path = Path(f"/tmp/smesh-{target_file_unique_id}")
            try:
                tg_file = await self.bot.get_file(target_file_id)
                if tg_file.file_path:
                    await self.bot.download_file(tg_file.file_path, destination=download_path)
                    return await self.media_analyzer.analyze_media_step1(
                        file_path=download_path,
                        kind=target_kind,
                        file_unique_id=target_file_unique_id,
                        caption=target_caption,
                        priority="P0",
                        lane=media_lane,
                    )
            except Exception as e:
                logger.error("Error downloading/analyzing media for ping: %s", e)
                return None
            finally:
                try:
                    download_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return None

        async def _load_hint():
            # Хинт грузим всегда: он нужен и ресёрчу, и детектору настроения.
            # Локальный SQLite — дёшево, на латентность не влияет.
            try:
                return await self.db.get_last_messages(chat_id, limit=10)
            except Exception:
                return []

        async def _load_reply_row():
            if not reply_to_id:
                return None
            try:
                return await self.db.get_message(chat_id, reply_to_id)
            except Exception:
                return None

        media_cache_item, hint_msgs, reply_row = await asyncio.gather(
            _load_media(), _load_hint(), _load_reply_row()
        )

        # 2b. Финальный снапшот цели: стабильные alias/ник из БД, текст из БД/апдейта.
        # Случай из чата: трек скинул другой чел, а ты реплаем попросил оценку —
        # модель должна видеть, что трек НЕ твой.
        if reply_to_id:
            tuid = None
            talias = None
            tnick = None
            ttext = ""
            if reply_row:
                tuid = reply_row.get("user_id")
                talias = reply_row.get("alias") or (f"u{tuid}" if tuid is not None else None)
                tnick = reply_row.get("nick") or None
                ttext = reply_row.get("text") or ""
            if reply_snapshot:
                tuid = tuid if tuid is not None else reply_snapshot.get("user_id")
                talias = talias or reply_snapshot.get("alias")
                tnick = tnick or reply_snapshot.get("nick")
                if not ttext:
                    ttext = reply_snapshot.get("text") or ""
            if tuid is None:
                tuid = msg_dict.get("reply_to_user_id")
            reply_snapshot = {
                "message_id": reply_to_id,
                "user_id": tuid,
                "alias": talias or (f"u{tuid}" if tuid is not None else "u0"),
                "nick": sanitize_nick(tnick or "Unknown", max_len=32),
                "text": ttext or "",
            }

        # 3. Research: на любое медиа — ВСЕГДА (детерминированный запрос
        # из разбора, без LLM-гейта: модель подцепит факты через search_web).
        # На текст — как раньше: быстрый гейт в коде + LLM формулирует query.
        need_search = False
        safe_query = None
        sources_text = None

        media_ctx = None
        if media_cache_item:
            m_sum = media_cache_item.get("summary") or ""
            m_ocr = media_cache_item.get("ocr_text") or ""
            m_cap = target_caption or ""
            parts = []
            if m_sum:
                parts.append(f"Медиа: {m_sum[:250]}")
            if m_ocr:
                parts.append(f"Текст на картинке: {m_ocr[:250]}")
            if m_cap:
                parts.append(f"Подпись: {m_cap[:150]}")
            if parts:
                media_ctx = f"[{' | '.join(parts)}]"

        if media_cache_item:
            history_lines = [f"{m.get('nick', 'User')}: {m.get('text', '')}" for m in hint_msgs if m.get("text")]
            known_nicks = [m.get("nick") for m in hint_msgs if m.get("nick")]
            safe_query = build_media_search_query(
                media_cache_item,
                caption=target_caption,
                user_text=text,
                known_nicks=known_nicks,
                bot_aliases=self.bot_cfg.BOT_ALIASES,
            )
            if safe_query:
                need_search = True
            else:
                # из разбора совсем нечего взять — пусть LLM попробует
                history_str = "\n".join(history_lines)
                need_search, safe_query = await self.research.decide_search(
                    history_text=history_str,
                    current_text=text,
                    known_nicks=known_nicks,
                    extra_context=media_ctx,
                )
                if not need_search:
                    # последний шанс: поиск по подписи/тексту как есть
                    fallback = build_media_search_query(
                        {"kind": target_kind, "summary": text or target_caption or ""},
                        caption=None, user_text=None,
                        known_nicks=known_nicks, bot_aliases=self.bot_cfg.BOT_ALIASES,
                    )
                    if fallback:
                        need_search, safe_query = True, fallback
        elif want_decide:
            history_lines = [f"{m.get('nick', 'User')}: {m.get('text', '')}" for m in hint_msgs if m.get("text")]
            history_str = "\n".join(history_lines)
            known_nicks = [m.get("nick") for m in hint_msgs if m.get("nick")]

            need_search, safe_query = await self.research.decide_search(
                history_text=history_str,
                current_text=text,
                known_nicks=known_nicks,
                extra_context=media_ctx,
            )

        # Настроение чата: serious — отвечают по делу, banter — угар.
        # Дешёвая эвристика по тексту пинга и последним сообщениям.
        mood = detect_mood(
            text,
            [m.get("text", "") for m in hint_msgs if m.get("text")],
        )

        # Do NOT inject fake sources_text. The model autonomously runs search_web with search_query.

        # 4. Mode: медиа всегда через media_reply (там контекст + запрет N/100),
        # поиск — флагом enable_search, а не отдельным режимом без разбора вложения
        if media_cache_item:
            mode = "media_reply"
        elif need_search:
            mode = "research"
        else:
            mode = "ping"

        # 5. Assemble context — полный (history+досье+тред), а не только разбор.
        # Ответ от контекста: кто кинул, что обсуждали, какое досье у автора.
        # Вложение принадлежит sent_by, просит ответить requested_by — не путать!
        step2_media_json = None
        media_kind_for_guard = None
        requester_alias = msg_dict.get("alias") or f"u{msg_dict.get('user_id', 0)}"
        requester_nick = msg_dict.get("nick") or "User"
        if media_cache_item:
            media_kind_for_guard = media_cache_item.get("kind")
            # Вложение из реплая — автор цели; своё вложение — автор запроса.
            media_from_reply = bool(reply_snapshot) and not media_kind
            if media_from_reply and reply_snapshot:
                sent_alias, sent_nick = reply_snapshot.get("alias"), reply_snapshot.get("nick")
            else:
                sent_alias, sent_nick = requester_alias, requester_nick
            step2_media_json = build_step2_media_json(
                media_cache_item,
                sent_by_alias=sent_alias,
                sent_by_nick=sent_nick,
                requested_by_alias=requester_alias,
                requested_by_nick=requester_nick,
            )

        # Обогащённый current: явная цель реплая для <current> (см. assembler).
        current_enriched = dict(msg_dict)
        if reply_snapshot:
            current_enriched["reply_target"] = reply_snapshot

        request_attrs = f'mode="{mode}"'
        if mode == "research":
            request_attrs += f' show_links="{"true" if show_links else "false"}"'
        if reply_snapshot and reply_snapshot.get("alias"):
            request_attrs += f' reply_target="{reply_snapshot.get("alias")}"'
        # Явный флаг для промпта: провокация в адрес бота → режим жёсткого отпора
        request_attrs += f' provoked="{"true" if provoked else "false"}"'
        # Настроение чата: serious — по делу без роаста, banter — угар
        request_attrs += f' mood="{mood}"'

        system_instruction, contents = await self.assembler.assemble(
            chat_id=chat_id,
            current_msg=current_enriched,
            mode=mode,
            media_json=step2_media_json,
            sources_text=sources_text,
            request_attrs=request_attrs,
        )

        reply_raw = None
        is_media_mode = (mode == "media_reply")
        # При провокации второй провал гарда не должен давать вялую отмазку —
        # лучше первый валидный (пусть длинноватый) кандидат, чем «башка не варит»
        provoked_fallback: Optional[str] = None
        # Анти-гиперфикс: свои последние реплики — кандидат не должен их повторять
        recent_bot_texts: list[str] = []
        try:
            last_all = await self.db.get_last_messages(chat_id, limit=30)
            bot_uid = self.bot_id
            for m in reversed(last_all):
                if m.get("is_bot") or (bot_uid and m.get("user_id") == bot_uid):
                    t = (m.get("text") or "").strip()
                    if t:
                        recent_bot_texts.append(t)
                    if len(recent_bot_texts) >= 3:
                        break
        except Exception:
            recent_bot_texts = []

        # Outgoing guard: leak/bad parse/forbidden N/100 на фото → 1 repair, сырое не шлём
        for attempt in range(2):
            res = await self.gateway.generate(
                prompt_name=f"persona_{mode}",
                contents=contents,
                system_instruction=system_instruction,
                priority="P0",
                response_schema=PersonaTextReply,
                estimated_tokens=700,
                enable_search=bool(need_search),
                search_query=safe_query if need_search else None,
                session_id=session_id,
                lane=lane,
            )

            if isinstance(res, dict) and (res.get("is_degraded") or res.get("is_blocked")):
                reply_raw = res
                break

            if isinstance(res, PersonaTextReply):
                candidate_text = res.reply.strip()
            elif isinstance(res, str) and res.strip():
                candidate_text = res.strip()
            else:
                logger.warning("Outgoing guard: failed to parse JSON {reply} on attempt %d/2", attempt + 1)
                if attempt == 0:
                    continue  # 1 repair retry
                else:
                    reply_raw = None
                    break

            # Validate reply per Point 2
            is_valid, reject_reason = validate_persona_reply(candidate_text, is_media=is_media_mode)
            if not is_valid:
                logger.warning(
                    "Outgoing guard: invalid reply on attempt %d/2: %s (text: %.80s)",
                    attempt + 1, reject_reason, candidate_text
                )
                if attempt == 0:
                    continue  # 1 repair retry
                else:
                    reply_raw = None
                    break
            # Запрет N/100 на любом не-аудио — тоже повод на repair, а не молча резать
            if is_media_mode and contains_forbidden_score(candidate_text, media_kind_for_guard):
                logger.warning(
                    "Outgoing guard: forbidden N/100 score on %s reply, repair attempt %d/2",
                    media_kind_for_guard, attempt + 1,
                )
                if attempt == 0:
                    continue
                candidate_text = SCORE_PATTERN.sub("", candidate_text).strip()
            # Провокация: отпор обязан быть гиперкратким — длинноту/вопросы в repair
            if provoked and not is_media_mode and not ProvocationDetector.provoked_reply_ok(candidate_text):
                logger.warning(
                    "Outgoing guard: provoked reply too soft/long on attempt %d/2 (text: %.80s)",
                    attempt + 1, candidate_text,
                )
                if provoked_fallback is None:
                    provoked_fallback = candidate_text
                if attempt == 0:
                    continue
                candidate_text = provoked_fallback
            # Анти-склейка: короткий ответ из двух мыслей через запятую — в repair.
            # Длинные ответы по запросу (объяснения) не трогаем.
            if not provoked and not is_media_mode and has_comma_glue(candidate_text):
                logger.warning(
                    "Outgoing guard: comma-glued reply on attempt %d/2 (text: %.80s)",
                    attempt + 1, candidate_text,
                )
                if attempt == 0:
                    continue
            # Анти-гиперфикс: повтор своего же недавнего панча — в repair
            if not provoked and is_repetitive(candidate_text, recent_bot_texts):
                logger.warning(
                    "Outgoing guard: repetitive reply on attempt %d/2 (text: %.80s)",
                    attempt + 1, candidate_text,
                )
                if attempt == 0:
                    continue
            reply_raw = candidate_text
            break

        if not reply_raw and provoked_fallback:
            reply_raw = provoked_fallback

        if not reply_raw:
            # Fallback to static degraded excuse instead of sending raw/broken text
            import random
            excuse = (
                random.choice(self.config.persona.degraded_replies)
                if self.config.persona.degraded_replies
                else "Башка не варит, позже."
            )
            await self.send_queue.enqueue(chat_id, excuse, message.message_id, priority="P0")
            return

        if isinstance(reply_raw, dict) and reply_raw.get("is_degraded"):
            # Instant degraded excuse
            import random
            excuse = (
                random.choice(self.config.persona.degraded_replies)
                if self.config.persona.degraded_replies
                else "Башка не варит, позже."
            )
            await self.send_queue.enqueue(chat_id, excuse, message.message_id, priority="P0")
            return

        if isinstance(reply_raw, dict) and reply_raw.get("is_blocked"):
            # Filter block comeback
            reply_raw = "Не, эту хуйню я озвучивать не буду."

        # If media reply, strip any boilerplate and verify/patch N/100 score ONLY for audio
        if is_media_mode and isinstance(reply_raw, str):
            reply_raw = clean_media_reply_boilerplate(reply_raw)
            if media_cache_item and media_cache_item.get("kind") == "audio":
                reply_raw = self.media_analyzer.verify_and_patch_reply_score(reply_raw, media_cache_item)
            else:
                reply_raw = SCORE_PATTERN.sub("", reply_raw).strip()

        final_text = await self._resolve_aliases_to_text(chat_id, str(reply_raw))
        await self.send_queue.enqueue(
            chat_id=chat_id,
            text=final_text,
            reply_to_message_id=message.message_id,
            priority="P0"
        )

    async def _resolve_aliases_to_text(self, chat_id: int, text: str) -> str:
        last_msgs = await self.db.get_last_messages(chat_id, limit=60)
        user_mapping = {
            m.get("alias") or f"u{m.get('user_id')}": {
                "user_id": m.get("user_id"),
                "nick": m.get("nick"),
                "username": None
            }
            for m in last_msgs
        }
        return replace_un_aliases(text, user_mapping)

    async def _process_proactive(self, message: types.Message, msg_dict: dict[str, Any]) -> None:
        chat_id = message.chat.id
        from src.gateway.schemas import ProactiveDecision

        force_val = self.proactive.should_force_proactive()
        mood = detect_mood(msg_dict.get("text") or "")
        system_instruction, contents = await self.assembler.assemble(
            chat_id=chat_id,
            current_msg=msg_dict,
            mode="proactive",
            request_attrs=f'force="{"true" if force_val else "false"}" mood="{mood}"'
        )

        decision: Optional[ProactiveDecision] = await self.gateway.generate(
            prompt_name="proactive",
            contents=contents,
            system_instruction=system_instruction,
            priority="P1",
            response_schema=ProactiveDecision,
            estimated_tokens=300,
            session_id=self._lane_session(chat_id, LANE_CHAT),
            lane=LANE_CHAT,
        )

        if not decision or not isinstance(decision, ProactiveDecision):
            return

        if decision.action.value == "reply" and decision.reply:
            # Check outgoing leak guard on proactive reply as well
            if has_persona_leak(decision.reply, is_media=False):
                logger.warning("Outgoing guard: proactive reply contained leak, dropping.")
                self.proactive.on_proactive_skipped()
                return
            # Анти-склейка через запятую — тоже пропуск, а не вялая склейка
            if has_comma_glue(decision.reply):
                logger.warning("Outgoing guard: proactive reply comma-glued, dropping.")
                self.proactive.on_proactive_skipped()
                return

            reply_to = decision.target_msg_id or None
            final_reply = await self._resolve_aliases_to_text(chat_id, decision.reply)
            await self.send_queue.enqueue(
                chat_id=chat_id,
                text=final_reply,
                reply_to_message_id=reply_to,
                priority="P1"
            )
            self.proactive.on_proactive_reply_sent()
        else:
            self.proactive.on_proactive_skipped()
