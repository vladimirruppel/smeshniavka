import hashlib
import re
from typing import Any, Optional

from src.ops.config import BotConfig
from src.ops.log import logger
from src.db.manager import DatabaseManager
from src.db.contracts import Gateway
from src.gateway.schemas import ResearchDecision
from src.context.prompts import PromptBuilder


USER_ANON_RE = re.compile(r"(@[a-zA-Z0-9_]+|\bu\d+\b|\b\d{8,12}\b)")


def anonymize_query(
    query: str,
    known_nicks: Optional[list[str]] = None,
    bot_aliases: Optional[list[str]] = None
) -> Optional[str]:
    """
    Anonymizes query per R8:
    - strips @username, uN, numeric ids
    - strips known participant nicks
    - strips bot name and aliases
    """
    if not query:
        return None

    clean = USER_ANON_RE.sub("", query)
    if known_nicks:
        for nick in known_nicks:
            if nick and len(nick) >= 3:
                clean = re.sub(re.escape(nick), "", clean, flags=re.IGNORECASE)

    aliases = list(bot_aliases or []) + ["дик", "дика", "дику", "диком", "дике"]
    for alias in aliases:
        if alias and len(alias) >= 3:
            if len(alias) <= 4:
                # короткий алиас — только целым словом, иначе «дикий» превратится в «ий»
                clean = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", "", clean, flags=re.IGNORECASE)
            else:
                clean = re.sub(re.escape(alias), "", clean, flags=re.IGNORECASE)

    clean = re.sub(r"\s+", " ", clean).strip()
    # If query is too short after anonymization, cannot search
    if len(clean) < 3:
        return None
    return clean


def build_media_search_query(
    media_cache_item: Optional[dict[str, Any]],
    caption: Optional[str] = None,
    user_text: Optional[str] = None,
    known_nicks: Optional[list[str]] = None,
    bot_aliases: Optional[list[str]] = None,
) -> Optional[str]:
    """Детерминированный поисковый запрос из разбора медиа (без LLM).

    Всегда даёт запрос, когда есть медиа: трек — по исполнителю/названию,
    картинка/мем — по тексту на ней и сути, видео — по сути кадра.
    Возвращает уже обезличенный запрос или None, если совсем не из чего.
    """
    import json as _json

    if not media_cache_item:
        return None
    kind = media_cache_item.get("kind")

    # 1. Трек: исполнитель + название — лучший запрос
    track_meta = media_cache_item.get("track_meta")
    if isinstance(track_meta, str):
        try:
            track_meta = _json.loads(track_meta)
        except Exception:
            track_meta = None
    if kind == "audio" and isinstance(track_meta, dict):
        artist = (track_meta.get("artist") or "").strip()
        title = (track_meta.get("title") or "").strip()
        if artist or title:
            q = f"{artist} {title}".strip()
            return anonymize_query(q, known_nicks, bot_aliases) or q

    # 2. Текст на картинке/меме — цепляемся за сущности
    ocr = (media_cache_item.get("ocr_text") or "").strip()
    if ocr:
        # берём первые осмысленные слова (не мусор), максимум 8
        words = [w for w in re.split(r"\s+", ocr) if len(w) >= 3][:8]
        if words:
            q = " ".join(words)
            safe = anonymize_query(q, known_nicks, bot_aliases)
            if safe:
                return safe

    # 3. Суть разбора + подпись + текст запроса
    bits: list[str] = []
    summary = (media_cache_item.get("summary") or "").strip()
    if summary:
        bits.append(summary[:120])
    if caption and caption.strip():
        bits.append(caption.strip()[:80])
    if user_text and user_text.strip():
        # из текста юзера — только если там вопрос про мем/событие
        t_low = user_text.lower().replace("ё", "е")
        if any(k in t_low for k in ("мем", "прикол", "причем", "при чем", "что за", "че за", "поясни", "кто такой", "что такое")):
            bits.append(user_text.strip()[:80])
    if bits:
        q = " ".join(bits)
        # выкидываем слова-мусор про сам мем
        q = re.sub(r"(?i)\b(мем|мема|меме|кроссовер|картинка|фото|видос|видео|сериал)\b", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        safe = anonymize_query(q, known_nicks, bot_aliases)
        if safe:
            # запрос не длиннее 8 слов
            return " ".join(safe.split()[:8]) or None
    return None


class ResearchModule:
    """Handles query deciding, query anonymization, caching, and research responses per R8."""

    def __init__(
        self,
        bot_config: BotConfig,
        db_manager: DatabaseManager,
        gateway: Gateway,
        prompt_builder: PromptBuilder
    ):
        self.config = bot_config
        self.db = db_manager
        self.gateway = gateway
        self.prompts = prompt_builder

    def should_force_search(self, text: str) -> bool:
        t_low = text.lower().replace("ё", "е")
        for trig in self.config.SEARCH_FORCE_TRIGGERS:
            trig_norm = trig.lower().replace("ё", "е")
            if trig_norm in t_low:
                return True
        return False

    def should_soft_search(self, text: str) -> bool:
        t_low = text.lower().replace("ё", "е")
        soft_phrases = (
            "что случилось",
            "что произошло",
            "в связи с",
            "последние события",
            "последние новости",
            "какие новости",
            "правда ли",
            "что за новость",
            "что там с",
            "кто такой",
            "что такое",
            "поясни за",
            "поясни",
            "разъясни",
            "причем тут",
            "при чем тут",
            "причем здесь",
            "при чем здесь",
            "причем",
            "при чем",
            "в чем мем",
            "в чем прикол",
            "в чем суть",
            "в чем смысл",
            "что за мем",
            "че за мем",
            "че за прикол",
            "че за",
            "что за",
            "о чем речь",
            "про что",
            "че с ",
            "что с ",
            "откуда это",
            "почему тут",
            "что значит",
        )
        return any(phrase in t_low for phrase in soft_phrases)

    def should_show_links(self, text: str) -> bool:
        t_low = text.lower().replace("ё", "е")
        for trig in self.config.LINK_TRIGGERS_SHOW:
            trig_norm = trig.lower().replace("ё", "е")
            if trig_norm in t_low:
                return True
        return False

    async def decide_search(
        self,
        history_text: str,
        current_text: str,
        known_nicks: Optional[list[str]] = None,
        extra_context: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Determines whether search is needed. Returns (need_search, anonymized_query).
        """
        forced = self.should_force_search(current_text)

        system_instruction = self.prompts.build_system_instruction(
            mode_or_modular="research_decide",
            bot_name="Дик"
        )
        cur_payload = current_text
        if extra_context:
            cur_payload = f"{current_text}\n{extra_context}"

        contents = PromptBuilder.build_contents(
            history=history_text,
            current=cur_payload,
            request=f'forced="{"true" if forced else "false"}"'
        )

        dec_result: Optional[ResearchDecision] = await self.gateway.generate(
            prompt_name="research_decide",
            contents=contents,
            system_instruction=system_instruction,
            priority="P0",
            response_schema=ResearchDecision,
            estimated_tokens=200,
            lane="chat",
        )

        is_decision = isinstance(dec_result, ResearchDecision)
        need_search = (dec_result.need_search if is_decision else False) or forced
        if not need_search:
            return False, None

        raw_query = (dec_result.query if is_decision and dec_result.query else current_text)
        safe_query = anonymize_query(raw_query, known_nicks, bot_aliases=self.config.BOT_ALIASES)
        if not safe_query and forced:
            # Fallback for forced search: strip commands/mentions and use current text
            safe_query = anonymize_query(current_text, known_nicks, bot_aliases=self.config.BOT_ALIASES)

        if not safe_query:
            logger.info("Search query could not be anonymized, skipping search")
            return False, None

        return True, safe_query

    async def execute_research(
        self,
        query: str,
        chat_id: int,
        current_msg: dict[str, Any],
        system_instruction: str,
        contents: list[dict[str, Any]],
        show_links: bool = False
    ) -> Optional[str]:
        """
        Executes Google Search tool call or retrieves from research_cache.
        """
        cache_key = hashlib.sha256(f"q:{query}".encode()).hexdigest()[:16]
        cached = await self.db.get_research_cache(cache_key, ttl_hours=self.config.RESEARCH_TTL_H)
        if cached:
            logger.info("Research query '%s' served from cache", query)
            return cached["result_text"]

        # Call Gemini with Google Search tool enabled
        res = await self.gateway.generate(
            prompt_name="persona_research",
            contents=contents,
            system_instruction=system_instruction,
            priority="P0",
            enable_search=True,
            estimated_tokens=700
        )

        if isinstance(res, str) and res.strip():
            # If show_links is false, ensure no bare URLs leaked
            if not show_links:
                # Strip markdown/html links
                res = re.sub(r'<a\s+href="[^"]*">([^<]*)</a>', r"\1", res)
                res = re.sub(r'https?://[^\s<>"]+', "", res)

            await self.db.save_research_cache(
                cache_key=cache_key,
                query=query,
                url=None,
                result_text=res,
                max_items=self.config.RESEARCH_CACHE_MAX
            )
            return res

        return None
