import json
import re
import time
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from src.ops.config import BotConfig
from src.ops.log import logger
from src.db.manager import DatabaseManager
from src.db.contracts import Gateway
from src.gateway.schemas import DossierExtract, Observation, Consolidation
from src.context.prompts import PromptBuilder
from src.context.assemble import filter_summary_imperatives


INJECTION_KEYWORDS = [
    r"забудь\w*",
    r"ты теперь\w*",
    r"игнорируй\w*",
    r"запиши в досье\w*",
    r"системн\w* промпт\w*",
    r"ты обязан\w*",
    r"смени роль\w*",
]
INJECTION_PATTERN = re.compile("|".join(INJECTION_KEYWORDS), re.IGNORECASE)


class DossierWorker:
    """Manages window-based dossier extraction, verification, and consolidation per R4."""

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
        self.dossier_every = bot_config.DOSSIER_EVERY
        self.claim_min_count = bot_config.CLAIM_MIN_COUNT
        self.tz_name = bot_config.TZ

    def _get_tz_today(self) -> str:
        tz = ZoneInfo(self.tz_name)
        return datetime.now(tz).strftime("%Y-%m-%d")

    async def process_window(self, chat_id: int, current_max_seq: int) -> bool:
        """
        Extracts observations and updates summary for the new message window.
        Returns True if window was processed successfully.
        """
        # 1. Fetch unread messages
        summary_row = await self.db.get_summary(chat_id)
        last_seq = summary_row["msg_seq"] if summary_row else 0
        if (current_max_seq - last_seq) < self.dossier_every:
            return False

        # Grab up to DOSSIER_MAX_WINDOW messages
        messages = await self.db.get_last_messages(chat_id, limit=min(current_max_seq - last_seq, self.config.DOSSIER_MAX_WINDOW))
        if not messages:
            return False

        # Filter out messages with NULL user_id or users who opted out.
        # Свои исходящие (is_bot) в досье не идут — иначе заведём досье на самого бота.
        bot_uids = {m["user_id"] for m in messages if m.get("is_bot") and m.get("user_id")}
        window_authors = {m["user_id"] for m in messages if m.get("user_id")}
        active_authors = []
        for uid in window_authors:
            if uid in bot_uids:
                continue
            if not await self.db.is_opt_out(chat_id, uid):
                active_authors.append(uid)

        # 2. Build existing dossiers string for authors
        existing_dossiers_lines = []
        author_dossiers: dict[int, dict[str, Any]] = {}
        for uid in active_authors:
            d_row = await self.db.get_dossier(chat_id, uid)
            if d_row:
                try:
                    data = json.loads(d_row["dossier_data"])
                    author_dossiers[uid] = data
                    alias = d_row.get("alias") or f"u{uid}"
                    for field_name, field_dict in data.items():
                        for item in field_dict.get("stable", []):
                            existing_dossiers_lines.append(
                                f"[id={item.get('id')}|{alias}|{field_name}|stable] {item.get('text')}"
                            )
                        for item in field_dict.get("recent", []):
                            existing_dossiers_lines.append(
                                f"[id={item.get('id')}|{alias}|{field_name}|recent] {item.get('text')}"
                            )
                except Exception:
                    pass

        existing_dossiers_str = "\n".join(existing_dossiers_lines)

        # 3. Build window lines
        window_lines = []
        msg_map: dict[int, dict[str, Any]] = {}
        for m in messages:
            mid = m["message_id"]
            msg_map[mid] = m
            alias = m.get("alias") or f"u{m.get('user_id', 0)}"
            nick = m.get("nick") or "User"
            text = m.get("text") or ""
            # If injection suspected or forward
            if m.get("fwd_kind"):
                text = f"[пересланное от {m.get('fwd_origin_name') or 'скрыто'}] {text}"
            window_lines.append(f"[{mid}|{alias}|{nick}]: {text}")

        window_str = "\n".join(window_lines)

        # 4. System instruction and contents
        system_instruction = self.prompts.build_system_instruction(
            mode_or_modular="dossier_extract",
            bot_name="Дик",
            now_str=self._get_tz_today(),
            summary_tok=self.config.SUMMARY_TOK
        )

        current_summary = summary_row["summary"] if summary_row else ""
        contents = PromptBuilder.build_contents(
            custom_blocks=[
                ("summary", current_summary),
                ("existing_dossiers", existing_dossiers_str),
                ("window", window_str),
                ("request", 'mode="full"')
            ]
        )

        # 5. Call Gateway with P2 priority (фон — sys-лейн, юзерские чат/медиа не трогаем)
        result: Optional[DossierExtract] = await self.gateway.generate(
            prompt_name="dossier_extract",
            contents=contents,
            system_instruction=system_instruction,
            priority="P2",
            response_schema=DossierExtract,
            estimated_tokens=3000,
            lane="sys",
        )

        if not result or not isinstance(result, DossierExtract):
            logger.warning("Dossier extract call returned empty/failed")
            return False

        # 6. Apply Observations with code verification
        today = self._get_tz_today()
        for obs in result.observations:
            await self._apply_single_observation(chat_id, obs, msg_map, author_dossiers, today)

        # 7. Update Chat Summary
        if result.new_summary:
            clean_sum = filter_summary_imperatives(result.new_summary)
            await self.db.save_summary(chat_id, clean_sum, msg_seq=messages[-1]["message_id"])

        logger.info(
            "Successfully processed dossier window up to msg %d (applied %d observations)",
            messages[-1]["message_id"], len(result.observations)
        )
        return True

    async def _apply_single_observation(
        self,
        chat_id: int,
        obs: Observation,
        msg_map: dict[int, dict[str, Any]],
        author_dossiers: dict[int, dict[str, Any]],
        today: str
    ) -> bool:
        # Rule 1: msg_id must exist in window
        msg = msg_map.get(obs.msg_id)
        if not msg:
            return False

        # Rule 1b: свои исходящие досье не пополняют (иначе досье на бота)
        if msg.get("is_bot"):
            return False

        # Rule 2: User must match author of msg
        user_alias = msg.get("alias") or f"u{msg.get('user_id')}"
        if obs.user != user_alias and obs.user != f"u{msg.get('user_id')}":
            return False

        user_id = msg.get("user_id")
        if not user_id or await self.db.is_opt_out(chat_id, user_id):
            return False

        # Rule 3: Forwarded messages cannot produce claimed facts
        if msg.get("fwd_kind"):
            return False

        # Rule 4: Anti-injection dictionary
        if INJECTION_PATTERN.search(obs.text) or INJECTION_PATTERN.search(obs.evidence):
            return False

        # Rule 5: Evidence must be exact substring of message text
        msg_text = msg.get("text") or ""
        if obs.evidence not in msg_text:
            return False

        # Load or initialize user dossier data
        dossier_data = author_dossiers.get(user_id)
        if not dossier_data:
            d_row = await self.db.get_dossier(chat_id, user_id)
            if d_row:
                try:
                    dossier_data = json.loads(d_row["dossier_data"])
                except Exception:
                    dossier_data = {}
            else:
                dossier_data = {}
            author_dossiers[user_id] = dossier_data

        field_data = dossier_data.setdefault(obs.field.value, {"stable": [], "recent": []})
        stable_list = field_data.setdefault("stable", [])
        recent_list = field_data.setdefault("recent", [])

        # Check match_id
        matched_item = None
        if obs.match_id:
            for item in (stable_list + recent_list):
                if item.get("id") == obs.match_id:
                    matched_item = item
                    break

        if matched_item:
            # Update existing fact
            matched_item["last_seen"] = today
            days_seen = matched_item.setdefault("days_seen", [])
            if today not in days_seen:
                days_seen.append(today)
            matched_item["count"] = matched_item.get("count", 1) + 1
            ev_list = matched_item.setdefault("evidence_msg_ids", [])
            if obs.msg_id not in ev_list:
                ev_list.append(obs.msg_id)

            # Promote to stable if len(days_seen) >= claim_min_count
            if len(days_seen) >= self.claim_min_count and matched_item in recent_list:
                recent_list.remove(matched_item)
                stable_list.append(matched_item)
        else:
            # Create new recent item
            new_id = f"d{obs.msg_id}_{int(time.time() % 1000)}"
            new_item = {
                "id": new_id,
                "text": obs.text,
                "first_seen": today,
                "last_seen": today,
                "days_seen": [today],
                "count": 1,
                "evidence_msg_ids": [obs.msg_id]
            }
            recent_list.append(new_item)

        # Save to DB
        await self.db.save_dossier(
            chat_id=chat_id,
            user_id=user_id,
            dossier_data=json.dumps(dossier_data, ensure_ascii=False)
        )
        return True
