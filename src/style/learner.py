"""Живой банк стиля: сбор кандидатов из чата + промоушн через style_filter.

Дообучения модели нет — адаптация только через few-shot примеры. Банк на
128 примеров (STYLE_EXAMPLES_MAX) — это манера речи бота: каждый запрос
тянет подвыборку примеров в промпт.

Правила потока:
- 3+ смеяки — сразу в банк, минуя очередь и фильтр (смех чата и есть вердикт).
- 1–2 смеяки — в очередь на отбор обычным путём через style_filter.
- без реакций — на усмотрение фильтра (collect_from_message + maybe_promote).
-_entry хранит контекст: на что был ответ (ctx), чтобы пример нёс ситуацию,
  а не голую фразу.
- пермач — только сиды (дефолт «бум»): при переполнении вытесняется всё
  кроме пермача. Что вытеснить из 127 остальных — решает ИИ (style_evict),
  глядя на банк и «предложку» (очередь кандидатов): может не вытеснить
  ничего, тогда банк временно пухнет (жёсткий потолок max+16 — FIFO).

Файлы в .gitignore, права 600; на RPi prompts/ должен быть на запись
(см. docker-compose.yml) и rsync не должен их затирать (см. deploy.sh).
"""
import asyncio
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from src.ops.log import logger
from src.context.assemble import scrub_pii
from src.gateway.schemas import StyleVerdicts

CANDIDATES_FILE = "style_examples.candidates.yaml"
ACTIVE_FILE = "style_examples.yaml"

# Жёсткий потолок сверх мягкого STYLE_EXAMPLES_MAX: дальше — FIFO,
# даже если ИИ решил ничего не вытеснять.
HARD_OVERFLOW_SLACK = 16

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+|\bu\d+\b|tg://\S+", re.IGNORECASE)
LONG_DIGITS_RE = re.compile(r"\b\d{8,}\b")
CODE_FENCE_RE = re.compile(r"```|^\s*[-*•]\s+|\d+\.\s+", re.MULTILINE)


def prefilter_ok(text: Optional[str], min_len: int = 6, max_len: int = 150, allow_single: bool = False) -> bool:
    """Дешёвый кодовый префильтр до LLM-проверки: только живой текст чата."""
    if not text:
        return False
    t = text.strip()
    if not (min_len <= len(t) <= max_len):
        return False
    if t.startswith("/") or t.startswith("["):
        return False  # команды и кодовые заглушки ([фото], [стикер]...)
    if URL_RE.search(t) or MENTION_RE.search(t) or LONG_DIGITS_RE.search(t):
        return False  # ссылки/упоминания/id — не тащим в банк
    if CODE_FENCE_RE.search(t):
        return False
    if not allow_single and len(t.split()) < 2 and len(t) < 10:
        return False  # одиночные междометия без контекста не самодостаточны
    return True


def norm_entry(raw: Any) -> Optional[dict]:
    """Нормализует entry банка: строка (легаси) или dict."""
    if isinstance(raw, str):
        t = raw.strip()
        return {"text": t, "ctx": None, "laughs": 0, "ts": 0} if t else None
    if isinstance(raw, dict) and isinstance(raw.get("text"), str) and raw["text"].strip():
        return {
            "text": raw["text"].strip(),
            "ctx": (raw.get("ctx") or None),
            "laughs": int(raw.get("laughs") or 0),
            "ts": int(raw.get("ts") or 0),
        }
    return None


def render_entry(entry: dict, ctx_len: int = 80) -> str:
    """Строка примера для промпта: ответ + контекст, на что отвечали."""
    text = entry.get("text") or ""
    ctx = (entry.get("ctx") or "").strip()
    if ctx:
        ctx = re.sub(r"\s+", " ", ctx)[:ctx_len]
        return f"{text} (на: «{ctx}»)"
    return text


def sample_for_prompt(
    examples: list[dict],
    pinned: list[str],
    n: int,
    prime_n: int = 5,
) -> list[str]:
    """Взвешенная подвыборка ровно n: прайм (хиты по смеякам) первым,
    пермач — всегда, остальное — случайное из банка (разнообразие)."""
    pool = [e for e in (examples or []) if e.get("text")]
    prime = [e for e in pool if int(e.get("laughs") or 0) >= 3]
    rest = [e for e in pool if int(e.get("laughs") or 0) < 3]
    pinned = [str(p) for p in (pinned or []) if str(p).strip()]
    picks: list[dict] = []
    if prime and n > len(pinned):
        picks += random.sample(prime, min(len(prime), prime_n, n - len(pinned)))
    rendered = [render_entry(e) for e in picks]
    for p in pinned:
        if p not in rendered:
            rendered.append(p)
    if len(rendered) < n and rest:
        need = n - len(rendered)
        rendered += [render_entry(e) for e in random.sample(rest, min(len(rest), need))]
    return rendered[:max(n, len(pinned))]


class StyleLearner:
    """Сбор + промоушн примеров стиля. Вся LLM-работа — фоном (sys-лейн)."""

    def __init__(
        self,
        bot_config: Any,
        db_manager: Any,
        gateway: Any,
        prompts_dir: str = "prompts",
    ):
        self.config = bot_config
        self.db = db_manager
        self.gateway = gateway
        self.prompts_dir = Path(prompts_dir)
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "STYLE_LEARN_ENABLED", True))

    def _paths(self) -> tuple[Path, Path]:
        return (
            self.prompts_dir / CANDIDATES_FILE,
            self.prompts_dir / ACTIVE_FILE,
        )

    def _read_yaml_list(self, path: Path, key: str) -> list:
        try:
            if not path.exists():
                return []
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            items = data.get(key, [])
            return items if isinstance(items, list) else []
        except Exception as e:
            logger.warning("Style learner: failed to read %s: %s", path, e)
            return []

    def _pinned_seeds(self) -> list[str]:
        seeds = getattr(self.config, "STYLE_PINNED_SEEDS", None) or []
        return [str(s) for s in seeds if str(s).strip()]

    def _read_active(self, path: Path) -> tuple[list[dict], list[str]]:
        """(examples, pinned): пермач отдельно, вытеснение его не касается."""
        raw_examples = self._read_yaml_list(path, "examples")
        examples: list[dict] = []
        for r in raw_examples:
            e = norm_entry(r)
            if e:
                examples.append(e)
        pinned = [a for a in self._read_yaml_list(path, "pinned") if isinstance(a, str)]
        return examples, pinned

    def _ensure_seeds(self) -> None:
        """Сиды пермача (дефолт «бум») — дописать в файл один раз, если их там нет."""
        seeds = self._pinned_seeds()
        if not seeds:
            return
        _, active_path = self._paths()
        examples, pinned = self._read_active(active_path)
        have_texts = {e["text"] for e in examples} | set(pinned)
        missing = [s for s in seeds if s not in have_texts]
        if missing:
            self._write_yaml(active_path, {
                "examples": [e for e in examples],
                "pinned": pinned + missing,
            })

    def _write_yaml(self, path: Path, payload: dict) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            os.replace(tmp, path)
            try:
                os.chmod(path, 0o600)
            except Exception:
                pass
            return True
        except Exception as e:
            logger.warning("Style learner: failed to write %s: %s", path, e)
            return False

    def _known_texts(self, candidates: list, examples: list[dict], pinned: list[str]) -> set[str]:
        known = {c.get("text") for c in candidates if isinstance(c, dict) and c.get("text")}
        known |= {e["text"] for e in examples}
        known |= set(pinned)
        return known

    async def collect_from_message(
        self,
        chat_id: int,
        user_id: Optional[int],
        text: Optional[str],
        ctx: Optional[str] = None,
    ) -> bool:
        """Положить живую фразу в кандидаты. Возвращает True если взяли."""
        if not self.enabled or not user_id or not text:
            return False
        min_len = int(getattr(self.config, "STYLE_MIN_LEN", 6) or 6)
        max_len = int(getattr(self.config, "STYLE_MAX_LEN", 150) or 150)
        clean = scrub_pii(text.strip())
        if not prefilter_ok(clean, min_len, max_len):
            return False
        try:
            if await self.db.is_opt_out(chat_id, user_id):
                return False
        except Exception:
            return False

        async with self._lock:
            self._ensure_seeds()
            cand_path, active_path = self._paths()
            candidates = self._read_yaml_list(cand_path, "candidates")
            examples, pinned = self._read_active(active_path)
            if clean in self._known_texts(candidates, examples, pinned):
                return False
            cid = f"c{int(time.time())}{random.randint(100, 999)}"
            entry: dict[str, Any] = {"id": cid, "text": clean, "ts": int(time.time())}
            if ctx and ctx.strip():
                entry["ctx"] = re.sub(r"\s+", " ", ctx.strip())[:200]
            candidates.append(entry)
            # Потолок кандидатов, чтобы файл не пух
            candidates = candidates[-200:]
            ok = self._write_yaml(cand_path, {"candidates": candidates})
        if ok:
            logger.info("Style learner: candidate collected (%.40s...)", clean)
        return ok

    async def collect_reacted(
        self,
        chat_id: int,
        user_id: Optional[int],
        text: Optional[str],
        direct: bool = False,
        ctx: Optional[str] = None,
        laughs: int = 0,
    ) -> bool:
        """Кандидат по смеякам (с контекстом: на что был ответ).

        Смеяки почти всегда приходят ПОЗЖЕ сбора фразы: handle_message уже
        положил её в кандидаты (collect_from_message), а MessageReaction
        приходит минутами спустя. Поэтому здесь не «берём или молча False»,
        а ДОЛЛИВАЕМ сигнал: laughs существующей записи не понижается.
        Возвращает True только если файловое состояние реально изменилось.

        direct=True (3+ смеяк): сразу в банк, минуя очередь и фильтр — смех чата
        и есть вердикт. Это НЕ пермач: при переполнении вытесняется решением ИИ
        на общих основаниях, пермач — только сиды. Безопасность кодовая:
        PII-маска, без ссылок/упоминаний. Если фраза уже в банке (промоушена
        без смеяков) — поднимаем ей laughs (поздний хост чата = прайм), а если
        в pinned — не трогаем сиды (False).
        direct=False (1–2 смеяки): в очередь на отбор обычным путём,
        но префильтр мягче — короткое «бум»/«вау» тоже смотрим. Если фраза
        уже висит в очереди кандидатов — обновляем ей laughs и ctx вместо
        тихого False.
        """
        if not self.enabled or not text:
            return False
        max_len = int(getattr(self.config, "STYLE_MAX_LEN", 150) or 150)
        clean = scrub_pii(text.strip())
        # Прайм по смеякам: одиночное «бум»/«вау» со смеяками тоже смотрим
        if not prefilter_ok(clean, min_len=2, max_len=max_len, allow_single=True):
            return False
        if user_id:
            try:
                if await self.db.is_opt_out(chat_id, user_id):
                    return False
            except Exception:
                return False
        clean_ctx = re.sub(r"\s+", " ", (ctx or "").strip())[:200] or None

        async with self._lock:
            self._ensure_seeds()
            cand_path, active_path = self._paths()
            examples, pinned = self._read_active(active_path)
            if direct:
                if clean in set(pinned):
                    return False  # сиды пермача не трогаем
                in_bank = next((e for e in examples if e["text"] == clean), None)
                if in_bank is not None:
                    # Фраза уже в банке (её промоушнули без смеяков) — поздние
                    # смеяки не теряем, а поднимаем laughs до прайма (>=3)
                    changed = False
                    have_laughs = int(in_bank.get("laughs") or 0)
                    new_laughs = max(have_laughs, int(laughs or 0), 3)
                    if new_laughs != have_laughs:
                        in_bank["laughs"] = new_laughs
                        changed = True
                    if clean_ctx and not (in_bank.get("ctx") or "").strip():
                        in_bank["ctx"] = clean_ctx
                        changed = True
                    # Дубль из очереди кандидатов убрать: в банке уже есть
                    candidates = self._read_yaml_list(cand_path, "candidates")
                    left = [
                        c for c in candidates
                        if not (isinstance(c, dict) and c.get("text") == clean)
                    ]
                    if len(left) != len(candidates):
                        self._write_yaml(cand_path, {"candidates": left})
                        changed = True
                    if not changed:
                        return False
                    self._write_yaml(active_path, {"examples": examples, "pinned": pinned})
                    logger.warning("Style learner: DIRECT laughs updated in bank (%.40s...)", clean)
                    return True
                examples.append({
                    "text": clean, "ctx": clean_ctx,
                    "laughs": max(3, int(laughs or 3)), "ts": int(time.time()),
                })
                self._write_yaml(active_path, {"examples": examples, "pinned": pinned})
                # В банке — из кандидатов убрать дубль
                candidates = self._read_yaml_list(cand_path, "candidates")
                candidates = [c for c in candidates if c.get("text") != clean]
                self._write_yaml(cand_path, {"candidates": candidates})
                logger.warning("Style learner: DIRECT to bank by laughs (%.40s...)", clean)
            else:
                if clean in set(pinned):
                    return False  # сиды пермача не трогаем
                candidates = self._read_yaml_list(cand_path, "candidates")
                queued = next(
                    (c for c in candidates if isinstance(c, dict) and c.get("text") == clean),
                    None,
                )
                # Фраза уже в очереди (её собрал collect_from_message ещё до
                # смеяков) — не теряем сигнал, а долливаем смеяки кандидату
                if queued is not None:
                    changed = False
                    have_laughs = int(queued.get("laughs") or 0)
                    new_laughs = max(have_laughs, int(laughs or 0))
                    if new_laughs != have_laughs:
                        queued["laughs"] = new_laughs
                        changed = True
                    if clean_ctx and not (queued.get("ctx") or "").strip():
                        queued["ctx"] = clean_ctx
                        changed = True
                    if not changed:
                        return False
                    ok = self._write_yaml(cand_path, {"candidates": candidates})
                    if ok:
                        logger.info("Style learner: laughed candidate updated (%.40s...)", clean)
                    return ok
                if clean in self._known_texts(candidates, examples, pinned):
                    return False
                cid = f"c{int(time.time())}{random.randint(100, 999)}"
                entry = {"id": cid, "text": clean, "ts": int(time.time()),
                         "laughs": int(laughs or 0)}
                if clean_ctx:
                    entry["ctx"] = clean_ctx
                candidates.append(entry)
                candidates = candidates[-200:]
                ok = self._write_yaml(cand_path, {"candidates": candidates})
                if ok:
                    logger.info("Style learner: reacted candidate queued (%.40s...)", clean)
                return ok
        # Переполнение прямым добавлением чистит sys-воркер через ensure_capacity,
        # здесь не ждём (иначе тормозим хендлер реакций сетевыми вызовами)
        return True

    async def ensure_capacity(self) -> bool:
        """Проверить переполнение банка и при нужде вытеснить решением ИИ.

        Вызывать из sys-воркера (фон), не из горячего пути.
        Возвращает True если чистка была.
        """
        if not self.enabled:
            return False
        max_active = int(getattr(self.config, "STYLE_EXAMPLES_MAX", 128) or 128)
        async with self._lock:
            _, active_path = self._paths()
            examples, _ = self._read_active(active_path)
            if len(examples) <= max_active:
                return False
            snapshot = list(examples)
        await self._apply_ai_eviction(snapshot, max_active)
        return True

    async def maybe_promote(self) -> int:
        """Прогнать полный батч кандидатов через style_filter и дописать ok в актив.

        Возвращает число добавленных примеров.
        """
        if not self.enabled:
            return 0
        batch_n = int(getattr(self.config, "STYLE_FILTER_BATCH", 30) or 30)
        max_active = int(getattr(self.config, "STYLE_EXAMPLES_MAX", 128) or 128)
        async with self._lock:
            self._ensure_seeds()
            cand_path, active_path = self._paths()
            candidates = self._read_yaml_list(cand_path, "candidates")
            if len(candidates) < batch_n:
                return 0
            batch = candidates[:batch_n]

        lines = "\n".join(
            f"[{c.get('id')}] {c.get('text')}" + (f" (на: {c.get('ctx')})" if c.get("ctx") else "")
            for c in batch
        )
        filter_path = self.prompts_dir / "style_filter.txt"
        try:
            system_instruction = (
                filter_path.read_text(encoding="utf-8")
                if filter_path.exists()
                else "Отбери короткие живые реплики."
            )
        except Exception:
            system_instruction = "Отбери короткие живые реплики."
        try:
            verdicts: Optional[StyleVerdicts] = await self.gateway.generate(
                prompt_name="style_filter",
                contents=[{"role": "user", "parts": [{"text": f"<candidates>\n{lines}\n</candidates>"}]}],
                system_instruction=system_instruction,
                priority="P2",
                response_schema=StyleVerdicts,
                estimated_tokens=2000,
                lane="sys",
            )
        except Exception as e:
            logger.warning("Style learner: filter call failed: %s", e)
            return 0

        ok_ids: set[str] = set()
        try:
            for v in (verdicts.verdicts if verdicts else []):
                if v.ok:
                    ok_ids.add(v.id)
        except Exception:
            pass
        # порядок батча сохраняем (иначе банк прыгает)
        ok_entries = [c for c in batch if c.get("id") in ok_ids and (c.get("text") or "").strip()]

        async with self._lock:
            self._ensure_seeds()
            cand_path, active_path = self._paths()
            candidates = self._read_yaml_list(cand_path, "candidates")
            done_ids = {c.get("id") for c in batch}
            candidates = [c for c in candidates if c.get("id") not in done_ids]
            self._write_yaml(cand_path, {"candidates": candidates})
            added = 0
            need_evict = False
            if ok_entries:
                examples, pinned = self._read_active(active_path)
                have = {e["text"] for e in examples} | set(pinned)
                for c in ok_entries:
                    t = (c.get("text") or "").strip()
                    if t and t not in have:
                        examples.append({
                            "text": t,
                            "ctx": (c.get("ctx") or None),
                            "laughs": int(c.get("laughs") or 0),
                            "ts": int(c.get("ts") or int(time.time())),
                        })
                        have.add(t)
                        added += 1
                if len(examples) > max_active:
                    need_evict = True
                    evict_snapshot = list(examples)
                else:
                    self._write_yaml(active_path, {"examples": examples, "pinned": pinned})

        # Переполнение: вытеснение решением ИИ вне лока (там сетевые вызовы),
        # иначе сбор кандидатов встанет на время фильтрации всего банка
        if need_evict:
            await self._apply_ai_eviction(evict_snapshot, max_active)
        if added:
            logger.info("Style learner: promoted %d examples to active bank", added)
        return added

    async def _apply_ai_eviction(self, snapshot: list[dict], max_active: int) -> None:
        """Вытеснение решением ИИ: смотрит банк + «предложку», выбирает слабых.

        Может не вытеснить ничего — тогда банк временно пухнет, жёсткий
        потолок (max+16) держит FIFO. Пермач (pinned) не трогаем никогда.
        """
        # «Предложка» — текущая очередь кандидатов: ИИ видит, ради чего чистит
        try:
            cand_path, _ = self._paths()
            proposal = self._read_yaml_list(cand_path, "candidates")[-30:]
        except Exception:
            proposal = []
        weak = await self._plan_eviction(snapshot, proposal, max_active)
        async with self._lock:
            _, active_path = self._paths()
            examples, pinned = self._read_active(active_path)
            if weak is None:
                # ИИ лёг — честный FIFO
                examples = examples[-max_active:] if max_active > 0 else []
            elif weak:
                drop = set(weak)
                dropped = len([e for e in examples if e["text"] in drop])
                examples = [e for e in examples if e["text"] not in drop]
                if dropped:
                    logger.info("Style learner: AI evicting %d weakest examples", dropped)
                if len(examples) > max_active:
                    # ИИ пожадничал — режем перебор FIFO, самых старых не-прайм
                    prime = [e for e in examples if int(e.get("laughs") or 0) >= 3]
                    plain = [e for e in examples if int(e.get("laughs") or 0) < 3]
                    keep_plain = plain[-(max_active - len(prime)):] if max_active > len(prime) else []
                    examples = (prime + keep_plain)[-max_active:] if max_active > 0 else []
            # жёсткий потолок: дальше только FIFO
            hard = max_active + HARD_OVERFLOW_SLACK
            if len(examples) > hard:
                prime = [e for e in examples if int(e.get("laughs") or 0) >= 3]
                plain = [e for e in examples if int(e.get("laughs") or 0) < 3]
                keep_plain = plain[-(hard - len(prime)):] if hard > len(prime) else []
                examples = (prime + keep_plain)[-hard:] if hard > 0 else []
            self._write_yaml(active_path, {"examples": examples, "pinned": pinned})

    async def _plan_eviction(
        self, active: list[dict], proposal: list[dict], max_active: int
    ) -> Optional[set[str]]:
        """Спросить ИИ, кого выкинуть. Возвращает тексты на удаление,
        пустой set (может ничего не вытеснять) или None при ошибке."""
        over = len(active) - max_active
        if over <= 0:
            return set()
        evict_path = self.prompts_dir / "style_evict.txt"
        try:
            system_instruction = (
                evict_path.read_text(encoding="utf-8")
                if evict_path.exists()
                else "Выбери слабые реплики для удаления."
            )
        except Exception:
            system_instruction = "Выбери слабые реплики для удаления."
        lines = []
        for i, e in enumerate(active):
            ctx = f" (на: {e.get('ctx')})" if e.get("ctx") else ""
            mark = " [хит чата]" if int(e.get("laughs") or 0) >= 3 else ""
            lines.append(f"[e{i}] {e.get('text')}{ctx}{mark}")
        prop_lines = []
        for c in proposal or []:
            if isinstance(c, dict) and c.get("text"):
                ctx = f" (на: {c.get('ctx')})" if c.get("ctx") else ""
                prop_lines.append(f"- {c.get('text')}{ctx}")
        body = (
            f"<bank size={len(active)} max={max_active} over={over}>\n"
            + "\n".join(lines)
            + "\n</bank>\n<proposal>\n"
            + ("\n".join(prop_lines) if prop_lines else "(пусто)")
            + "\n</proposal>"
        )
        try:
            from src.gateway.schemas import StyleVerdicts as _SV
            verdicts: Optional[_SV] = await self.gateway.generate(
                prompt_name="style_evict",
                contents=[{"role": "user", "parts": [{"text": body}]}],
                system_instruction=system_instruction,
                priority="P2",
                response_schema=_SV,
                estimated_tokens=2000,
                lane="sys",
            )
            by_idx = {v.id: v.ok for v in (verdicts.verdicts if verdicts else [])}
            weak = {active[i]["text"] for i in range(len(active)) if by_idx.get(f"e{i}") is False}
            return weak
        except Exception as e:
            logger.warning("Style learner: eviction plan failed, FIFO fallback: %s", e)
            return None

    # --- back-compat для старых тестов/вызовов ---

    async def _filter_weak(self, active: list) -> Optional[set[str]]:
        """Легаси-обёртка: принимает и строки, и dict."""
        normed = [norm_entry(a) for a in active]
        normed = [e for e in normed if e]
        texts = [e["text"] for e in normed]
        weak_texts = await self._plan_eviction_texts(texts)
        return weak_texts

    async def _plan_eviction_texts(self, texts: list[str]) -> Optional[set[str]]:
        batch_n = int(getattr(self.config, "STYLE_FILTER_BATCH", 30) or 30)
        filter_path = self.prompts_dir / "style_filter.txt"
        try:
            system_instruction = (
                filter_path.read_text(encoding="utf-8")
                if filter_path.exists()
                else "Отбери короткие живые реплики."
            )
        except Exception:
            system_instruction = "Отбери короткие живые реплики."
        try:
            weak: set[str] = set()
            for start in range(0, len(texts), batch_n):
                chunk = texts[start:start + batch_n]
                lines = "\n".join(f"[e{start + i}] {t}" for i, t in enumerate(chunk))
                verdicts: Optional[StyleVerdicts] = await self.gateway.generate(
                    prompt_name="style_filter",
                    contents=[{"role": "user", "parts": [{"text": f"<candidates>\n{lines}\n</candidates>"}]}],
                    system_instruction=system_instruction,
                    priority="P2",
                    response_schema=StyleVerdicts,
                    estimated_tokens=2000,
                    lane="sys",
                )
                by_idx = {v.id: v.ok for v in (verdicts.verdicts if verdicts else [])}
                for i, t in enumerate(chunk):
                    if by_idx.get(f"e{start + i}") is False:
                        weak.add(t)
            return weak
        except Exception as e:
            logger.warning("Style learner: eviction filter failed, FIFO fallback: %s", e)
            return None

    async def _apply_weakest_eviction(self, snapshot: list, max_active: int) -> None:
        """Легаси-обёртка для тестов: строки или dict."""
        normed = [norm_entry(a) for a in snapshot]
        normed = [e for e in normed if e]
        weak = await self._plan_eviction_texts([e["text"] for e in normed])
        async with self._lock:
            _, active_path = self._paths()
            examples, pinned = self._read_active(active_path)
            if weak is None:
                examples = examples[-max_active:] if max_active > 0 else []
            else:
                dropped = len([e for e in examples if e["text"] in weak])
                examples = [e for e in examples if e["text"] not in weak]
                if dropped:
                    logger.info("Style learner: evicting %d weakest examples", dropped)
                if len(examples) > max_active:
                    examples = examples[-max_active:] if max_active > 0 else []
            self._write_yaml(active_path, {"examples": examples, "pinned": pinned})
