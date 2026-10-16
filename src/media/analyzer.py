import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from src.ops.config import BotConfig
from src.ops.log import logger
from src.db.manager import DatabaseManager
from src.db.contracts import Gateway
from src.gateway.schemas import MediaAnalysis, MediaKind
from src.context.prompts import PromptBuilder


SCORE_PATTERN = re.compile(r"\b(\d{1,3})/100\b")


def contains_forbidden_score(reply_text: str, kind: Optional[str]) -> bool:
    """True, если в ответе есть N/100, а kind — не audio (фото/мем/видео/войс)."""
    if kind == "audio":
        return False
    return bool(SCORE_PATTERN.search(reply_text or ""))


def describe_person(alias: Optional[str], nick: Optional[str]) -> str:
    """Короткое человекочитаемое «u7 «Петя»» для JSON и логов."""
    a = (alias or "").strip() or "u0"
    n = (nick or "").strip() or "Unknown"
    return f"{a} «{n}»"


def build_step2_media_json(
    media_cache_item: dict[str, Any],
    sent_by_alias: Optional[str] = None,
    sent_by_nick: Optional[str] = None,
    requested_by_alias: Optional[str] = None,
    requested_by_nick: Optional[str] = None,
) -> str:
    """Урезанный JSON разбора для <media> + КТО прислал вложение и КТО просит ответить.

    Различает автора вложения (sent_by) и автора запроса (requested_by):
    другой чел скинул трек, а ты реплаем попросил оценку — трек не твой.
    """
    summary_val = media_cache_item.get("summary") or ""
    kind = media_cache_item.get("kind")
    is_audio = (kind == "audio")
    data = {
        "kind": kind,
        "summary": summary_val[:300],
        "ocr_text": media_cache_item.get("ocr_text")[:300] if media_cache_item.get("ocr_text") else None,
        "verdict": media_cache_item.get("verdict"),
        "score100": media_cache_item.get("score100") if is_audio else None,
        "score_reason": media_cache_item.get("score_reason") if is_audio else None,
        "track_meta": media_cache_item.get("track_meta"),
        "tags": media_cache_item.get("tags"),
        "unintelligible": media_cache_item.get("unintelligible"),
        "injection_suspected": media_cache_item.get("injection_suspected"),
        "sent_by": describe_person(sent_by_alias, sent_by_nick),
        "requested_by": describe_person(requested_by_alias, requested_by_nick),
    }
    return json.dumps(data, ensure_ascii=False)


class MediaAnalyzer:
    """Manages media normalization with ffmpeg, inline base64 calls to Gemini, caching, and score verification."""

    def __init__(
        self,
        bot_config: BotConfig,
        db_manager: DatabaseManager,
        gateway: Gateway,
        prompt_builder: PromptBuilder,
        tmp_dir: str = "/tmp"
    ):
        self.config = bot_config
        self.db = db_manager
        self.gateway = gateway
        self.prompts = prompt_builder
        self.tmp_dir = Path(tmp_dir)
        # Дедуп параллельных разборов одного file_unique_id (спам одной гифкой)
        self._inflight: dict[str, asyncio.Task] = {}

    def _cleanup_old_tmp_files(self, max_age_hours: int = 1) -> None:
        """Sweeps temp files older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        try:
            for p in self.tmp_dir.glob("smesh-*"):
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Error cleaning temp files: %s", e)

    def _photo_downscale(self, input_path: Path) -> Path:
        """Ужимает фото до PHOTO_MAX_DIM по длинной стороне (PIL, иначе ffmpeg)."""
        max_dim = int(getattr(self.config, "PHOTO_MAX_DIM", 1280) or 1280)
        quality = int(getattr(self.config, "PHOTO_JPEG_Q", 72) or 72)
        out_path = self.tmp_dir / f"smesh-{input_path.stem}.photo.jpg"
        # 1. Пробуем PIL — быстро и без форка ffmpeg
        try:
            from PIL import Image  # type: ignore

            def _do_pil() -> Optional[Path]:
                try:
                    with Image.open(input_path) as im:
                        im = im.convert("RGB")
                        w, h = im.size
                        scale = min(1.0, max_dim / max(w, h))
                        if scale < 1.0:
                            im = im.resize((int(w * scale), int(h * scale)))
                        im.save(out_path, "JPEG", quality=quality, optimize=True)
                    return out_path if out_path.exists() else None
                except Exception as e:
                    logger.warning("PIL photo downscale failed: %s", e)
                    return None

            # вызывающий код уже в to_thread, тут можно синхронно
            res = _do_pil()
            if res:
                return res
        except ImportError:
            pass
        # 2. Fallback: ffmpeg scale
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-vf", f"scale='min(iw,{max_dim})':'min(ih,{max_dim})':eval=frame",
                "-q:v", "4",
                str(out_path),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=25)
            if out_path.exists():
                return out_path
        except Exception as e:
            logger.warning("ffmpeg photo downscale failed, using original: %s", e)
        return input_path

    def convert_media_if_needed(self, input_path: Path, kind: str) -> Path:
        """
        Конвертация/ужимка (синхронная, вызывать через to_thread):
        - audio/voice -> mp3 mono 64k
        - video/video_note -> mp4 первые VIDEO_MAX_S сек, 480p, veryfast (было 60с/fast)
        - animation (гифка) -> mp4 первые ANIM_MAX_S сек, 480p, без аудио (главный фикс ступора)
        - sticker -> jpg первый кадр
        - photo -> downscale до PHOTO_MAX_DIM (было: оригинал 5МБ в base64)
        """
        self._cleanup_old_tmp_files()
        out_path = self.tmp_dir / f"smesh-{input_path.stem}.converted"

        if kind in ("voice", "audio"):
            out_path = out_path.with_suffix(".mp3")
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-ac", "1", "-b:a", "64k",
                str(out_path)
            ]
        elif kind in ("video", "video_note"):
            max_s = int(getattr(self.config, "VIDEO_MAX_S", 20) or 20)
            height = int(getattr(self.config, "VIDEO_HEIGHT", 480) or 480)
            out_path = out_path.with_suffix(".mp4")
            cmd = [
                "ffmpeg", "-y", "-t", str(max_s), "-i", str(input_path),
                "-vf", f"scale=-1:{height}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                "-c:a", "aac", "-b:a", "48k",
                str(out_path)
            ]
        elif kind == "animation":
            max_s = int(getattr(self.config, "ANIM_MAX_S", 4) or 4)
            height = int(getattr(self.config, "ANIM_HEIGHT", 480) or 480)
            out_path = out_path.with_suffix(".mp4")
            cmd = [
                "ffmpeg", "-y", "-t", str(max_s), "-i", str(input_path),
                "-vf", f"fps=12,scale=-1:{height}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                "-an",
                str(out_path)
            ]
        elif kind == "sticker":
            out_path = out_path.with_suffix(".jpg")
            cmd = [
                "ffmpeg", "-y", "-vframes", "1", "-i", str(input_path),
                str(out_path)
            ]
        elif kind == "photo":
            return self._photo_downscale(input_path)
        else:
            return input_path

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=60)
            if out_path.exists():
                return out_path
        except Exception as e:
            logger.warning("ffmpeg conversion failed, using original file: %s", e)

        return input_path

    def _detect_silence_db(self, input_path: Path) -> Optional[float]:
        """Средняя громкость через ffmpeg volumedetect (синхронно, вызывать через to_thread).

        Возвращает mean_volume в дБ или None, если замерить не вышло.
        """
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-i", str(input_path),
                 "-af", "volumedetect", "-f", "null", os.devnull],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                check=False, timeout=30, text=True,
            )
            m = re.search(r"mean_volume:\s*(-?[\d.]+|-\s*inf)\s*dB", proc.stderr or "")
            if not m:
                return None
            val = m.group(1).replace(" ", "")
            if "inf" in val:
                return float("-inf")
            return float(val)
        except Exception as e:
            logger.warning("Silence detect failed for %s: %s", input_path, e)
            return None

    def _silent_audio_item(
        self, file_unique_id: str, mean_db: Optional[float]
    ) -> dict[str, Any]:
        """Готовый разбор «тишины» без LLM: тихий файл нельзя оценить в 60+."""
        loud = f"{mean_db:.1f} дБ" if mean_db is not None and mean_db != float("-inf") else "тишина"
        return {
            "file_unique_id": file_unique_id,
            "kind": "audio",
            "transcript": None,
            "ocr_text": None,
            "summary": f"тишина, почти ничего не слышно ({loud})",
            "track_meta": None,
            "tags": '["тишина"]',
            "verdict": "три минуты тишины",
            "score100": 0,
            "score_reason": "там тупо тишина",
            "unintelligible": 0,
            "injection_suspected": 0,
        }

    async def analyze_media_step1(
        self,
        file_path: Path,
        kind: str,
        file_unique_id: str,
        caption: Optional[str] = None,
        priority: str = "P0",
        lane: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Step 1: cache -> convert (to_thread, не блокирует loop) -> base64 -> LLM -> cache.
        Один file_unique_id разбирается один раз (in-flight дедуп).
        lane: track для audio, media для остального — чтобы треки не ждали гифки.
        """
        # 1. Кэш
        cached = await self.db.get_media_cache(file_unique_id)
        if cached:
            logger.info("Media %s served from cache", file_unique_id)
            return cached

        # 2. In-flight дедуп: повторный запрос того же файла ждёт первый
        existing = self._inflight.get(file_unique_id)
        if existing and not existing.done():
            try:
                return await existing
            except Exception:
                pass

        task: asyncio.Task = asyncio.create_task(
            self._analyze_uncached(file_path, kind, file_unique_id, caption, priority, lane)
        )
        self._inflight[file_unique_id] = task
        try:
            return await task
        finally:
            self._inflight.pop(file_unique_id, None)

    async def _analyze_uncached(
        self,
        file_path: Path,
        kind: str,
        file_unique_id: str,
        caption: Optional[str],
        priority: str,
        lane: Optional[str],
    ) -> Optional[dict[str, Any]]:
        # Конвертация вне event loop (ffmpeg/PIL блокируют!)
        try:
            converted_path: Path = await asyncio.to_thread(
                self.convert_media_if_needed, file_path, kind
            )
        except Exception as e:
            logger.warning("Media convert to_thread failed: %s", e)
            converted_path = file_path

        # Тихий/пустой трек режем кодом, а не вкусом модели: иначе тишина
        # получает 60+ и выдуманные «арабские дудки». Порог −50 дБ.
        if kind == "audio":
            try:
                mean_db: Optional[float] = await asyncio.to_thread(
                    self._detect_silence_db, converted_path
                )
            except Exception:
                mean_db = None
            if mean_db is not None and mean_db <= -50.0:
                item = self._silent_audio_item(file_unique_id, mean_db)
                try:
                    await self.db.save_media_cache(item, max_items=self.config.MEDIA_CACHE_MAX)
                except Exception as e:
                    logger.warning("Failed to cache silence verdict: %s", e)
                try:
                    if converted_path != file_path and converted_path.exists():
                        converted_path.unlink(missing_ok=True)
                except Exception:
                    pass
                logger.info("Media %s is silence (%.1f dB), scored 0 without LLM", file_unique_id, mean_db)
                return item

        # Чтение + base64 тоже вне loop
        def _read_b64(p: Path) -> Optional[str]:
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                logger.error("Failed to read media file for inline base64: %s", e)
                return None

        b64_data = await asyncio.to_thread(_read_b64, converted_path)
        try:
            if converted_path != file_path and converted_path.exists():
                converted_path.unlink(missing_ok=True)
        except Exception:
            pass
        if not b64_data:
            return None

        mime_type_map = {
            "voice": "audio/mp3",
            "audio": "audio/mp3",
            "video": "video/mp4",
            "video_note": "video/mp4",
            "photo": "image/jpeg",
            "animation": "video/mp4",
            "sticker": "image/jpeg",
        }
        mime_type = mime_type_map.get(kind, "application/octet-stream")

        system_instruction = self.prompts.build_system_instruction(
            mode_or_modular="media_analyze",
            bot_name="Дик"
        )

        parts: list[dict[str, Any]] = [
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": b64_data
                }
            }
        ]
        if caption:
            parts.append({"text": f"<caption>{caption}</caption>"})
        if kind == "sticker":
            req_kind = "photo"
        elif kind == "animation":
            req_kind = "video"
        else:
            req_kind = kind
        parts.append({"text": f'<request kind="{req_kind}"/>'})

        contents = [{"role": "user", "parts": parts}]

        resolved_lane = lane or ("track" if kind == "audio" else "media")
        analysis_obj: Optional[MediaAnalysis] = await self.gateway.generate(
            prompt_name="media_analyze",
            contents=contents,
            system_instruction=system_instruction,
            priority=priority,
            response_schema=MediaAnalysis,
            estimated_tokens=1500,
            lane=resolved_lane,
        )

        if not analysis_obj or not isinstance(analysis_obj, MediaAnalysis):
            logger.warning("media_analyze call failed for file %s", file_unique_id)
            return None

        # 4. Save to SQLite media_cache
        item_dict = {
            "file_unique_id": file_unique_id,
            "kind": kind,
            "transcript": analysis_obj.transcript,
            "ocr_text": analysis_obj.ocr_text,
            "summary": analysis_obj.summary,
            "track_meta": json.dumps(analysis_obj.track_meta.model_dump(), ensure_ascii=False) if analysis_obj.track_meta else None,
            "tags": json.dumps(analysis_obj.tags, ensure_ascii=False),
            "verdict": analysis_obj.verdict,
            "score100": analysis_obj.score100 if kind == "audio" else None,
            "score_reason": analysis_obj.score_reason if kind == "audio" else None,
            "unintelligible": 1 if analysis_obj.unintelligible else 0,
            "injection_suspected": 1 if analysis_obj.injection_suspected else 0,
        }

        await self.db.save_media_cache(item_dict, max_items=self.config.MEDIA_CACHE_MAX)
        return item_dict

    def verify_and_patch_reply_score(self, reply_text: str, media_cache_item: dict[str, Any]) -> str:
        """
        Step 2 post-processing per R3/TB-07:
        Checks that reply has N/100 matching the cache. If missing and score100 is present
        (and unintelligible/injection_suspected are False), appends N/100 + reason.
        IMPORTANT: Scores (N/100) are ONLY allowed for audio tracks.
        For all other media (photo, video, meme, sticker, voice), scores are strictly prohibited.
        """
        kind = media_cache_item.get("kind")
        if kind != "audio":
            # For non-audio media, remove any hallucinated N/100 score and NEVER append a score!
            return SCORE_PATTERN.sub("", reply_text).strip()

        score100 = media_cache_item.get("score100")
        unintelligible = bool(media_cache_item.get("unintelligible"))
        injection = bool(media_cache_item.get("injection_suspected"))

        # If unintelligible or injection, no score
        if unintelligible or injection or score100 is None:
            # Strip any hallucinated N/100 score
            return SCORE_PATTERN.sub("", reply_text).strip()

        expected_score_str = f"{score100}/100"
        match = SCORE_PATTERN.search(reply_text)
        if match:
            # Check if number matches
            found_str = match.group(0)
            if found_str != expected_score_str:
                # Replace with accurate score
                reply_text = reply_text[:match.start()] + expected_score_str + reply_text[match.end():]
            return reply_text

        # Score missing entirely: append it
        reason = media_cache_item.get("score_reason") or media_cache_item.get("verdict") or ""
        reason_part = f" — {reason}" if reason else ""
        return f"{reply_text}\n{expected_score_str}{reason_part}".strip()
