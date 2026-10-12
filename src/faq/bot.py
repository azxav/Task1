import logging

import httpx
from telegram.ext import Application, CommandHandler, MessageHandler, filters

RETRY = "Xizmat vaqtincha ishlamayapti. Iltimos, birozdan so'ng qayta urinib ko'ring."


def split_message(text: str, limit: int = 4096) -> list[str]:
    # Conservative UTF-16 counting keeps astral characters within Telegram limits.
    chunks, current, units = [], [], 0
    for char in text:
        width = len(char.encode("utf-16-le")) // 2
        if units + width > limit:
            chunks.append("".join(current))
            current, units = [], 0
        current.append(char)
        units += width
    if current:
        chunks.append("".join(current))
    return chunks


def response_messages(result):
    text = result["answer"] if result["status"] == "matched" else result["message"]
    return split_message(text)


def run_bot(settings):
    if not settings.token:
        raise ValueError("Set TELEGRAM_BOT_TOKEN in local .env")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    async def initialize(application):
        application.bot_data["api_client"] = httpx.AsyncClient(base_url=settings.api_url, timeout=45)

    async def shutdown(application):
        await application.bot_data["api_client"].aclose()

    async def start(update, context):
        await update.message.reply_text(
            "Davlat xizmatlari bo'yicha savolingizni matn shaklida yozing. "
            "Javoblar berilgan savol-javob bazasidan olinadi."
        )

    async def query(update, context):
        try:
            response = await context.application.bot_data["api_client"].post(
                "/search", json={"question": update.message.text}
            )
            if response.status_code == 422:
                messages = ["Savolni bo'sh bo'lmagan, 2000 belgidan qisqa matn sifatida yuboring."]
            else:
                response.raise_for_status()
                messages = response_messages(response.json())
        except (httpx.HTTPError, ValueError, KeyError):
            messages = [RETRY]
        for message in messages:
            await update.message.reply_text(message, parse_mode=None)

    async def nontext(update, context):
        await update.message.reply_text("Iltimos, savolingizni matn shaklida yuboring.")

    async def error(update, context):
        # Exception reprs can contain request URLs; never log the bot token or chat text.
        logging.getLogger(__name__).warning("Telegram handler failure kind=%s", type(context.error).__name__)

    application = (
        Application.builder().token(settings.token).post_init(initialize).post_shutdown(shutdown).build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, query))
    application.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, nontext))
    application.add_error_handler(error)
    application.run_polling(drop_pending_updates=False, allowed_updates=["message"])
