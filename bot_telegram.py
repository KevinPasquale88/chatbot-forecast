import argparse
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from chatbot_core import DATASETS, dataset_selection_prompt, handle_message, start_session

load_dotenv()
logging.basicConfig(level=logging.INFO)

sessions = {}
pending_dataset_choice = set()
preset_dataset = None


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    sessions.pop(chat_id, None)
    pending_dataset_choice.discard(chat_id)

    if preset_dataset:
        session, greeting = start_session(preset_dataset)
        sessions[chat_id] = session
        await update.message.reply_text(greeting)
        return

    pending_dataset_choice.add(chat_id)
    await update.message.reply_text(dataset_selection_prompt())


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text

    if chat_id in pending_dataset_choice:
        choice = text.strip().lower()
        if choice not in DATASETS:
            await update.message.reply_text(dataset_selection_prompt())
            return
        pending_dataset_choice.discard(chat_id)
        session, greeting = start_session(choice)
        sessions[chat_id] = session
        await update.message.reply_text(greeting)
        return

    if chat_id not in sessions:
        if preset_dataset:
            session, greeting = start_session(preset_dataset)
            sessions[chat_id] = session
            await update.message.reply_text(greeting)
            return
        pending_dataset_choice.add(chat_id)
        await update.message.reply_text(dataset_selection_prompt())
        return

    reply, updated_session = handle_message(sessions[chat_id], text)
    sessions[chat_id] = updated_session
    await update.message.reply_text(reply)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the clinical risk Telegram chatbot.")
    parser.add_argument(
        "--dataset", choices=DATASETS, default=None,
        help="Clinical dataset to fix for every chat (default: ask each user to choose)."
    )
    return parser.parse_args()


def main(dataset=None):
    global preset_dataset
    preset_dataset = dataset

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN non impostato in .env")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    print("[INFO] Telegram bot avviato. In ascolto di messaggi...")
    if preset_dataset:
        print(f"[INFO] Dataset fissato: {preset_dataset}")
    app.run_polling()


if __name__ == "__main__":
    main(dataset=parse_args().dataset)
