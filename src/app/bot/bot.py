import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../recsys/'))
sys.path.insert(1, os.path.join(sys.path[0], '../db/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../config/'))

import logging
import pandas as pd

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, \
    filters, CallbackQueryHandler, PicklePersistence, ConversationHandler
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from utils import get_dists, get_recommendations
from db_connection import DB
from config import db_config


DATA_PATH = "../../../data/"
MODELS_PATH = "../../../models/"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

matrix = get_dists(DATA_PATH, cosine_similarity)
df = pd.read_csv(DATA_PATH + 'anime_cl.csv')
names_list = df['Name'].to_list()
db = DB(db_config)
CHOOSING, TYPING_REPLY, TYPING_CHOICE = range(3)

reply_keyboard = [
    ["Rate anime", "Evaluate the recommendation"],
    ["Done"],
]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}!\n"
        f"I'm a bot who knows how to recommend anime similar to your favorite, "
        f"enter the name of your favorite anime, and I'll show you the top 5 most similar :)",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Enter the name of the anime, and I'll show you the top 5 most similar anime :)")


async def anime_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    request = update.message.text
    user = update.effective_user
    db.save_user_info(user['id'], user['language_code'], user['first_name'], user['last_name'], user['username'],
                      update.message.date, request)
    name_anime = get_close_matches(request, names_list)
    if name_anime:
        keyboard = [[InlineKeyboardButton(name, callback_data=name)] for name in name_anime]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text('Please confirm the name', reply_markup=reply_markup)
    else:
        await update.message.reply_text("There is no such anime in the list, please repeat the request again")


async def button_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    variant = query.data

    recs = get_recommendations(matrix, df, variant)
    await query.message.reply_text(f"TOP similar anime to {variant}:\n")
    for idx, rec in recs.iterrows():
        if idx == 5:
            break
        else:
            await query.message.reply_text(f"{idx + 1}. {rec['Name']}\n"
                                           f"Score: {rec['Score']}\n"
                                           f"Year: {rec['year']}\n"
                                           f"Type: {rec['Type']}\n"
                                           f"Episodes: {int(rec['Episodes'])}\n"
                                           f"Description:\n{rec['sypnopsis']}\n")


async def helpme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You can improve me!\n"
                                    "Enter the name and rating of your favorite "
                                    "anime or rate the recommendations I gave you :)", reply_markup=markup)

    return CHOOSING


async def regular_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Ask the user for info about the selected predefined choice."""
    text = update.message.text.lower()
    if text == 'rate anime':
        reply_text = (
            f"Please enter the name of the anime and the rating you want to give it, separated by commas.\n"
            f"Example: Death Note, 9/10"
        )
    else:
        reply_text = f"Please enter the name of the anime you were looking for a recommendation for, " \
                     f"the name of the anime I recommended to you and a rating from 0 to 5." \
                     f"\nExample: Your name, Gothic, 4/5"

    await update.message.reply_text(reply_text)

    return TYPING_REPLY


async def received_information(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store info provided by user and ask for the next category."""
    user = update.effective_user
    feedback = update.message.text
    db.save_user_feedback(user['id'], user['language_code'], user['first_name'], user['last_name'], user['username'],
                          update.message.date, feedback)

    await update.message.reply_text("Thank you so much for the feedback.\n"
                                    "Can you give feedback about some more anime?", reply_markup=markup)

    return CHOOSING


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Display the gathered info and end the conversation."""
    await update.message.reply_text("Thank you for your feedback!\n"
                                    "I will study and my recommendations will be even better, although much better :)",
                                    reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


def main() -> None:
    """Start the bot."""
    persistence = PicklePersistence(filepath="conversationbot")
    application = Application.builder().token("TOKEN")\
        .persistence(persistence).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("helpme", helpme)],
        states={
            CHOOSING: [
                MessageHandler(
                    filters.Regex("^(Rate anime|Evaluate the recommendation)$"), regular_choice
                )
            ],
            TYPING_CHOICE: [
                MessageHandler(
                    filters.TEXT & ~(filters.COMMAND | filters.Regex("^Done$")), regular_choice
                )
            ],
            TYPING_REPLY: [
                MessageHandler(
                    filters.TEXT & ~(filters.COMMAND | filters.Regex("^Done$")),
                    received_information,
                )
            ],
        },
        fallbacks=[MessageHandler(filters.Regex("^Done$"), done)],
        name="my_conversation",
        persistent=True,
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, anime_name))
    application.add_handler(CallbackQueryHandler(button_name))

    application.run_polling()


main()
