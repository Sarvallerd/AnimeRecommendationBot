mod db;

use db::Db;
use std::{str::FromStr, sync::Arc};
use teloxide::{
    dispatching::{
        dialogue::{self, InMemStorage},
        UpdateHandler,
    },
    prelude::*,
    types::{InlineKeyboardButton, InlineKeyboardMarkup},
    utils::command::BotCommands,
};

type MyDialogue = Dialogue<State, InMemStorage<State>>;
type HandlerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Default)]
pub enum State {
    #[default]
    Start,
    ReceiveAnimeName,
    ReceiveUserChoice,
    ReceiveUserFeedback,
    SuccessRecommendation,
}

/// These commands are supported:
#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase")]
enum Command {
    /// Display this text.
    Help,
    /// Start the purchase procedure.
    Start,
    /// Cancel the purchase procedure.
    Cancel,
}

#[tokio::main]
async fn main() {
    pretty_env_logger::init();
    log::info!("Starting purchase bot...");

    let bot = Bot::from_env();
    let db = Db::new().await.expect("Failed to initialize database");
    db.create().await.unwrap();
    let arc_db = Arc::new(db);

    Dispatcher::builder(bot, schema(arc_db))
        .dependencies(dptree::deps![InMemStorage::<State>::new()])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;
}

fn schema(db: Arc<Db>) -> UpdateHandler<Box<dyn std::error::Error + Send + Sync + 'static>> {
    use dptree::case;
    let db_name: Arc<Db> = db.clone();
    let db_feedback = db.clone();

    let command_handler = teloxide::filter_command::<Command, _>()
        .branch(
            case![State::Start]
                .branch(case![Command::Help].endpoint(help))
                .branch(case![Command::Start].endpoint(move |bot, dialogue, msg| {
                    let db = Arc::clone(&db);
                    start(bot, dialogue, msg, db)
                })),
        )
        .branch(case![Command::Cancel].endpoint(cancel));

    let message_handler = Update::filter_message()
        .branch(command_handler)
        .branch(case![State::SuccessRecommendation].endpoint(success_recommendation))
        .branch(
            case![State::ReceiveAnimeName].endpoint(move |bot, dialogue, msg| {
                let db_name = Arc::clone(&db_name);
                receive_anime_name(bot, dialogue, msg, db_name)
            }),
        )
        .branch(
            case![State::ReceiveUserFeedback].endpoint(move |bot, dialogue, msg| {
                let db_feedback = Arc::clone(&db_feedback);
                receive_user_feedback(bot, dialogue, msg, db_feedback)
            }),
        )
        .branch(dptree::endpoint(invalid_state));

    let callback_query_handler = Update::filter_callback_query()
        .branch(case![State::ReceiveUserChoice].endpoint(receive_user_choice));

    dialogue::enter::<Update, InMemStorage<State>, State, _>()
        .branch(message_handler)
        .branch(callback_query_handler)
}

async fn start(bot: Bot, dialogue: MyDialogue, msg: Message, db: Arc<Db>) -> HandlerResult {
    let user = msg.from.unwrap();
    let response = format!(
        "Hi {0}!
        \nI'm a bot who knows how to recommend anime similar to your favorite.",
        user.first_name
    );

    db.insert_user(
        user.id.to_string(),
        user.language_code
            .unwrap_or(String::from_str("NULL").unwrap()),
        user.first_name,
        user.last_name.unwrap_or(String::from_str("NULL").unwrap()),
        user.username.unwrap_or(String::from_str("NULL").unwrap()),
    )
    .await?;
    bot.send_message(msg.chat.id, response).await?;
    bot.send_message(
        msg.chat.id,
        "Enter the name of your favorite anime, and I'll show you the top 5 most similar :)",
    )
    .await?;
    dialogue.update(State::ReceiveAnimeName).await?;
    Ok(())
}

async fn help(bot: Bot, msg: Message) -> HandlerResult {
    bot.send_message(msg.chat.id, Command::descriptions().to_string())
        .await?;
    Ok(())
}

async fn cancel(bot: Bot, dialogue: MyDialogue, msg: Message) -> HandlerResult {
    bot.send_message(msg.chat.id, "Cancelling the dialogue.")
        .await?;
    dialogue.exit().await?;
    Ok(())
}

async fn invalid_state(bot: Bot, msg: Message) -> HandlerResult {
    bot.send_message(
        msg.chat.id,
        "Unable to handle the message. Type /help to see the usage.",
    )
    .await?;
    Ok(())
}

async fn receive_anime_name(
    bot: Bot,
    dialogue: MyDialogue,
    msg: Message,
    db: Arc<Db>,
) -> HandlerResult {
    match msg.text().map(ToOwned::to_owned) {
        Some(anime_name) => {
            let user = msg.clone().from.unwrap();
            db.insert_msg("request", user.id.to_string(), anime_name.clone())
                .await?;

            let anime_recs = vec!["Anime1", "Anime2", "Anime3", "Anime4", "Anime5"]; // Здесь место рекоммендашки!
            bot.send_message(
                msg.chat.id,
                format!("Here's what I found for {}", anime_name),
            )
            .await?;
            for rec in anime_recs {
                bot.send_message(msg.chat.id, rec).await?;
            }
            success_recommendation(bot, dialogue, msg).await?;
        }
        None => {
            bot.send_message(msg.chat.id, "Please, send me name of anime.")
                .await?;
        }
    }

    Ok(())
}

async fn receive_user_feedback(
    bot: Bot,
    dialogue: MyDialogue,
    msg: Message,
    db: Arc<Db>,
) -> HandlerResult {
    match msg.text().map(ToOwned::to_owned) {
        Some(feedback) => {
            let user = msg.from.unwrap();
            db.insert_msg("feedback", user.id.to_string(), feedback)
                .await?;
            bot.send_message(dialogue.chat_id(), "Thank you!").await?;
            dialogue.exit().await?;
        }
        None => {
            bot.send_message(msg.chat.id, "Please, send feedback again.")
                .await?;
        }
    }

    Ok(())
}

async fn receive_user_choice(bot: Bot, dialogue: MyDialogue, q: CallbackQuery) -> HandlerResult {
    match q.data.unwrap().as_str() {
        "More Recommendations" => {
            bot.send_message(dialogue.chat_id(), "Enter the name of anime")
                .await?;
            dialogue.update(State::ReceiveAnimeName).await?;
        }
        "Feedback" => {
            bot.send_message(
                dialogue.chat_id(),
                "Write your feedback. I'll save it for my developer",
            )
            .await?;
            dialogue.update(State::ReceiveUserFeedback).await?;
        }
        "Exit the dialog :(" => {
            bot.send_message(dialogue.chat_id(), "Good Bye").await?;
            dialogue.exit().await?;
        }
        _ => {}
    }

    Ok(())
}

async fn success_recommendation(bot: Bot, dialogue: MyDialogue, msg: Message) -> HandlerResult {
    let products = ["More Recommendations", "Feedback", "Exit the dialog :("]
        .map(|product| InlineKeyboardButton::callback(product, product));

    bot.send_message(
        msg.chat.id,
        "How do you like my recommendations?\nYou can leave feedback or continue using it!",
    )
    .reply_markup(InlineKeyboardMarkup::new([products]))
    .await?;

    dialogue.update(State::ReceiveUserChoice).await?;

    Ok(())
}
