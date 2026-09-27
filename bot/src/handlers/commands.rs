use super::{feedback, search, ui, HandlerResult};
use crate::{
    db::UserProfile,
    dialogue::{
        callback::Action,
        context::AppContext,
        state::{Actor, AnimeIntent},
        storage::Session,
    },
};
use teloxide::{prelude::*, types::User};

pub const HELP: &str = "Команды: /start — начать; /help — помощь; /cancel — отменить; /recommend — рекомендации; /rate — оценить аниме; /feedback — отзыв.";
pub const STALE: &str = "Эта кнопка устарела. Начните заново: /start";

pub async fn start(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    user: &User,
    session: &mut Session,
) -> HandlerResult {
    session.reset();
    let profile = UserProfile {
        tg_id: actor.user_id,
        first_name: user.first_name.clone(),
        language_code: user.language_code.clone(),
        last_name: user.last_name.clone(),
        username: user.username.clone(),
    };
    let saved = ctx.repository.upsert_user(&profile).await.is_ok();
    if !saved {
        bot.send_message(
            ChatId(actor.chat_id),
            "Не удалось подтвердить сохранение профиля. Навигация доступна; попробуйте /start позже.",
        )
        .await?;
    }
    ui::send_keyboard(
        bot,
        actor,
        session,
        "Привет! Я помогу найти похожее аниме. Выберите действие или используйте /help.".into(),
        vec![vec![
            ("Рекомендации".into(), Action::Recommend),
            ("Оценить аниме".into(), Action::Rate),
            ("Отзыв".into(), Action::Feedback),
        ]],
    )
    .await?;
    Ok(())
}

pub async fn cancel(bot: &Bot, actor: Actor, session: &mut Session) -> HandlerResult {
    bot.send_message(
        ChatId(actor.chat_id),
        "Действие отменено. Начните заново: /start",
    )
    .await?;
    session.reset();
    Ok(())
}

pub async fn route(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    user: &User,
    session: &mut Session,
    text: &str,
) -> HandlerResult {
    let command = text.split_whitespace().next().unwrap_or("");
    match command.to_ascii_lowercase().as_str() {
        "/start" => start(bot, ctx, actor, user, session).await,
        "/help" => {
            bot.send_message(ChatId(actor.chat_id), HELP).await?;
            Ok(())
        }
        "/cancel" => cancel(bot, actor, session).await,
        "/recommend" => search::begin(bot, ctx, actor, session, AnimeIntent::Recommend).await,
        "/rate" => search::begin(bot, ctx, actor, session, AnimeIntent::Rate).await,
        "/feedback" => feedback::begin(bot, ctx, actor, session).await,
        _ => {
            bot.send_message(
                ChatId(actor.chat_id),
                "Неизвестная команда. Посмотрите /help.",
            )
            .await?;
            Ok(())
        }
    }
}
