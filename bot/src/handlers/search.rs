use super::{ratings, recommendations, HandlerResult};
use crate::dialogue::{
    context::AppContext,
    state::{Actor, AnimeIntent, ResolvedSelection, State},
    storage::Session,
};
use teloxide::prelude::*;

pub async fn begin(
    bot: &Bot,
    _ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
    intent: AnimeIntent,
) -> HandlerResult {
    session.begin(State::AwaitingQuery { intent });
    bot.send_message(ChatId(actor.chat_id), "Напишите название аниме.")
        .await?;
    Ok(())
}

pub async fn on_message(
    bot: &Bot,
    _ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
    text: Option<&str>,
    _message_id: i32,
) -> HandlerResult {
    let valid_query = text.is_some_and(|value| !value.trim().is_empty());
    let notice = if valid_query {
        "Поиск аниме пока недоступен. Попробуйте позже или выберите другую команду."
    } else {
        "Пришлите название аниме текстом."
    };
    if valid_query {
        session.reset();
    }
    bot.send_message(ChatId(actor.chat_id), notice).await?;
    Ok(())
}

pub async fn on_select(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
    intent: AnimeIntent,
    selection: ResolvedSelection,
) -> HandlerResult {
    let previous = std::mem::replace(
        &mut session.state,
        State::Selected {
            intent,
            selection: selection.clone(),
        },
    );
    let result = match intent {
        AnimeIntent::Recommend => {
            recommendations::begin(bot, ctx, actor, session, &selection).await
        }
        AnimeIntent::Rate => ratings::begin(bot, ctx, actor, session, &selection).await,
    };
    if result.is_err() {
        session.state = previous;
    }
    result
}
