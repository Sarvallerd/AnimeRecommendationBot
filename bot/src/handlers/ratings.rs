use super::HandlerResult;
use crate::dialogue::{
    context::AppContext,
    state::{Actor, ResolvedSelection},
    storage::Session,
};
use teloxide::prelude::*;

pub async fn begin(
    bot: &Bot,
    _ctx: &AppContext,
    actor: Actor,
    _session: &mut Session,
    _selection: &ResolvedSelection,
) -> HandlerResult {
    bot.send_message(
        ChatId(actor.chat_id),
        "Оценка аниме пока недоступна. Попробуйте позже.",
    )
    .await?;
    Ok(())
}
pub async fn on_score(
    bot: &Bot,
    _ctx: &AppContext,
    actor: Actor,
    _session: &mut Session,
    _selection: &ResolvedSelection,
    _score: i16,
    _action_key: &str,
) -> HandlerResult {
    bot.send_message(
        ChatId(actor.chat_id),
        "Оценка аниме пока недоступна. Попробуйте позже.",
    )
    .await?;
    Ok(())
}
