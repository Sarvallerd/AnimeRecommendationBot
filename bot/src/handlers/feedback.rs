use super::HandlerResult;
use crate::dialogue::{
    context::AppContext,
    state::{Actor, State},
    storage::Session,
};
use teloxide::prelude::*;

pub async fn begin(
    bot: &Bot,
    _ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
) -> HandlerResult {
    bot.send_message(
        ChatId(actor.chat_id),
        "Отзывы пока недоступны. Попробуйте позже.",
    )
    .await?;
    session.begin(State::Idle);
    Ok(())
}
pub async fn on_message(
    bot: &Bot,
    _ctx: &AppContext,
    actor: Actor,
    _session: &mut Session,
    _text: Option<&str>,
    _message_id: i32,
) -> HandlerResult {
    bot.send_message(
        ChatId(actor.chat_id),
        "Отзывы пока недоступны. Попробуйте позже.",
    )
    .await?;
    Ok(())
}
