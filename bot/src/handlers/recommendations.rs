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
        "Рекомендации пока недоступны. Попробуйте позже.",
    )
    .await?;
    Ok(())
}

pub struct ScoreAction<'a> {
    pub position_id: crate::db::PositionId,
    pub score: i16,
    pub action_key: &'a str,
}

pub async fn on_score(
    bot: &Bot,
    _ctx: &AppContext,
    actor: Actor,
    _session: &mut Session,
    _selection: &ResolvedSelection,
    _action: ScoreAction<'_>,
) -> HandlerResult {
    bot.send_message(
        ChatId(actor.chat_id),
        "Оценка рекомендации пока недоступна. Попробуйте позже.",
    )
    .await?;
    Ok(())
}
