pub mod commands;
pub mod feedback;
pub mod ratings;
pub mod recommendations;
pub mod search;
mod synopsis;
mod titles;
pub mod ui;

use crate::{
    db::DbError,
    dialogue::{
        callback::Action,
        context::AppContext,
        state::{Actor, AnimeIntent, ResolvedSelection, State},
    },
};
use std::sync::Arc;
use teloxide::{
    dispatching::UpdateHandler,
    prelude::*,
    types::{CallbackQuery, Message, Update},
};

type HandlerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

pub fn schema() -> UpdateHandler<Box<dyn std::error::Error + Send + Sync>> {
    dptree::entry()
        .branch(Update::filter_message().endpoint(handle_message))
        .branch(Update::filter_callback_query().endpoint(handle_callback))
}

fn actor(chat_id: i64, user_id: u64) -> Option<Actor> {
    let user_id = i64::try_from(user_id).ok().filter(|id| *id > 0)?;
    Some(Actor { chat_id, user_id })
}

async fn handle_message(bot: Bot, msg: Message, ctx: Arc<AppContext>) -> HandlerResult {
    let text = msg.text();
    if !msg.chat.is_private() {
        if text.is_some_and(|t| t.starts_with('/')) {
            bot.send_message(msg.chat.id, "Напишите мне в личном чате.")
                .await?;
        }
        return Ok(());
    }
    let Some(user) = msg.from.as_ref() else {
        return Ok(());
    };
    let Some(actor) = actor(msg.chat.id.0, user.id.0) else {
        return Ok(());
    };
    let session = ctx.sessions.get(actor);
    let mut session = session.lock().await;
    if let Some(pending) = session.pending_feedback.as_ref() {
        let incoming_key = format!("msg:{}:{}", actor.chat_id, msg.id.0);
        if pending.action_key == incoming_key {
            if text != Some(pending.body.as_str()) {
                return feedback::reject_changed(&bot, actor).await;
            }
            if !matches!(session.state, State::AwaitingFeedback) {
                return Ok(());
            }
        }
    }
    if let Some(command) = text.filter(|t| t.starts_with('/')) {
        return commands::route(&bot, &ctx, actor, user, &mut session, command).await;
    }
    match session.state.clone() {
        State::AwaitingQuery { .. } | State::PendingQuery { .. } | State::ChoosingAnime { .. } => {
            search::on_message(&bot, &ctx, actor, user, &mut session, text, msg.id.0).await
        }
        State::AwaitingFeedback => {
            feedback::on_message(&bot, &ctx, actor, user, &mut session, text, msg.id.0).await
        }
        _ => {
            bot.send_message(
                msg.chat.id,
                "Выберите действие: /start. Список команд: /help.",
            )
            .await?;
            Ok(())
        }
    }
}

async fn handle_callback(bot: Bot, q: CallbackQuery, ctx: Arc<AppContext>) -> HandlerResult {
    // Telegram should stop showing the spinner before we acquire a session lock or call the database.
    let mut acknowledgment = bot.answer_callback_query(q.id.clone());
    if q.message
        .as_ref()
        .is_some_and(|message| !message.chat().is_private())
    {
        acknowledgment = acknowledgment.text("Напишите мне в личном чате.");
    }
    if acknowledgment.await.is_err() {
        return Ok(());
    }
    let Some(message) = q.regular_message() else {
        return Ok(());
    };
    if !message.chat.is_private() {
        return Ok(());
    }
    let Some(actor) = actor(message.chat.id.0, q.from.id.0) else {
        return Ok(());
    };
    let Some(data) = q.data.as_deref() else {
        stale(&bot, actor).await?;
        return Ok(());
    };
    let Some(session) = ctx.sessions.existing(actor) else {
        stale(&bot, actor).await?;
        return Ok(());
    };
    let mut session = session.lock().await;
    let Some((action, action_key)) = session.claim(data, message.id.0) else {
        if let (
            Some(record),
            State::ChoosingAnime {
                selected_mal_id: Some(selected),
                ..
            },
        ) = (session.callbacks.get(data), &session.state)
        {
            if record.generation == session.generation
                && record.message_id == Some(message.id.0)
                && matches!(record.action, Action::Select { mal_id, .. } if mal_id != *selected)
            {
                bot.send_message(message.chat.id, "Для этого запроса уже выбран вариант. Повторите выбранную кнопку или отправьте название новым сообщением.").await?;
                return Ok(());
            }
        }
        stale(&bot, actor).await?;
        return Ok(());
    };
    let result = match action {
        Action::Recommend => {
            search::begin(&bot, &ctx, actor, &mut session, AnimeIntent::Recommend).await
        }
        Action::Rate => search::begin(&bot, &ctx, actor, &mut session, AnimeIntent::Rate).await,
        Action::Feedback => feedback::begin(&bot, &ctx, actor, &mut session).await,
        Action::Cancel => commands::cancel(&bot, actor, &mut session).await,
        Action::RetryQuery { .. } => search::on_retry(&bot, &ctx, actor, &mut session).await,
        Action::RetryFeedback { .. } => feedback::on_retry(&bot, &ctx, actor, &mut session).await,
        Action::Select {
            intent,
            request_id,
            mal_id,
        } => {
            let valid = if let State::ChoosingAnime {
                query,
                candidates,
                selected_mal_id,
                ..
            } = &mut session.state
            {
                if query.request_id == request_id
                    && candidates.contains(&mal_id)
                    && selected_mal_id.is_none_or(|selected| selected == mal_id)
                {
                    *selected_mal_id = Some(mal_id);
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if !valid {
                session.finish(data, false);
                return stale(&bot, actor).await;
            }
            match ctx
                .repository
                .resolve_request(actor.user_id, request_id, mal_id, ctx.bundle.identity())
                .await
            {
                Ok(_) => {
                    let Some(query) = (match &session.state {
                        State::ChoosingAnime { query, .. } => Some(query.clone()),
                        _ => None,
                    }) else {
                        session.finish(data, false);
                        return stale(&bot, actor).await;
                    };
                    let selection = ResolvedSelection {
                        query,
                        seed_mal_id: mal_id,
                        bundle_id: ctx.bundle.identity().to_owned(),
                    };
                    search::on_select(&bot, &ctx, actor, &mut session, intent, selection).await
                }
                Err(DbError::Conflict | DbError::NotFound | DbError::InvalidInput(_)) => {
                    session.reset();
                    bot.send_message(message.chat.id, "Выбор устарел. Начните заново: /start")
                        .await
                        .map(|_| ())
                        .map_err(Into::into)
                }
                Err(DbError::DatabaseFailure) => {
                    match bot
                        .send_message(
                            message.chat.id,
                            "Не удалось подтвердить сохранение. Попробуйте ещё раз.",
                        )
                        .await
                    {
                        Ok(_) => Err(DbError::DatabaseFailure.into()),
                        Err(error) => Err(error.into()),
                    }
                }
            }
        }
        Action::AnimeScore { score, .. } => {
            let Some(selection) = (match &session.state {
                State::Selected { selection, .. } => Some(selection.clone()),
                _ => None,
            }) else {
                session.finish(data, false);
                return stale(&bot, actor).await;
            };
            ratings::on_score(
                &bot,
                &ctx,
                actor,
                &mut session,
                &selection,
                score,
                &action_key,
            )
            .await
        }
        Action::RecommendationScore { position_id, score } => {
            let Some(selection) = (match &session.state {
                State::Selected {
                    intent: AnimeIntent::Recommend,
                    selection,
                } => Some(selection.clone()),
                _ => None,
            }) else {
                session.finish(data, false);
                return stale(&bot, actor).await;
            };
            match ctx
                .repository
                .list_delivered_positions(actor.user_id, selection.query.request_id)
                .await
            {
                Ok(positions)
                    if positions.iter().any(|p| {
                        p.id == position_id
                            && p.request_id == selection.query.request_id
                            && p.chat_id == actor.chat_id
                            && p.message_id == message.id.0
                    }) =>
                {
                    let position = positions
                        .iter()
                        .find(|p| {
                            p.id == position_id
                                && p.request_id == selection.query.request_id
                                && p.chat_id == actor.chat_id
                                && p.message_id == message.id.0
                        })
                        .unwrap();
                    recommendations::on_score(
                        &bot,
                        &ctx,
                        actor,
                        &mut session,
                        &selection,
                        recommendations::ScoreAction {
                            position,
                            score,
                            action_key: &action_key,
                        },
                    )
                    .await
                }
                Ok(_) | Err(DbError::Conflict | DbError::NotFound | DbError::InvalidInput(_)) => {
                    stale(&bot, actor).await
                }
                Err(DbError::DatabaseFailure) => match bot
                    .send_message(
                        message.chat.id,
                        "Не удалось подтвердить сохранение. Попробуйте ещё раз.",
                    )
                    .await
                {
                    Ok(_) => Err(DbError::DatabaseFailure.into()),
                    Err(error) => Err(error.into()),
                },
            }
        }
        Action::RecommendationDescription { position_id, view } => {
            let Some(selection) = (match &session.state {
                State::Selected {
                    intent: AnimeIntent::Recommend,
                    selection,
                } => Some(selection.clone()),
                _ => None,
            }) else {
                session.finish(data, false);
                return stale(&bot, actor).await;
            };
            match ctx
                .repository
                .list_delivered_positions(actor.user_id, selection.query.request_id)
                .await
            {
                Ok(positions) => {
                    if let Some(position) = positions.iter().find(|p| {
                        p.id == position_id
                            && p.request_id == selection.query.request_id
                            && p.chat_id == actor.chat_id
                            && p.message_id == message.id.0
                    }) {
                        recommendations::on_description(
                            &bot,
                            &ctx,
                            actor,
                            &mut session,
                            &selection,
                            position,
                            view,
                        )
                        .await
                    } else {
                        stale(&bot, actor).await
                    }
                }
                Err(DbError::DatabaseFailure) => Err(DbError::DatabaseFailure.into()),
                Err(_) => stale(&bot, actor).await,
            }
        }
    };
    session.finish(data, result.is_ok());
    // Transport failures are logged by the dispatcher. The same action key remains available for an explicit retry.
    result
}

async fn stale(bot: &Bot, actor: Actor) -> HandlerResult {
    bot.send_message(ChatId(actor.chat_id), commands::STALE)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn user_id_conversion_is_checked() {
        assert_eq!(actor(1, 0), None);
        assert_eq!(actor(1, u64::MAX), None);
        assert_eq!(
            actor(1, 1),
            Some(Actor {
                chat_id: 1,
                user_id: 1
            })
        );
    }
    #[test]
    fn commands_are_global_and_complete() {
        let help = commands::HELP;
        for cmd in [
            "/start",
            "/help",
            "/cancel",
            "/recommend",
            "/rate",
            "/feedback",
        ] {
            assert!(help.contains(cmd));
        }
    }
}
