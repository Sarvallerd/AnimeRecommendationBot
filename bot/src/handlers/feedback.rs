use super::{ui, HandlerResult};
use crate::{
    db::{DbError, UserProfile, WriteOutcome},
    dialogue::{
        callback::Action,
        context::AppContext,
        state::{Actor, State},
        storage::{CallbackStatus, PendingFeedback, Session},
    },
};
use teloxide::{prelude::*, types::User};

const SAVED: &str = "Спасибо! Ваш отзыв сохранён.";
const INVALID: &str = "Пришлите отзыв текстом новым сообщением.";
const PENDING: &str =
    "Сначала завершите предыдущий отзыв: повторите его кнопкой или откройте /feedback.";
const CHANGED: &str = "Это сообщение уже использовано для другого отзыва. Повторите прежний отзыв.";
const TERMINAL: &str = "Не удалось подтвердить этот отзыв. Отправьте отзыв новым сообщением.";

pub async fn begin(
    bot: &Bot,
    _ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
) -> HandlerResult {
    if session.pending_feedback.is_some() {
        session.begin(State::AwaitingFeedback);
        offer_retry(bot, actor, session).await
    } else {
        bot.send_message(
            ChatId(actor.chat_id),
            "Напишите отзыв одним текстовым сообщением.",
        )
        .await?;
        session.begin(State::AwaitingFeedback);
        Ok(())
    }
}

pub async fn on_message(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    user: &User,
    session: &mut Session,
    text: Option<&str>,
    message_id: i32,
) -> HandlerResult {
    let action_key = format!("msg:{}:{message_id}", actor.chat_id);
    if let Some(pending) = session.pending_feedback.as_ref() {
        if pending.actor == actor && pending.action_key == action_key && text == Some(&pending.body)
        {
            return run_pending(bot, ctx, actor, session).await;
        }
        bot.send_message(
            ChatId(actor.chat_id),
            if pending.action_key == action_key {
                CHANGED
            } else {
                PENDING
            },
        )
        .await?;
        return Ok(());
    }
    let Some(body) = text else {
        bot.send_message(ChatId(actor.chat_id), INVALID).await?;
        return Ok(());
    };
    if message_id <= 0
        || body.trim().is_empty()
        || body.contains('\0')
        || actor.user_id != i64::try_from(user.id.0).unwrap_or_default()
    {
        bot.send_message(ChatId(actor.chat_id), INVALID).await?;
        return Ok(());
    }
    session.pending_feedback = Some(PendingFeedback {
        actor,
        profile: UserProfile {
            tg_id: actor.user_id,
            first_name: user.first_name.clone(),
            language_code: user.language_code.clone(),
            last_name: user.last_name.clone(),
            username: user.username.clone(),
        },
        action_key,
        body: body.to_owned(),
        saved: false,
    });
    run_pending(bot, ctx, actor, session).await
}

pub async fn on_retry(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
) -> HandlerResult {
    run_pending(bot, ctx, actor, session).await
}

async fn run_pending(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
) -> HandlerResult {
    let Some(pending) = session.pending_feedback.as_ref() else {
        return Ok(());
    };
    if pending.actor != actor || !matches!(session.state, State::AwaitingFeedback) {
        return Err(DbError::Conflict.into());
    }
    if !pending.saved {
        let profile = pending.profile.clone();
        let action_key = pending.action_key.clone();
        let body = pending.body.clone();
        if let Err(error) = ctx.repository.upsert_user(&profile).await {
            return failed_write(bot, actor, session, error).await;
        }
        match ctx
            .repository
            .save_feedback(actor.user_id, &action_key, &body)
            .await
        {
            Ok(WriteOutcome::Created(_) | WriteOutcome::AlreadyRecorded(_)) => {
                session
                    .pending_feedback
                    .as_mut()
                    .expect("retained feedback")
                    .saved = true;
            }
            Err(error) => return failed_write(bot, actor, session, error).await,
        }
    }
    match bot.send_message(ChatId(actor.chat_id), SAVED).await {
        Ok(_) => {
            session.pending_feedback = None;
            session.reset();
            Ok(())
        }
        Err(error) => {
            let _ = offer_retry(bot, actor, session).await;
            Err(error.into())
        }
    }
}

async fn failed_write(
    bot: &Bot,
    actor: Actor,
    session: &mut Session,
    error: DbError,
) -> HandlerResult {
    match error {
        DbError::DatabaseFailure => {
            let _ = offer_retry(bot, actor, session).await;
            Err(error.into())
        }
        DbError::Conflict | DbError::NotFound | DbError::InvalidInput(_) => {
            bot.send_message(ChatId(actor.chat_id), TERMINAL).await?;
            session.pending_feedback = None;
            session.begin(State::AwaitingFeedback);
            Ok(())
        }
    }
}

async fn offer_retry(bot: &Bot, actor: Actor, session: &mut Session) -> HandlerResult {
    let Some(pending) = session.pending_feedback.as_ref() else {
        return Ok(());
    };
    let action_key = pending.action_key.clone();
    if session.callbacks.values().any(|record| {
        matches!(&record.action, Action::RetryFeedback { action_key: key } if key == &action_key)
            && matches!(
                record.status,
                CallbackStatus::Active | CallbackStatus::Processing
            )
    }) {
        bot.send_message(ChatId(actor.chat_id), "Повторите отзыв той же кнопкой.")
            .await?;
        return Ok(());
    }
    ui::send_keyboard(
        bot,
        actor,
        session,
        "Не удалось подтвердить сохранение отзыва. Попробуйте ещё раз.".into(),
        vec![vec![(
            "Повторить отзыв".into(),
            Action::RetryFeedback { action_key },
        )]],
    )
    .await?;
    Ok(())
}

pub(crate) async fn reject_changed(bot: &Bot, actor: Actor) -> HandlerResult {
    bot.send_message(ChatId(actor.chat_id), CHANGED).await?;
    Ok(())
}
