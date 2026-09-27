use super::{ui, HandlerResult};
use crate::{
    db::{DbError, WriteOutcome},
    dialogue::{
        callback::Action,
        context::AppContext,
        state::{Actor, ResolvedSelection},
        storage::Session,
    },
};
use teloxide::prelude::*;

pub async fn begin(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
    selection: &ResolvedSelection,
) -> HandlerResult {
    let title = title(ctx, selection);
    let request_id = selection.query.request_id;
    let rows = (1..=10)
        .collect::<Vec<i16>>()
        .chunks(5)
        .map(|scores| {
            scores
                .iter()
                .map(|score| {
                    (
                        score.to_string(),
                        Action::AnimeScore {
                            request_id,
                            score: *score,
                        },
                    )
                })
                .collect()
        })
        .collect();
    ui::send_keyboard(
        bot,
        actor,
        session,
        format!(
            "Оцените «{title}» · MAL ID {} от 1 до 10.\nЧтобы изменить сохранённую оценку, начните новый поиск: /rate.",
            selection.seed_mal_id
        ),
        rows,
    )
    .await?;
    Ok(())
}

pub async fn on_score(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
    selection: &ResolvedSelection,
    score: i16,
    action_key: &str,
) -> HandlerResult {
    let request_id = selection.query.request_id;
    // Keep only the claimed score for this request. A failed or uncertain write
    // can be retried with exactly the same button after the router reactivates it.
    session.callbacks.retain(|_, record| {
        !matches!(&record.action, Action::AnimeScore { request_id: id, .. } if *id == request_id)
            || record.action_key == action_key
                && matches!(&record.action, Action::AnimeScore { score: s, .. } if *s == score)
    });

    match ctx
        .repository
        .rate_anime(actor.user_id, request_id, score)
        .await
    {
        Ok(WriteOutcome::Created(_) | WriteOutcome::AlreadyRecorded(_)) => {
            bot.send_message(
                ChatId(actor.chat_id),
                format!(
                    "Оценка {score}/10 для «{}» · MAL ID {} сохранена. Чтобы изменить оценку, начните заново: /rate.",
                    title(ctx, selection),
                    selection.seed_mal_id
                ),
            )
            .await?;
            session.reset();
            Ok(())
        }
        Err(DbError::DatabaseFailure) => {
            bot.send_message(
                ChatId(actor.chat_id),
                format!(
                    "Не удалось подтвердить сохранение оценки. Повторите ту же кнопку «{score}»."
                ),
            )
            .await?;
            Err(DbError::DatabaseFailure.into())
        }
        Err(DbError::Conflict | DbError::NotFound | DbError::InvalidInput(_)) => {
            bot.send_message(
                ChatId(actor.chat_id),
                "Не удалось подтвердить эту оценку. Начните новый поиск: /rate.",
            )
            .await?;
            session.reset();
            Ok(())
        }
    }
}

fn title(ctx: &AppContext, selection: &ResolvedSelection) -> String {
    let anime = ctx
        .bundle
        .catalog()
        .get(selection.seed_mal_id)
        .expect("selected MAL ID belongs to loaded bundle");
    ui::bounded(&anime.title, 240)
}
