use super::{ui, HandlerResult};
use crate::{
    catalog::{Anime, MalId},
    db::{DbError, DeliveredPosition, DeliveryInput, WriteOutcome},
    dialogue::{
        callback::Action,
        context::AppContext,
        state::{Actor, AnimeIntent, ResolvedSelection, State},
        storage::{CallbackStatus, PendingRecommendationDelivery, Session},
    },
};
use std::collections::HashSet;
use teloxide::prelude::*;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup, MessageId};

const RETRY_NOTICE: &str = "Не удалось завершить выдачу. Нажмите выбранное аниме ещё раз.";
const EMPTY_NOTICE: &str = "Для этого аниме пока нет рекомендаций. Попробуйте другое: /recommend.";
const CARD_LIMIT: usize = 4000;

pub async fn begin(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
    selection: &ResolvedSelection,
) -> HandlerResult {
    let result = deliver(bot, ctx, actor, session, selection).await;
    if result.is_err() {
        // The selection callback remains retryable when search::on_select restores its state.
        let _ = bot.send_message(ChatId(actor.chat_id), RETRY_NOTICE).await;
    }
    result
}

async fn deliver(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
    selection: &ResolvedSelection,
) -> HandlerResult {
    if !matches!(&session.state,
        State::Selected { intent: AnimeIntent::Recommend, selection: current } if current == selection)
        || selection.bundle_id != ctx.bundle.identity()
    {
        return Err(DbError::Conflict.into());
    }
    let seed = ctx
        .bundle
        .catalog()
        .get(selection.seed_mal_id)
        .ok_or(DbError::Conflict)?;
    let neighbors = ctx
        .bundle
        .neighbors(selection.seed_mal_id)
        .ok_or(DbError::Conflict)?;

    // A confirmed send is always reconciled with its original request and coordinates.
    if let Some(pending) = session.pending_recommendation_delivery.as_ref() {
        if pending.actor != actor
            || (pending.selection.query.request_id == selection.query.request_id
                && pending.selection != *selection)
        {
            return Err(DbError::Conflict.into());
        }
        if pending.selection.query.request_id != selection.query.request_id {
            ctx.repository
                .resolve_request(
                    pending.actor.user_id,
                    pending.selection.query.request_id,
                    pending.selection.seed_mal_id,
                    &pending.selection.bundle_id,
                )
                .await?;
        }
        persist_pending(ctx, session).await?;
    }

    let recorded = ctx
        .repository
        .list_delivered_positions(actor.user_id, selection.query.request_id)
        .await?;
    let mut seen_ranks = HashSet::new();
    let mut seen_targets = HashSet::new();
    for row in &recorded {
        let index = usize::try_from(row.rank.checked_sub(1).ok_or(DbError::Conflict)?)
            .map_err(|_| DbError::Conflict)?;
        if row.id <= 0
            || row.request_id != selection.query.request_id
            || row.chat_id != actor.chat_id
            || row.message_id <= 0
            || neighbors.get(index).map(|n| n.mal_id) != Some(row.mal_id)
            || !seen_ranks.insert(row.rank)
            || !seen_targets.insert(row.mal_id)
        {
            return Err(DbError::Conflict.into());
        }
    }
    if neighbors.is_empty() {
        bot.send_message(ChatId(actor.chat_id), EMPTY_NOTICE)
            .await?;
        return Ok(());
    }
    for (index, neighbor) in neighbors.iter().enumerate() {
        let rank = i16::try_from(index + 1).map_err(|_| DbError::Conflict)?;
        if let Some(row) = recorded.iter().find(|row| row.rank == rank) {
            attach_scores(bot, actor, session, row).await?;
            continue;
        }
        let anime = ctx
            .bundle
            .catalog()
            .get(neighbor.mal_id)
            .ok_or(DbError::Conflict)?;
        let sent = bot
            .send_message(
                ChatId(actor.chat_id),
                render_card(seed, rank, neighbors.len(), neighbor.mal_id, anime),
            )
            .await?;
        if sent.id.0 <= 0 {
            return Err(DbError::Conflict.into());
        }
        session.pending_recommendation_delivery = Some(PendingRecommendationDelivery {
            actor,
            selection: selection.clone(),
            input: DeliveryInput {
                rank,
                mal_id: neighbor.mal_id,
                chat_id: sent.chat.id.0,
                message_id: sent.id.0,
            },
        });
        let row = persist_pending(ctx, session).await?;
        attach_scores(bot, actor, session, &row).await?;
    }
    Ok(())
}

async fn attach_scores(
    bot: &Bot,
    actor: Actor,
    session: &mut Session,
    position: &DeliveredPosition,
) -> HandlerResult {
    if position.chat_id != actor.chat_id || position.id <= 0 || position.message_id <= 0 {
        return Err(DbError::Conflict.into());
    }
    if session.callbacks.values().any(|record| {
        record.generation == session.generation
            && record.message_id == Some(position.message_id)
            && matches!(record.status, CallbackStatus::Active | CallbackStatus::Processing | CallbackStatus::Consumed)
            && matches!(record.action, Action::RecommendationScore { position_id, .. } if position_id == position.id)
    }) {
        return Ok(());
    }
    let mut tokens = Vec::new();
    let mut buttons = Vec::new();
    for score in 0..=5 {
        let token = match session.issue(Action::RecommendationScore {
            position_id: position.id,
            score,
        }) {
            Ok(token) => token,
            Err(error) => {
                session.discard(&tokens);
                return Err(error.into());
            }
        };
        buttons.push(InlineKeyboardButton::callback(
            score.to_string(),
            token.clone(),
        ));
        tokens.push(token);
    }
    let edited = bot
        .edit_message_reply_markup(ChatId(position.chat_id), MessageId(position.message_id))
        .reply_markup(InlineKeyboardMarkup::new(vec![buttons]))
        .await;
    match edited {
        Ok(message)
            if message.chat.id.0 == position.chat_id && message.id.0 == position.message_id =>
        {
            session.activate(&tokens, position.message_id);
            Ok(())
        }
        Ok(_) => {
            session.discard(&tokens);
            Err(DbError::Conflict.into())
        }
        Err(error) => {
            session.discard(&tokens);
            Err(error.into())
        }
    }
}

async fn persist_pending(
    ctx: &AppContext,
    session: &mut Session,
) -> Result<DeliveredPosition, DbError> {
    let pending = session
        .pending_recommendation_delivery
        .as_ref()
        .ok_or(DbError::Conflict)?;
    let result = ctx
        .repository
        .record_delivery(
            pending.actor.user_id,
            pending.selection.query.request_id,
            &pending.input,
        )
        .await?;
    let id = match result {
        WriteOutcome::Created(id) | WriteOutcome::AlreadyRecorded(id) if id > 0 => id,
        _ => return Err(DbError::Conflict),
    };
    let row = DeliveredPosition {
        id,
        request_id: pending.selection.query.request_id,
        rank: pending.input.rank,
        mal_id: pending.input.mal_id,
        chat_id: pending.input.chat_id,
        message_id: pending.input.message_id,
    };
    session.pending_recommendation_delivery = None;
    Ok(row)
}

fn render_card(seed: &Anime, rank: i16, total: usize, mal_id: MalId, anime: &Anime) -> String {
    let seed_title = ui::bounded(&seed.title, 160);
    let title = ui::bounded(&anime.title, 240);
    let score = anime
        .score
        .map_or_else(|| "нет данных".to_owned(), |v| v.to_string());
    let year = anime
        .year
        .map_or_else(|| "нет данных".to_owned(), |v| v.to_string());
    let kind = ui::bounded(anime.anime_type.as_deref().unwrap_or("нет данных"), 32);
    let episodes = ui::bounded(
        anime.episodes.as_ref().map_or("нет данных", |v| v.as_str()),
        32,
    );
    let genres = if anime.genres.is_empty() {
        "нет данных".to_owned()
    } else {
        ui::bounded(&anime.genres.join(", "), 256)
    };
    let mut card = format!(
        "Рекомендация {rank}/{total}\nПо запросу: {seed_title}\nНазвание: {title}\nMAL ID: {mal_id}\nОценка MAL: {score}\nГод: {year}\nТип: {kind}\nЭпизоды: {episodes}\nЖанры: {genres}\nПолезность рекомендации: 0 — не полезна, 5 — очень полезна.\nОписание: "
    );
    let remaining = CARD_LIMIT.saturating_sub(card.encode_utf16().count());
    card.push_str(&ui::bounded(
        anime.synopsis.as_deref().unwrap_or("Описание отсутствует."),
        remaining,
    ));
    card
}

pub struct ScoreAction<'a> {
    pub position: &'a DeliveredPosition,
    pub score: i16,
    pub action_key: &'a str,
}

pub async fn on_score(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
    _selection: &ResolvedSelection,
    action: ScoreAction<'_>,
) -> HandlerResult {
    session.callbacks.retain(|_, record| {
        !matches!(record.action, Action::RecommendationScore { position_id, .. } if position_id == action.position.id)
            || record.action_key == action.action_key
                && matches!(record.action, Action::RecommendationScore { score, .. } if score == action.score)
    });
    match ctx
        .repository
        .rate_recommendation(actor.user_id, action.position.id, action.score)
        .await
    {
        Ok(WriteOutcome::Created(()) | WriteOutcome::AlreadyRecorded(())) => {
            let title = ctx
                .bundle
                .catalog()
                .get(action.position.mal_id)
                .map(|anime| ui::bounded(&anime.title, 240))
                .unwrap_or_else(|| "Аниме".to_owned());
            bot.send_message(
                ChatId(actor.chat_id),
                format!(
                "Оценка полезности рекомендации «{title}» · MAL ID {}: {}/5 сохранена. Спасибо!",
                action.position.mal_id, action.score
            ),
            )
            .await?;
            Ok(())
        }
        Err(DbError::DatabaseFailure) => {
            bot.send_message(
                ChatId(actor.chat_id),
                format!(
                    "Не удалось подтвердить сохранение оценки. Повторите ту же кнопку «{}».",
                    action.score
                ),
            )
            .await?;
            Err(DbError::DatabaseFailure.into())
        }
        Err(DbError::Conflict | DbError::NotFound | DbError::InvalidInput(_)) => {
            bot.send_message(ChatId(actor.chat_id),
                "Не удалось подтвердить эту оценку. Выберите другую рекомендацию или начните новый поиск: /recommend.")
                .await?;
            Ok(())
        }
    }
}
