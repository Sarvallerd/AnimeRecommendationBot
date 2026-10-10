use super::{synopsis, titles::ResponseTitles, ui, HandlerResult};
use crate::{
    catalog::{Anime, MalId},
    db::{DbError, DeliveredPosition, DeliveryInput, WriteOutcome},
    dialogue::{
        callback::{Action, DescriptionView},
        context::AppContext,
        state::{Actor, AnimeIntent, ResolvedSelection, State},
        storage::{CallbackStatus, PendingRecommendationDelivery, Session},
    },
};
use std::collections::HashSet;
use std::time::Duration;
use teloxide::prelude::*;
use teloxide::types::{
    InlineKeyboardButton, InlineKeyboardMarkup, InputFile, MessageId, ReplyParameters,
};

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
    let titles = ResponseTitles::resolve(
        &ctx.search,
        &selection.query.raw_query,
        selection.seed_mal_id,
        seed,
    );

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
    let mut photos_available = true;
    for (index, neighbor) in neighbors.iter().enumerate() {
        let rank = i16::try_from(index + 1).map_err(|_| DbError::Conflict)?;
        if let Some(row) = recorded.iter().find(|row| row.rank == rank) {
            let anime = ctx
                .bundle
                .catalog()
                .get(row.mal_id)
                .ok_or(DbError::Conflict)?;
            attach_scores(bot, actor, session, row, &titles, neighbors.len(), anime).await?;
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
                render_card(
                    &titles,
                    rank,
                    neighbors.len(),
                    neighbor.mal_id,
                    anime,
                    &DescriptionView::Summary,
                )
                .0,
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
        attach_scores(bot, actor, session, &row, &titles, neighbors.len(), anime).await?;
        if photos_available {
            photos_available = send_cover(bot, ctx, actor, neighbor.mal_id, sent.id).await;
        }
    }
    Ok(())
}

async fn attach_scores(
    bot: &Bot,
    actor: Actor,
    session: &mut Session,
    position: &DeliveredPosition,
    titles: &ResponseTitles,
    total: usize,
    anime: &Anime,
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
    let mut rows = vec![buttons];
    if render_card(
        titles,
        position.rank,
        total,
        position.mal_id,
        anime,
        &DescriptionView::Summary,
    )
    .1
    {
        let (button, token) = description_button(
            session,
            position.id,
            DescriptionView::Page(0),
            "Развернуть описание",
        )?;
        // Read-only tokens are safe to retain after an ambiguous edit outcome.
        session.activate(&[token], position.message_id);
        rows.push(vec![button]);
    }
    let edited = bot
        .edit_message_reply_markup(ChatId(position.chat_id), MessageId(position.message_id))
        .reply_markup(InlineKeyboardMarkup::new(rows))
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

fn render_card(
    titles: &ResponseTitles,
    rank: i16,
    total: usize,
    mal_id: MalId,
    anime: &Anime,
    view: &DescriptionView,
) -> (String, bool) {
    let seed_title = ui::bounded(titles.seed_title(), 160);
    let title = ui::bounded(titles.recommendation_title(anime), 240);
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
    let header = format!(
        "Рекомендация {rank}/{total}\nПо запросу: {seed_title}\nНазвание: {title}\nMAL ID: {mal_id}\nОценка MAL: {score}\nГод: {year}\nТип: {kind}\nЭпизоды: {episodes}\nЖанры: {genres}\nПолезность рекомендации: 0 — не полезна, 5 — очень полезна.\n"
    );
    let description = anime.synopsis.as_deref().unwrap_or("Описание отсутствует.");
    let summary_end = synopsis::preview_end(description);
    let prefix = format!("{header}Описание: ");
    let budget = CARD_LIMIT.saturating_sub(prefix.encode_utf16().count());
    let preview = &description[..summary_end];
    let expandable = anime.synopsis.is_some()
        && (summary_end < description.len() || preview.encode_utf16().count() > budget);
    if matches!(view, DescriptionView::Summary) {
        let mut card = prefix;
        card.push_str(&ui::bounded(preview, budget));
        return (card, expandable);
    }
    // The reserved space covers any plausible decimal page count, keeping page boundaries stable.
    let page_budget = CARD_LIMIT
        .saturating_sub(header.encode_utf16().count() + 80)
        .max(1);
    let pages = synopsis::pages(description, page_budget);
    let DescriptionView::Page(page) = view else {
        unreachable!()
    };
    let Some(payload) = pages.get(*page) else {
        return (String::new(), expandable);
    };
    let mut card = format!(
        "{header}Описание (страница {}/{}):\n",
        page + 1,
        pages.len()
    );
    card.push_str(payload);
    (card, expandable)
}

fn description_button(
    session: &mut Session,
    position_id: i64,
    view: DescriptionView,
    label: &str,
) -> Result<(InlineKeyboardButton, String), getrandom::Error> {
    let token = session.callbacks.iter().find_map(|(token, record)| {
        (record.generation == session.generation
            && matches!(&record.action, Action::RecommendationDescription { position_id: id, view: existing }
                if *id == position_id && *existing == view))
        .then(|| token.clone())
    }).map_or_else(|| session.issue(Action::RecommendationDescription { position_id, view: view.clone() }), Ok)?;
    Ok((InlineKeyboardButton::callback(label, token.clone()), token))
}

pub async fn on_description(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
    selection: &ResolvedSelection,
    position: &DeliveredPosition,
    view: DescriptionView,
) -> HandlerResult {
    if selection.bundle_id != ctx.bundle.identity()
        || position.chat_id != actor.chat_id
        || position.request_id != selection.query.request_id
        || position.message_id <= 0
    {
        return Err(DbError::Conflict.into());
    }
    let seed = ctx
        .bundle
        .catalog()
        .get(selection.seed_mal_id)
        .ok_or(DbError::Conflict)?;
    let anime = ctx
        .bundle
        .catalog()
        .get(position.mal_id)
        .ok_or(DbError::Conflict)?;
    let neighbors = ctx
        .bundle
        .neighbors(selection.seed_mal_id)
        .ok_or(DbError::Conflict)?;
    let index = usize::try_from(position.rank.checked_sub(1).ok_or(DbError::Conflict)?)
        .map_err(|_| DbError::Conflict)?;
    if neighbors.get(index).map(|n| n.mal_id) != Some(position.mal_id) {
        return Err(DbError::Conflict.into());
    }
    let titles = ResponseTitles::resolve(
        &ctx.search,
        &selection.query.raw_query,
        selection.seed_mal_id,
        seed,
    );
    let (text, expandable) = render_card(
        &titles,
        position.rank,
        neighbors.len(),
        position.mal_id,
        anime,
        &view,
    );
    if !expandable || text.is_empty() {
        return Err(DbError::Conflict.into());
    }
    let mut score_buttons: Vec<(i16, InlineKeyboardButton)> = session
        .callbacks
        .iter()
        .filter_map(|(token, record)| {
            if record.generation == session.generation
                && record.message_id == Some(position.message_id)
                && matches!(
                    record.status,
                    CallbackStatus::Active | CallbackStatus::Processing | CallbackStatus::Consumed
                )
            {
                if let Action::RecommendationScore { position_id, score } = record.action {
                    if position_id == position.id {
                        return Some((
                            score,
                            InlineKeyboardButton::callback(score.to_string(), token.clone()),
                        ));
                    }
                }
            }
            None
        })
        .collect();
    score_buttons.sort_by_key(|(score, _)| *score);
    let mut rows: Vec<Vec<InlineKeyboardButton>> = Vec::new();
    if !score_buttons.is_empty() {
        rows.push(
            score_buttons
                .into_iter()
                .map(|(_, button)| button)
                .collect(),
        );
    }
    let mut tokens = Vec::new();
    match view {
        DescriptionView::Summary => {
            let (button, token) = description_button(
                session,
                position.id,
                DescriptionView::Page(0),
                "Развернуть описание",
            )?;
            rows.push(vec![button]);
            tokens.push(token);
        }
        DescriptionView::Page(page) => {
            let mut navigation = Vec::new();
            if page > 0 {
                let (button, token) = description_button(
                    session,
                    position.id,
                    DescriptionView::Page(page - 1),
                    "◀ Назад",
                )?;
                navigation.push(button);
                tokens.push(token);
            }
            if !render_card(
                &titles,
                position.rank,
                neighbors.len(),
                position.mal_id,
                anime,
                &DescriptionView::Page(page + 1),
            )
            .0
            .is_empty()
            {
                let (button, token) = description_button(
                    session,
                    position.id,
                    DescriptionView::Page(page + 1),
                    "Далее ▶",
                )?;
                navigation.push(button);
                tokens.push(token);
            }
            if !navigation.is_empty() {
                rows.push(navigation);
            }
            let (button, token) = description_button(
                session,
                position.id,
                DescriptionView::Summary,
                "Свернуть описание",
            )?;
            rows.push(vec![button]);
            tokens.push(token);
        }
    }
    session.activate(&tokens, position.message_id);
    match bot
        .edit_message_text(
            ChatId(position.chat_id),
            MessageId(position.message_id),
            text,
        )
        .reply_markup(InlineKeyboardMarkup::new(rows))
        .await
    {
        Ok(message)
            if message.chat.id.0 == position.chat_id && message.id.0 == position.message_id =>
        {
            Ok(())
        }
        Ok(_) => Err(DbError::Conflict.into()),
        Err(teloxide::RequestError::Api(teloxide::ApiError::MessageNotModified)) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

async fn send_cover(
    bot: &Bot,
    ctx: &AppContext,
    actor: Actor,
    mal_id: MalId,
    message_id: MessageId,
) -> bool {
    tokio::time::timeout(Duration::from_secs(2), async {
        let Some(url) = ctx.covers.lookup(mal_id).await else {
            return true;
        };
        bot.send_photo(ChatId(actor.chat_id), InputFile::url(url))
            .reply_parameters(ReplyParameters::new(message_id))
            .disable_notification(true)
            .await
            .is_ok()
    })
    .await
    .unwrap_or_default()
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
    selection: &ResolvedSelection,
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
            let seed = ctx.bundle.catalog().get(selection.seed_mal_id);
            let titles = seed.map(|seed| {
                ResponseTitles::resolve(
                    &ctx.search,
                    &selection.query.raw_query,
                    selection.seed_mal_id,
                    seed,
                )
            });
            let title = ctx
                .bundle
                .catalog()
                .get(action.position.mal_id)
                .map(|anime| {
                    ui::bounded(
                        titles.as_ref().map_or(anime.title.as_str(), |titles| {
                            titles.recommendation_title(anime)
                        }),
                        240,
                    )
                })
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
