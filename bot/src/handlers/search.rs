use super::{ratings, recommendations, ui, HandlerResult};
use crate::{
    db::{DbError, UserProfile, WriteOutcome},
    dialogue::{
        callback::Action,
        context::AppContext,
        state::{Actor, AnimeIntent, QueryContext, QueryInput, ResolvedSelection, State},
        storage::{CallbackStatus, Session},
    },
    search::QueryError,
};
use teloxide::{prelude::*, types::User};

pub async fn begin(
    bot: &Bot,
    _ctx: &AppContext,
    actor: Actor,
    session: &mut Session,
    intent: AnimeIntent,
) -> HandlerResult {
    bot.send_message(ChatId(actor.chat_id), "Напишите название аниме.")
        .await?;
    session.begin(State::AwaitingQuery { intent });
    Ok(())
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
    let Some(raw_query) = text else {
        bot.send_message(ChatId(actor.chat_id), "Пришлите название аниме текстом.")
            .await?;
        return Ok(());
    };
    let action_key = format!("msg:{}:{message_id}", actor.chat_id);
    if let State::ChoosingAnime { query, .. } = &session.state {
        if query.action_key == action_key {
            if query.raw_query != raw_query {
                bot.send_message(ChatId(actor.chat_id), "Это сообщение уже использовано для другого поиска. Пришлите название новым сообщением.").await?;
            }
            return Ok(());
        }
    }
    if let State::PendingQuery { input, .. } = &session.state {
        if input.action_key == action_key {
            if input.raw_query != raw_query {
                bot.send_message(ChatId(actor.chat_id), "Это сообщение уже использовано для другого поиска. Пришлите название новым сообщением.").await?;
                return Ok(());
            }
            return run_pending(bot, ctx, actor, session).await;
        }
    }
    match ctx.search.find(raw_query) {
        Ok(_) => {}
        Err(error) => {
            let notice = match error {
                QueryError::Empty => "Пришлите название аниме текстом.",
                QueryError::TooLong => "Название слишком длинное. Уточните запрос до 256 символов.",
                QueryError::InvalidText => {
                    "Название содержит недопустимый символ. Пришлите новое название."
                }
            };
            bot.send_message(ChatId(actor.chat_id), notice).await?;
            return Ok(());
        }
    }
    let Some(intent) = (match &session.state {
        State::AwaitingQuery { intent }
        | State::PendingQuery { intent, .. }
        | State::ChoosingAnime { intent, .. } => Some(*intent),
        _ => None,
    }) else {
        return Ok(());
    };
    if actor.user_id != i64::try_from(user.id.0).unwrap_or_default() {
        return Ok(());
    }
    let input = QueryInput {
        raw_query: raw_query.to_owned(),
        action_key,
        profile: UserProfile {
            tg_id: actor.user_id,
            first_name: user.first_name.clone(),
            language_code: user.language_code.clone(),
            last_name: user.last_name.clone(),
            username: user.username.clone(),
        },
    };
    session.begin(State::PendingQuery { intent, input });
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
    let State::PendingQuery { intent, input } = &session.state else {
        return Ok(());
    };
    let intent = *intent;
    let input = input.clone();
    let matches = ctx
        .search
        .find(&input.raw_query)
        .expect("pending query was validated");
    if let Err(error) = ctx.repository.upsert_user(&input.profile).await {
        return failed_write(bot, actor, session, error).await;
    }
    let request_id = match ctx
        .repository
        .record_query(actor.user_id, &input.action_key, &input.raw_query)
        .await
    {
        Ok(WriteOutcome::Created(id) | WriteOutcome::AlreadyRecorded(id)) => id,
        Err(error) => return failed_write(bot, actor, session, error).await,
    };
    if matches.candidates.is_empty() {
        bot.send_message(
            ChatId(actor.chat_id),
            "Совпадений нет. Попробуйте английское, romaji или японское название.",
        )
        .await?;
        session.begin(State::AwaitingQuery { intent });
        return Ok(());
    }
    let mut lines = vec!["Подтвердите аниме:".to_owned()];
    let mut rows = Vec::new();
    for (index, mal_id) in matches.candidates.iter().enumerate() {
        let anime = ctx
            .bundle
            .catalog()
            .get(*mal_id)
            .expect("indexed catalog ID");
        let display_title = matches.display_title(*mal_id).unwrap_or(&anime.title);
        let title = ui::bounded(display_title, 240);
        let kind = ui::bounded(anime.anime_type.as_deref().unwrap_or("—"), 32);
        let year = anime
            .year
            .map_or_else(|| "—".to_owned(), |year| year.to_string());
        lines.push(format!(
            "{}. {} ({year}, {kind}) · MAL ID {mal_id}",
            index + 1,
            title
        ));
        rows.push(vec![(
            format!(
                "{}. {} · MAL ID {mal_id}",
                index + 1,
                ui::bounded(display_title, 50)
            ),
            Action::Select {
                intent,
                request_id,
                mal_id: *mal_id,
            },
        )]);
    }
    if matches.has_more {
        lines.push(
            "Показаны первые 5 совпадений. Если нужного аниме нет, уточните название.".to_owned(),
        );
    }
    let query = QueryContext {
        request_id,
        raw_query: input.raw_query,
        action_key: input.action_key,
    };
    if let Err(error) = ui::send_keyboard(bot, actor, session, lines.join("\n"), rows).await {
        // Candidate tokens were discarded. The pending payload remains retryable.
        let _ = offer_retry(bot, actor, session).await;
        return Err(error);
    }
    session.state = State::ChoosingAnime {
        intent,
        query,
        candidates: matches.candidates,
        selected_mal_id: None,
    };
    Ok(())
}

async fn failed_write(
    bot: &Bot,
    actor: Actor,
    session: &mut Session,
    error: DbError,
) -> HandlerResult {
    match error {
        DbError::DatabaseFailure => {
            offer_retry(bot, actor, session).await?;
            Err(error.into())
        }
        DbError::Conflict | DbError::NotFound | DbError::InvalidInput(_) => {
            let intent = match &session.state {
                State::PendingQuery { intent, .. } => *intent,
                _ => return Ok(()),
            };
            bot.send_message(
                ChatId(actor.chat_id),
                "Не удалось сохранить этот поиск. Пришлите название новым сообщением.",
            )
            .await?;
            session.begin(State::AwaitingQuery { intent });
            Ok(())
        }
    }
}

async fn offer_retry(bot: &Bot, actor: Actor, session: &mut Session) -> HandlerResult {
    let State::PendingQuery { input, .. } = &session.state else {
        return Ok(());
    };
    let action_key = input.action_key.clone();
    // A failed callback keeps its existing token active. Do not accumulate retry buttons.
    if session.callbacks.values().any(|record| {
        matches!(&record.action,
        Action::RetryQuery { action_key: key } if key == &action_key)
            && matches!(
                record.status,
                CallbackStatus::Active | CallbackStatus::Processing
            )
    }) {
        bot.send_message(
            ChatId(actor.chat_id),
            "Не удалось подтвердить сохранение. Повторите поиск той же кнопкой.",
        )
        .await?;
        return Ok(());
    }
    ui::send_keyboard(
        bot,
        actor,
        session,
        "Не удалось подтвердить сохранение поиска. Попробуйте ещё раз.".into(),
        vec![vec![(
            "Повторить поиск".into(),
            Action::RetryQuery { action_key },
        )]],
    )
    .await?;
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
