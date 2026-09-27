mod config;
use bot::{
    catalog::Bundle,
    db::Db,
    dialogue::{context::AppContext, storage::SessionStore},
    handlers,
};
use config::Config;
use std::{sync::Arc, time::Duration};
use teloxide::prelude::*;

#[tokio::main]
async fn main() {
    if let Err(message) = run().await {
        eprintln!("{message}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let check_config = match (args.next(), args.next()) {
        (None, None) => false,
        (Some(arg), None) if arg == "--check-config" => true,
        _ => return Err("Usage: bot [--check-config]".to_owned()),
    };
    let config = Config::from_env().map_err(|error| error.to_string())?;
    if check_config {
        println!("Configuration is valid.");
        return Ok(());
    }
    config.init_logging().map_err(|error| error.to_string())?;
    log::info!("Starting anime recommendation bot");
    let bundle = Arc::new(
        Bundle::load(config.artifacts_dir())
            .map_err(|error| format!("Recommendation bundle failed to load: {error}"))?,
    );
    log::info!(
        "Loaded recommendation bundle {} ({} records)",
        bundle.identity(),
        bundle.catalog().len()
    );
    let bot = Bot::new(config.token());
    let db = Db::new(config.database())
        .await
        .map_err(|_| "Database connection failed.".to_owned())?;
    db.migrate()
        .await
        .map_err(|_| "Database setup failed.".to_owned())?;
    let context = Arc::new(AppContext::new(
        bundle,
        Arc::new(db),
        Arc::new(SessionStore::new()),
    ));
    let webhook = bot
        .get_webhook_info()
        .await
        .map_err(|_| "Telegram webhook check failed.".to_owned())?;
    if webhook.url.is_some() {
        bot.delete_webhook()
            .await
            .map_err(|_| "Telegram webhook removal failed.".to_owned())?;
    }
    let listener = teloxide::update_listeners::Polling::builder(bot.clone())
        .timeout(Duration::from_secs(10))
        .build();
    let mut dispatcher = Dispatcher::builder(bot, handlers::schema())
        .error_handler(Arc::new(
            |_error: Box<dyn std::error::Error + Send + Sync>| async {
                log::warn!("Bot update failed; details omitted from logs");
            },
        ))
        .dependencies(dptree::deps![context])
        .enable_ctrlc_handler()
        .build();
    dispatcher
        .try_dispatch_with_listener(
            listener,
            Arc::new(|_error: teloxide::RequestError| async {
                log::warn!("Telegram polling failed; details omitted from logs");
            }),
        )
        .await
        .map_err(|_| "Telegram startup failed.".to_owned())?;
    Ok(())
}
