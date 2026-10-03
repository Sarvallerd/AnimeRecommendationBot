mod telegram;
pub use telegram::*;

use bot::{
    catalog::Bundle,
    dialogue::{context::AppContext, repository::Repository, storage::SessionStore},
};
use std::sync::Arc;

#[allow(dead_code)]
pub fn fixture(repo: Arc<dyn Repository>, sessions: Arc<SessionStore>) -> Arc<AppContext> {
    let bundle = Bundle::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bundle"
    ))
    .unwrap();
    Arc::new(AppContext::new(Arc::new(bundle), repo, sessions))
}
