use std::sync::Arc;

use crate::{
    catalog::Bundle,
    dialogue::{repository::Repository, storage::SessionStore},
};

#[derive(Clone)]
pub struct AppContext {
    pub bundle: Arc<Bundle>,
    pub repository: Arc<dyn Repository>,
    pub sessions: Arc<SessionStore>,
}
