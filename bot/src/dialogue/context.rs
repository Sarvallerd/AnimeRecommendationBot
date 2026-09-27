use std::sync::Arc;

use crate::{
    catalog::Bundle,
    dialogue::{repository::Repository, storage::SessionStore},
    search::SearchIndex,
};

#[derive(Clone)]
pub struct AppContext {
    pub bundle: Arc<Bundle>,
    pub search: Arc<SearchIndex>,
    pub repository: Arc<dyn Repository>,
    pub sessions: Arc<SessionStore>,
}

impl AppContext {
    pub fn new(
        bundle: Arc<Bundle>,
        repository: Arc<dyn Repository>,
        sessions: Arc<SessionStore>,
    ) -> Self {
        let search = Arc::new(SearchIndex::new(bundle.catalog()));
        Self {
            bundle,
            search,
            repository,
            sessions,
        }
    }
}
