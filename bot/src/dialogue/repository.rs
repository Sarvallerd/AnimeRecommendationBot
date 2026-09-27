use crate::db::{
    AnimeRatingEventId, Db, DbError, DeliveredPosition, DeliveryInput, PositionId, RequestId,
    UserProfile, WriteOutcome,
};
use std::{future::Future, pin::Pin};

pub type DbFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, DbError>> + Send + 'a>>;

pub trait Repository: Send + Sync {
    fn upsert_user<'a>(&'a self, profile: &'a UserProfile) -> DbFuture<'a, ()>;
    fn record_query<'a>(
        &'a self,
        user_id: i64,
        key: &'a str,
        query: &'a str,
    ) -> DbFuture<'a, WriteOutcome<RequestId>>;
    fn resolve_request<'a>(
        &'a self,
        user_id: i64,
        request_id: RequestId,
        mal_id: i32,
        bundle_id: &'a str,
    ) -> DbFuture<'a, WriteOutcome<()>>;
    fn record_delivery<'a>(
        &'a self,
        user_id: i64,
        request_id: RequestId,
        input: &'a DeliveryInput,
    ) -> DbFuture<'a, WriteOutcome<PositionId>>;
    fn list_delivered_positions<'a>(
        &'a self,
        user_id: i64,
        request_id: RequestId,
    ) -> DbFuture<'a, Vec<DeliveredPosition>>;
    fn rate_anime<'a>(
        &'a self,
        user_id: i64,
        request_id: RequestId,
        score: i16,
    ) -> DbFuture<'a, WriteOutcome<AnimeRatingEventId>>;
    fn rate_recommendation<'a>(
        &'a self,
        user_id: i64,
        position_id: PositionId,
        score: i16,
    ) -> DbFuture<'a, WriteOutcome<()>>;
    fn save_feedback<'a>(
        &'a self,
        user_id: i64,
        key: &'a str,
        body: &'a str,
    ) -> DbFuture<'a, WriteOutcome<i64>>;
}

impl Repository for Db {
    fn upsert_user<'a>(&'a self, p: &'a UserProfile) -> DbFuture<'a, ()> {
        Box::pin(async move { Db::upsert_user(self, p).await })
    }
    fn record_query<'a>(
        &'a self,
        u: i64,
        k: &'a str,
        q: &'a str,
    ) -> DbFuture<'a, WriteOutcome<RequestId>> {
        Box::pin(async move { Db::record_query(self, u, k, q).await })
    }
    fn resolve_request<'a>(
        &'a self,
        u: i64,
        r: RequestId,
        m: i32,
        b: &'a str,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        Box::pin(async move { Db::resolve_request(self, u, r, m, b).await })
    }
    fn record_delivery<'a>(
        &'a self,
        u: i64,
        r: RequestId,
        i: &'a DeliveryInput,
    ) -> DbFuture<'a, WriteOutcome<PositionId>> {
        Box::pin(async move { Db::record_delivery(self, u, r, i).await })
    }
    fn list_delivered_positions<'a>(
        &'a self,
        u: i64,
        r: RequestId,
    ) -> DbFuture<'a, Vec<DeliveredPosition>> {
        Box::pin(async move { Db::list_delivered_positions(self, u, r).await })
    }
    fn rate_anime<'a>(
        &'a self,
        u: i64,
        r: RequestId,
        s: i16,
    ) -> DbFuture<'a, WriteOutcome<AnimeRatingEventId>> {
        Box::pin(async move { Db::rate_anime(self, u, r, s).await })
    }
    fn rate_recommendation<'a>(
        &'a self,
        u: i64,
        p: PositionId,
        s: i16,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        Box::pin(async move { Db::rate_recommendation(self, u, p, s).await })
    }
    fn save_feedback<'a>(
        &'a self,
        u: i64,
        k: &'a str,
        b: &'a str,
    ) -> DbFuture<'a, WriteOutcome<i64>> {
        Box::pin(async move { Db::save_feedback(self, u, k, b).await })
    }
}
