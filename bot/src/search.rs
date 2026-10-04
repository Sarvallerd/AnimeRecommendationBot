//! Immutable Unicode title index. Results always refer to catalog MAL IDs.
use crate::catalog::{Catalog, MalId};
use std::collections::{BTreeMap, HashSet};
use unicode_normalization::UnicodeNormalization;

pub const MAX_QUERY_CHARS: usize = 256;
pub const MAX_CANDIDATES: usize = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryError {
    Empty,
    TooLong,
    InvalidText,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchMatches {
    pub candidates: Vec<MalId>,
    pub has_more: bool,
    display_titles: BTreeMap<MalId, String>,
}

impl SearchMatches {
    pub fn display_title(&self, mal_id: MalId) -> Option<&str> {
        self.display_titles.get(&mal_id).map(String::as_str)
    }
}

struct TitleVariant {
    mal_id: MalId,
    normalized: String,
    char_count: usize,
    display_title: String,
}

pub struct SearchIndex {
    variants: Vec<TitleVariant>,
}

type MatchScore = (u8, usize, usize, MalId);
type RankedVariant = (MatchScore, usize);

impl SearchIndex {
    pub fn new(catalog: &Catalog) -> Self {
        let mut variants = Vec::new();
        for (id, anime) in catalog.iter() {
            let mut seen = HashSet::new();
            for title in std::iter::once(&anime.title).chain(anime.aliases.iter()) {
                let normalized = normalize(title);
                if !normalized.is_empty() && seen.insert(normalized.clone()) {
                    let len = normalized.chars().count();
                    variants.push(TitleVariant {
                        mal_id: id,
                        normalized,
                        char_count: len,
                        display_title: title.clone(),
                    });
                }
            }
        }
        Self { variants }
    }

    pub fn find(&self, raw_query: &str) -> Result<SearchMatches, QueryError> {
        if raw_query.contains('\0') {
            return Err(QueryError::InvalidText);
        }
        if raw_query.chars().count() > MAX_QUERY_CHARS {
            return Err(QueryError::TooLong);
        }
        let query = normalize(raw_query);
        let query_len = query.chars().count();
        if query_len > MAX_QUERY_CHARS {
            return Err(QueryError::TooLong);
        }
        if query.is_empty() {
            return Err(QueryError::Empty);
        }
        // Variants are grouped by numeric ID. Finalize each ID's best variant before
        // admitting it to the bounded result list.
        let mut top = Vec::<RankedVariant>::new();
        let mut matching_ids = 0;
        let mut current_id = None;
        let mut best_score = None;
        for (variant_index, variant) in self.variants.iter().enumerate() {
            if current_id != Some(variant.mal_id) {
                if let Some(score) = best_score.take() {
                    admit(&mut top, score);
                    matching_ids += 1;
                }
                current_id = Some(variant.mal_id);
            }
            let rank = if variant.normalized == query {
                Some((0, 0))
            } else if query_len < 3 {
                None
            } else if variant.normalized.starts_with(&query) {
                Some((1, 0))
            } else if variant.normalized.contains(&query) {
                Some((2, 0))
            } else {
                let limit = if query_len <= 5 { 1 } else { 2 };
                bounded_distance(
                    &query,
                    &variant.normalized,
                    query_len,
                    variant.char_count,
                    limit,
                )
                .map(|d| (3, d))
            };
            if let Some((class, distance)) = rank {
                let score = (
                    class,
                    distance,
                    variant.char_count.saturating_sub(query_len),
                    variant.mal_id,
                );
                let score = (score, variant_index);
                best_score = Some(best_score.map_or(score, |old| old.min(score)));
            }
        }
        if let Some(score) = best_score {
            admit(&mut top, score);
            matching_ids += 1;
        }
        let mut candidates = Vec::with_capacity(top.len());
        let mut display_titles = BTreeMap::new();
        for (score, variant_index) in top {
            let mal_id = score.3;
            candidates.push(mal_id);
            display_titles.insert(mal_id, self.variants[variant_index].display_title.clone());
        }
        Ok(SearchMatches {
            candidates,
            has_more: matching_ids > MAX_CANDIDATES,
            display_titles,
        })
    }
}

fn admit(top: &mut Vec<RankedVariant>, score: RankedVariant) {
    top.push(score);
    top.sort_unstable();
    top.truncate(MAX_CANDIDATES);
}

pub(crate) fn normalize(input: &str) -> String {
    let mut result = String::new();
    let mut separator = false;
    for ch in input.nfkc().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() {
            if separator && !result.is_empty() {
                result.push(' ');
            }
            result.push(ch);
            separator = false;
        } else {
            separator = true;
        }
    }
    result
}

fn bounded_distance(a: &str, b: &str, a_len: usize, b_len: usize, limit: usize) -> Option<usize> {
    if a_len.abs_diff(b_len) > limit {
        return None;
    }
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut row = vec![limit + 1; b_len + 1];
    for (i, left) in a.iter().enumerate() {
        row.fill(limit + 1);
        row[0] = i + 1;
        let start = (i + 1).saturating_sub(limit).max(1);
        let end = (i + 1 + limit).min(b_len);
        let mut min = limit + 1;
        for j in start..=end {
            row[j] = (prev[j] + 1)
                .min(row[j - 1] + 1)
                .min(prev[j - 1] + usize::from(*left != b[j - 1]));
            min = min.min(row[j]);
        }
        if min > limit {
            return None;
        }
        std::mem::swap(&mut prev, &mut row);
    }
    (prev[b_len] <= limit).then_some(prev[b_len])
}
