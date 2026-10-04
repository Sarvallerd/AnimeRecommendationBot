use crate::{
    catalog::{Anime, MalId},
    search::{normalize, SearchIndex},
};
use unicode_normalization::UnicodeNormalization;

pub(super) struct ResponseTitles {
    seed_title: String,
    prefer_english: bool,
}

impl ResponseTitles {
    pub(super) fn resolve(
        search: &SearchIndex,
        raw_query: &str,
        seed_mal_id: MalId,
        seed: &Anime,
    ) -> Self {
        let matched = search
            .find(raw_query)
            .ok()
            .and_then(|matches| matches.display_title(seed_mal_id).map(str::to_owned));
        let preferred = preferred_title(seed);
        let prefer_english = matched.as_deref().is_some_and(|title| {
            latin_eligible(preferred) && normalize(title) == normalize(preferred)
        });
        Self {
            seed_title: matched.unwrap_or_else(|| seed.title.clone()),
            prefer_english,
        }
    }

    pub(super) fn seed_title(&self) -> &str {
        &self.seed_title
    }

    pub(super) fn recommendation_title<'a>(&self, anime: &'a Anime) -> &'a str {
        if self.prefer_english {
            preferred_title(anime)
        } else {
            &anime.title
        }
    }
}

fn preferred_title(anime: &Anime) -> &str {
    anime
        .aliases
        .first()
        .filter(|alias| latin_eligible(alias))
        .map(String::as_str)
        .unwrap_or(&anime.title)
}

fn latin_eligible(title: &str) -> bool {
    let mut ascii_letter = false;
    for ch in title.nfkd() {
        if ch.is_alphabetic() {
            if !ch.is_ascii_alphabetic() {
                return false;
            }
            ascii_letter = true;
        }
    }
    ascii_letter
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anime(title: &str, aliases: &[&str]) -> Anime {
        Anime {
            title: title.into(),
            aliases: aliases.iter().map(|alias| (*alias).into()).collect(),
            genres: vec![],
            score: None,
            year: None,
            anime_type: None,
            episodes: None,
            synopsis: None,
        }
    }

    #[test]
    fn first_alias_is_the_only_english_candidate() {
        assert_eq!(
            preferred_title(&anime("Karas", &["鴉 -KARAS-", "Crow"])),
            "Karas"
        );
        assert_eq!(
            preferred_title(&anime(
                "Koutetsujou no Kabaneri",
                &["Kabaneri of the Iron Fortress", "Other"]
            )),
            "Kabaneri of the Iron Fortress"
        );
        assert_eq!(preferred_title(&anime("Naruto", &[])), "Naruto");
    }

    #[test]
    fn latin_filter_allows_decomposable_accents_and_rejects_other_scripts() {
        for title in ["Attack on Titan", "Café", "KARAS-2"] {
            assert!(latin_eligible(title), "{title}");
        }
        for title in ["鴉 -KARAS-", "進撃の巨人", "123", "Łódź"] {
            assert!(!latin_eligible(title), "{title}");
        }
    }
}
