//! Strict, immutable reader for the three-file recommendation bundle.
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs;
use std::path::Path;

pub type MalId = i32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleError {
    location: String,
    reason: String,
}

impl BundleError {
    fn new(location: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.location, self.reason)
    }
}
impl std::error::Error for BundleError {}

type Result<T> = std::result::Result<T, BundleError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeCount(String);
impl EpisodeCount {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for EpisodeCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone)]
pub struct Anime {
    pub title: String,
    pub aliases: Vec<String>,
    pub genres: Vec<String>,
    pub score: Option<f64>,
    pub year: Option<u16>,
    pub anime_type: Option<String>,
    pub episodes: Option<EpisodeCount>,
    pub synopsis: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Neighbor {
    pub mal_id: MalId,
    pub similarity: f64,
}

#[derive(Debug)]
pub struct Catalog {
    anime: BTreeMap<MalId, Anime>,
}
impl Catalog {
    pub fn get(&self, id: MalId) -> Option<&Anime> {
        self.anime.get(&id)
    }
    pub fn iter(&self) -> impl Iterator<Item = (MalId, &Anime)> {
        self.anime.iter().map(|(&id, anime)| (id, anime))
    }
    pub fn len(&self) -> usize {
        self.anime.len()
    }
    pub fn is_empty(&self) -> bool {
        self.anime.is_empty()
    }
}

#[derive(Debug)]
pub struct Bundle {
    identity: String,
    catalog: Catalog,
    neighbors: BTreeMap<MalId, Vec<Neighbor>>,
}
impl Bundle {
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }
    pub fn neighbors(&self, id: MalId) -> Option<&[Neighbor]> {
        self.neighbors.get(&id).map(Vec::as_slice)
    }
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let manifest_bytes = read_file(directory, "manifest.json")?;
        let manifest = parse_file(&manifest_bytes, "manifest.json")?;
        let hashes = validate_manifest(&manifest)?;
        let catalog_bytes = read_file(directory, "catalog.json")?;
        let neighbors_bytes = read_file(directory, "neighbors.json")?;
        for (name, bytes) in [
            ("catalog.json", &catalog_bytes),
            ("neighbors.json", &neighbors_bytes),
        ] {
            if digest(bytes) != hashes[name] {
                return Err(BundleError::new(
                    format!("manifest.json.files.{name}.sha256"),
                    "raw byte hash mismatch",
                ));
            }
        }
        let catalog = validate_catalog(&parse_file(&catalog_bytes, "catalog.json")?)?;
        let neighbors =
            validate_neighbors(&parse_file(&neighbors_bytes, "neighbors.json")?, &catalog)?;
        Ok(Self {
            identity: format!("sha256:{}", digest(&manifest_bytes)),
            catalog,
            neighbors,
        })
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read_file(dir: &Path, name: &str) -> Result<Vec<u8>> {
    fs::read(dir.join(name))
        .map_err(|error| BundleError::new(name, format!("cannot read file: {error}")))
}

// RawValue retains number lexemes, including integers larger than u64. Every object is
// decoded through Pairs so duplicate decoded keys are visible at any nesting depth.
#[derive(Debug)]
enum Tree {
    Null,
    Bool,
    String(String),
    Number(String),
    Array(Vec<Tree>),
    Object(BTreeMap<String, Tree>),
}
struct Pairs(Vec<(String, Box<RawValue>)>);
impl<'de> Deserialize<'de> for Pairs {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct PairVisitor;
        impl<'de> Visitor<'de> for PairVisitor {
            type Value = Pairs;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Pairs, M::Error> {
                let mut pairs = Vec::new();
                while let Some(pair) = map.next_entry::<String, Box<RawValue>>()? {
                    pairs.push(pair);
                }
                Ok(Pairs(pairs))
            }
        }
        deserializer.deserialize_map(PairVisitor)
    }
}
fn parse_file(bytes: &[u8], name: &str) -> Result<Tree> {
    if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        return Err(BundleError::new(name, "UTF-8 BOM is forbidden"));
    }
    if !bytes.ends_with(b"\n") || bytes.ends_with(b"\n\n") || bytes.windows(2).any(|w| w == b"\r\n")
    {
        return Err(BundleError::new(
            name,
            "must end with exactly one LF and contain no CRLF",
        ));
    }
    let content =
        std::str::from_utf8(bytes).map_err(|_| BundleError::new(name, "invalid UTF-8"))?;
    let raw: Box<RawValue> = serde_json::from_str(content)
        .map_err(|error| BundleError::new(name, format!("invalid JSON: {error}")))?;
    parse_tree(raw.get(), name)
}
fn parse_tree(raw: &str, path: &str) -> Result<Tree> {
    match raw.as_bytes().first() {
        Some(b'{') => {
            let Pairs(pairs) = serde_json::from_str(raw)
                .map_err(|error| BundleError::new(path, format!("invalid object: {error}")))?;
            let mut entries = BTreeMap::new();
            for (key, value) in pairs {
                let child = format!("{path}.{key}");
                if entries.contains_key(&key) {
                    return Err(BundleError::new(child, "duplicate key"));
                }
                entries.insert(key, parse_tree(value.get(), &child)?);
            }
            Ok(Tree::Object(entries))
        }
        Some(b'[') => {
            let elements: Vec<Box<RawValue>> = serde_json::from_str(raw)
                .map_err(|error| BundleError::new(path, format!("invalid array: {error}")))?;
            let values = elements
                .iter()
                .enumerate()
                .map(|(index, value)| parse_tree(value.get(), &format!("{path}[{index}]")))
                .collect::<Result<Vec<_>>>()?;
            Ok(Tree::Array(values))
        }
        Some(b'"') => serde_json::from_str(raw)
            .map(Tree::String)
            .map_err(|error| BundleError::new(path, format!("invalid string: {error}"))),
        Some(b't') | Some(b'f') => Ok(Tree::Bool),
        Some(b'n') => Ok(Tree::Null),
        Some(_) => {
            if raw.contains(['.', 'e', 'E']) && !raw.parse::<f64>().is_ok_and(f64::is_finite) {
                return Err(BundleError::new(path, "nonfinite number"));
            }
            Ok(Tree::Number(raw.to_owned()))
        }
        None => Err(BundleError::new(path, "empty JSON value")),
    }
}
fn object<'a>(value: &'a Tree, path: &str) -> Result<&'a BTreeMap<String, Tree>> {
    match value {
        Tree::Object(map) => Ok(map),
        _ => Err(BundleError::new(path, "expected object")),
    }
}
fn array<'a>(value: &'a Tree, path: &str) -> Result<&'a [Tree]> {
    match value {
        Tree::Array(items) => Ok(items),
        _ => Err(BundleError::new(path, "expected array")),
    }
}
fn fields<'a>(
    value: &'a Tree,
    path: &str,
    expected: &[&str],
) -> Result<&'a BTreeMap<String, Tree>> {
    let map = object(value, path)?;
    for field in expected {
        if !map.contains_key(*field) {
            return Err(BundleError::new(format!("{path}.{field}"), "missing field"));
        }
    }
    for field in map.keys() {
        if !expected.contains(&field.as_str()) {
            return Err(BundleError::new(format!("{path}.{field}"), "unknown field"));
        }
    }
    Ok(map)
}
fn at<'a>(map: &'a BTreeMap<String, Tree>, key: &str) -> &'a Tree {
    &map[key]
}
fn string(value: &Tree, path: &str) -> Result<String> {
    match value {
        Tree::String(s) if !s.trim().is_empty() => Ok(s.clone()),
        _ => Err(BundleError::new(path, "expected nonblank string")),
    }
}
fn integer<'a>(value: &'a Tree, path: &str) -> Result<&'a str> {
    match value {
        Tree::Number(s) if !s.contains(['.', 'e', 'E']) => Ok(s),
        _ => Err(BundleError::new(path, "expected integer token")),
    }
}
fn bounded_integer(value: &Tree, path: &str, min: i32, max: i32) -> Result<i32> {
    let number = integer(value, path)?;
    match number.parse::<i32>() {
        Ok(n) if (min..=max).contains(&n) => Ok(n),
        _ => Err(BundleError::new(
            path,
            format!("expected integer from {min} through {max}"),
        )),
    }
}
fn number(value: &Tree, path: &str, min: f64, max: f64, inclusive_min: bool) -> Result<f64> {
    let Tree::Number(text) = value else {
        return Err(BundleError::new(path, "expected number"));
    };
    let n = text
        .parse::<f64>()
        .map_err(|_| BundleError::new(path, "invalid number"))?;
    if !n.is_finite() || n > max || (if inclusive_min { n < min } else { n <= min }) {
        return Err(BundleError::new(path, "number out of range"));
    }
    Ok(n)
}
fn nullable<T>(value: &Tree, parse: impl FnOnce(&Tree) -> Result<T>) -> Result<Option<T>> {
    if matches!(value, Tree::Null) {
        Ok(None)
    } else {
        parse(value).map(Some)
    }
}
fn version(map: &BTreeMap<String, Tree>, path: &str) -> Result<()> {
    if integer(at(map, "schema_version"), &format!("{path}.schema_version"))? != "1" {
        return Err(BundleError::new(
            format!("{path}.schema_version"),
            "expected version 1",
        ));
    }
    Ok(())
}
fn sha(value: &Tree, path: &str) -> Result<String> {
    let Tree::String(s) = value else {
        return Err(BundleError::new(path, "expected SHA-256 hex digest"));
    };
    if s.len() != 64
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(BundleError::new(
            path,
            "expected lowercase 64-character SHA-256 hex digest",
        ));
    }
    Ok(s.clone())
}
fn exact(value: &Tree, path: &str, expected: &str) -> Result<()> {
    if matches!(value, Tree::String(s) if s == expected) {
        Ok(())
    } else {
        Err(BundleError::new(path, format!("expected {expected}")))
    }
}
fn validate_manifest(value: &Tree) -> Result<BTreeMap<String, String>> {
    let root = fields(
        value,
        "manifest.json",
        &["schema_version", "algorithm", "sources", "files"],
    )?;
    version(root, "manifest.json")?;
    let algorithm = fields(
        at(root, "algorithm"),
        "manifest.json.algorithm",
        &[
            "name",
            "version",
            "max_neighbors",
            "similarity",
            "parameters",
        ],
    )?;
    exact(
        at(algorithm, "name"),
        "manifest.json.algorithm.name",
        "glove-mean-cosine",
    )?;
    string(at(algorithm, "version"), "manifest.json.algorithm.version")?;
    bounded_integer(
        at(algorithm, "max_neighbors"),
        "manifest.json.algorithm.max_neighbors",
        5,
        5,
    )?;
    exact(
        at(algorithm, "similarity"),
        "manifest.json.algorithm.similarity",
        "cosine",
    )?;
    object(
        at(algorithm, "parameters"),
        "manifest.json.algorithm.parameters",
    )?;
    let sources = array(at(root, "sources"), "manifest.json.sources")?;
    if sources.is_empty() {
        return Err(BundleError::new(
            "manifest.json.sources",
            "must be nonempty",
        ));
    }
    let mut previous: Option<(String, String)> = None;
    for (index, source) in sources.iter().enumerate() {
        let path = format!("manifest.json.sources[{index}]");
        let entry = fields(source, &path, &["name", "version", "sha256", "metadata"])?;
        let pair = (
            string(at(entry, "name"), &format!("{path}.name"))?,
            string(at(entry, "version"), &format!("{path}.version"))?,
        );
        if previous.as_ref().is_some_and(|last| last >= &pair) {
            return Err(BundleError::new(
                path,
                "sources must be sorted by unique (name, version) pairs",
            ));
        }
        previous = Some(pair);
        sha(at(entry, "sha256"), &format!("{path}.sha256"))?;
        object(at(entry, "metadata"), &format!("{path}.metadata"))?;
    }
    let files = fields(
        at(root, "files"),
        "manifest.json.files",
        &["catalog.json", "neighbors.json"],
    )?;
    let mut hashes = BTreeMap::new();
    for name in ["catalog.json", "neighbors.json"] {
        let path = format!("manifest.json.files.{name}");
        let entry = fields(at(files, name), &path, &["sha256"])?;
        hashes.insert(
            name.to_owned(),
            sha(at(entry, "sha256"), &format!("{path}.sha256"))?,
        );
    }
    Ok(hashes)
}
fn mal_key(key: &str, path: &str) -> Result<MalId> {
    let id = key
        .parse::<MalId>()
        .map_err(|_| BundleError::new(path, "invalid MAL_ID key"))?;
    if id < 1 || id.to_string() != key {
        return Err(BundleError::new(path, "noncanonical MAL_ID key"));
    }
    Ok(id)
}
fn strings(value: &Tree, path: &str) -> Result<Vec<String>> {
    let mut unique = HashSet::new();
    array(value, path)?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let location = format!("{path}[{index}]");
            let text = string(value, &location)?;
            if !unique.insert(text.clone()) {
                return Err(BundleError::new(location, "duplicate string"));
            }
            Ok(text)
        })
        .collect()
}
fn nullable_string(value: &Tree, path: &str) -> Result<Option<String>> {
    nullable(value, |value| string(value, path))
}
fn validate_catalog(value: &Tree) -> Result<Catalog> {
    let root = fields(value, "catalog.json", &["schema_version", "anime"])?;
    version(root, "catalog.json")?;
    let map = object(at(root, "anime"), "catalog.json.anime")?;
    if map.is_empty() {
        return Err(BundleError::new("catalog.json.anime", "must be nonempty"));
    }
    let mut anime = BTreeMap::new();
    for (key, value) in map {
        let path = format!("catalog.json.anime.{key}");
        let id = mal_key(key, &path)?;
        let entry = fields(
            value,
            &path,
            &[
                "title", "aliases", "genres", "score", "year", "type", "episodes", "synopsis",
            ],
        )?;
        let title = string(at(entry, "title"), &format!("{path}.title"))?;
        let aliases = strings(at(entry, "aliases"), &format!("{path}.aliases"))?;
        let genres = strings(at(entry, "genres"), &format!("{path}.genres"))?;
        let score = nullable(at(entry, "score"), |v| {
            number(v, &format!("{path}.score"), 1.0, 10.0, true)
        })?;
        let year = nullable(at(entry, "year"), |v| {
            bounded_integer(v, &format!("{path}.year"), 1, 9999).map(|n| n as u16)
        })?;
        let anime_type = nullable_string(at(entry, "type"), &format!("{path}.type"))?;
        let episodes = nullable(at(entry, "episodes"), |v| {
            let token = integer(v, &format!("{path}.episodes"))?;
            if !token
                .bytes()
                .next()
                .is_some_and(|b| (b'1'..=b'9').contains(&b))
                || !token.bytes().all(|b| b.is_ascii_digit())
            {
                return Err(BundleError::new(
                    format!("{path}.episodes"),
                    "expected positive integer",
                ));
            }
            Ok(EpisodeCount(token.to_owned()))
        })?;
        let synopsis = nullable_string(at(entry, "synopsis"), &format!("{path}.synopsis"))?;
        anime.insert(
            id,
            Anime {
                title,
                aliases,
                genres,
                score,
                year,
                anime_type,
                episodes,
                synopsis,
            },
        );
    }
    Ok(Catalog { anime })
}
fn validate_neighbors(value: &Tree, catalog: &Catalog) -> Result<BTreeMap<MalId, Vec<Neighbor>>> {
    let root = fields(value, "neighbors.json", &["schema_version", "neighbors"])?;
    version(root, "neighbors.json")?;
    let map = object(at(root, "neighbors"), "neighbors.json.neighbors")?;
    let mut neighbors = BTreeMap::new();
    for (key, value) in map {
        let path = format!("neighbors.json.neighbors.{key}");
        let source = mal_key(key, &path)?;
        if catalog.get(source).is_none() {
            return Err(BundleError::new(path, "source missing from catalog"));
        }
        let items = array(value, &path)?;
        if items.len() > 5 {
            return Err(BundleError::new(path, "more than five neighbors"));
        }
        let mut list = Vec::new();
        let mut targets = HashSet::new();
        for (index, value) in items.iter().enumerate() {
            let item_path = format!("{path}[{index}]");
            let entry = fields(value, &item_path, &["mal_id", "similarity"])?;
            let id_path = format!("{item_path}.mal_id");
            let target = bounded_integer(at(entry, "mal_id"), &id_path, 1, i32::MAX)?;
            if target == source {
                return Err(BundleError::new(id_path, "self neighbor"));
            }
            if catalog.get(target).is_none() {
                return Err(BundleError::new(id_path, "target missing from catalog"));
            }
            if !targets.insert(target) {
                return Err(BundleError::new(id_path, "duplicate neighbor"));
            }
            let similarity = number(
                at(entry, "similarity"),
                &format!("{item_path}.similarity"),
                0.0,
                1.0,
                false,
            )?;
            if list.last().is_some_and(|prior: &Neighbor| {
                prior.similarity < similarity
                    || (prior.similarity == similarity && prior.mal_id >= target)
            }) {
                return Err(BundleError::new(
                    item_path,
                    "neighbors must be ordered by descending similarity then ascending MAL_ID",
                ));
            }
            list.push(Neighbor {
                mal_id: target,
                similarity,
            });
        }
        neighbors.insert(source, list);
    }
    if neighbors.len() != catalog.len()
        || catalog.iter().any(|(id, _)| !neighbors.contains_key(&id))
    {
        return Err(BundleError::new(
            "neighbors.json.neighbors",
            "source IDs must exactly match catalog IDs",
        ));
    }
    Ok(neighbors)
}
