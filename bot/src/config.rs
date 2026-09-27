use std::{
    env, fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

use log::LevelFilter;
use tokio_postgres::{config::SslMode, Config as PgConfig};

pub(crate) struct Config {
    token: String,
    database: PgConfig,
    artifacts_dir: PathBuf,
    log_directives: Vec<LogDirective>,
}

struct LogDirective {
    module: Option<String>,
    level: LevelFilter,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ConfigError {
    variable: &'static str,
    reason: &'static str,
}

impl ConfigError {
    fn new(variable: &'static str, reason: &'static str) -> Self {
        Self { variable, reason }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.variable, self.reason)
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    pub(crate) fn from_env() -> Result<Self, ConfigError> {
        Self::load_with(|name| env::var(name))
    }

    fn load_with<F>(mut lookup: F) -> Result<Self, ConfigError>
    where
        F: FnMut(&str) -> Result<String, env::VarError>,
    {
        let token = required(&mut lookup, "TELOXIDE_TOKEN")?;
        if !valid_token(&token) {
            return Err(ConfigError::new(
                "TELOXIDE_TOKEN",
                "invalid bot token format",
            ));
        }

        let database_url = required(&mut lookup, "DATABASE_URL")?;
        let database = parse_database(&database_url)?;

        let artifact_path = required(&mut lookup, "ARTIFACTS_DIR")?;
        let artifacts_dir = resolve_artifacts_dir(&artifact_path)?;

        let log_spec = match lookup("RUST_LOG") {
            Ok(value) => value,
            Err(env::VarError::NotPresent) => "info".to_owned(),
            Err(env::VarError::NotUnicode(_)) => {
                return Err(ConfigError::new("RUST_LOG", "must be valid Unicode"));
            }
        };
        let log_directives = parse_log_directives(&log_spec)?;

        Ok(Self {
            token,
            database,
            artifacts_dir,
            log_directives,
        })
    }

    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    pub(crate) fn database(&self) -> &PgConfig {
        &self.database
    }

    pub(crate) fn artifacts_dir(&self) -> &Path {
        &self.artifacts_dir
    }

    pub(crate) fn init_logging(&self) -> Result<(), ConfigError> {
        let mut builder = pretty_env_logger::formatted_builder();
        for directive in &self.log_directives {
            builder.filter(directive.module.as_deref(), directive.level);
        }
        builder
            .try_init()
            .map_err(|_| ConfigError::new("RUST_LOG", "cannot initialize logger"))
    }
}

fn required<F>(lookup: &mut F, name: &'static str) -> Result<String, ConfigError>
where
    F: FnMut(&str) -> Result<String, env::VarError>,
{
    let value = match lookup(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Err(ConfigError::new(name, "is required")),
        Err(env::VarError::NotUnicode(_)) => {
            return Err(ConfigError::new(name, "must be valid Unicode"));
        }
    };
    if value.trim().is_empty() {
        return Err(ConfigError::new(name, "must not be blank"));
    }
    Ok(value)
}

fn valid_token(token: &str) -> bool {
    let Some((bot_id, secret)) = token.split_once(':') else {
        return false;
    };
    !bot_id.is_empty()
        && bot_id.bytes().all(|ch| ch.is_ascii_digit())
        && !secret.is_empty()
        && secret
            .bytes()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == b'_' || ch == b'-')
}

fn parse_database(url: &str) -> Result<PgConfig, ConfigError> {
    if !(url.starts_with("postgres://") || url.starts_with("postgresql://")) {
        return Err(ConfigError::new("DATABASE_URL", "must be a PostgreSQL URI"));
    }
    let database = PgConfig::from_str(url)
        .map_err(|_| ConfigError::new("DATABASE_URL", "invalid PostgreSQL URI"))?;
    if database.get_user().is_none_or(str::is_empty)
        || database.get_dbname().is_none_or(str::is_empty)
        || (database.get_hosts().is_empty() && database.get_hostaddrs().is_empty())
    {
        return Err(ConfigError::new(
            "DATABASE_URL",
            "user, database, and host are required",
        ));
    }
    if database.get_ssl_mode() == SslMode::Require {
        return Err(ConfigError::new(
            "DATABASE_URL",
            "sslmode=require is unsupported",
        ));
    }
    Ok(database)
}

fn resolve_artifacts_dir(value: &str) -> Result<PathBuf, ConfigError> {
    let path = PathBuf::from(value);
    let path = if path.is_absolute() {
        path
    } else {
        env::current_dir()
            .map_err(|_| ConfigError::new("ARTIFACTS_DIR", "cannot resolve current directory"))?
            .join(path)
    };
    if !path.is_dir() {
        return Err(ConfigError::new(
            "ARTIFACTS_DIR",
            "must identify an existing directory",
        ));
    }
    Ok(path)
}

fn parse_log_directives(value: &str) -> Result<Vec<LogDirective>, ConfigError> {
    let invalid = || ConfigError::new("RUST_LOG", "invalid log level or module directive");
    let mut default = LevelFilter::Info;
    let mut modules = Vec::new();
    let mut saw_default = false;
    for raw in value.split(',') {
        let directive = raw.trim();
        if directive.is_empty() {
            return Err(invalid());
        }
        if let Some((module, level)) = directive.split_once('=') {
            if !valid_module(module) {
                return Err(invalid());
            }
            modules.push(LogDirective {
                module: Some(module.to_owned()),
                level: parse_level(level).ok_or_else(invalid)?,
            });
        } else {
            if saw_default {
                return Err(invalid());
            }
            default = parse_level(directive).ok_or_else(invalid)?;
            saw_default = true;
        }
    }
    let mut directives = vec![LogDirective {
        module: None,
        level: default,
    }];
    directives.extend(modules);
    Ok(directives)
}

fn parse_level(value: &str) -> Option<LevelFilter> {
    match value {
        "off" => Some(LevelFilter::Off),
        "error" => Some(LevelFilter::Error),
        "warn" => Some(LevelFilter::Warn),
        "info" => Some(LevelFilter::Info),
        "debug" => Some(LevelFilter::Debug),
        "trace" => Some(LevelFilter::Trace),
        _ => None,
    }
}

fn valid_module(value: &str) -> bool {
    !value.is_empty()
        && value.split("::").all(|segment| {
            !segment.is_empty()
                && segment
                    .bytes()
                    .next()
                    .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == b'_')
                && segment
                    .bytes()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == b'_')
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::HashMap,
        ffi::OsString,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static NEXT_PATH: AtomicUsize = AtomicUsize::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!("arb-001-config-{}-{id}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn values(dir: &TestDir) -> HashMap<&'static str, String> {
        HashMap::from([
            ("TELOXIDE_TOKEN", "12345:ABC_def-9".to_owned()),
            (
                "DATABASE_URL",
                "postgresql://user:p%40ss@localhost:5432/anime?sslmode=disable".to_owned(),
            ),
            ("ARTIFACTS_DIR", dir.0.to_str().unwrap().to_owned()),
        ])
    }

    fn load(values: &HashMap<&str, String>) -> Result<Config, ConfigError> {
        Config::load_with(|key| values.get(key).cloned().ok_or(env::VarError::NotPresent))
    }

    #[test]
    fn accepts_valid_configuration_and_log_overrides() {
        let dir = TestDir::new();
        let mut input = values(&dir);
        let config = load(&input).unwrap();
        assert_eq!(config.token(), "12345:ABC_def-9");
        assert_eq!(config.artifacts_dir(), dir.0.as_path());
        assert_eq!(config.database().get_user(), Some("user"));
        assert_eq!(config.database().get_password(), Some(b"p@ss".as_slice()));
        assert_eq!(config.log_directives[0].level, LevelFilter::Info);
        input.insert("RUST_LOG", "warn,teloxide=debug,bot::db=trace".to_owned());
        let config = load(&input).unwrap();
        assert_eq!(config.log_directives[0].level, LevelFilter::Warn);
        assert_eq!(config.log_directives[1].module.as_deref(), Some("teloxide"));
        assert_eq!(config.log_directives[1].level, LevelFilter::Debug);
        assert_eq!(config.log_directives[2].module.as_deref(), Some("bot::db"));
        input.insert("ARTIFACTS_DIR", "src".to_owned());
        let config = load(&input).unwrap();
        assert_eq!(
            config.artifacts_dir(),
            env::current_dir().unwrap().join("src")
        );
    }

    #[test]
    fn missing_blank_and_non_unicode_are_safe() {
        let dir = TestDir::new();
        let mut input = values(&dir);
        input.remove("TELOXIDE_TOKEN");
        assert_eq!(load(&input).err().unwrap().variable, "TELOXIDE_TOKEN");
        input.insert("TELOXIDE_TOKEN", "  ".to_owned());
        assert_eq!(load(&input).err().unwrap().reason, "must not be blank");
        let err = Config::load_with(|key| {
            if key == "TELOXIDE_TOKEN" {
                Err(env::VarError::NotUnicode(OsString::from("secret")))
            } else {
                Ok(input[key].clone())
            }
        })
        .err()
        .unwrap();
        assert_eq!(err.to_string(), "TELOXIDE_TOKEN: must be valid Unicode");
    }

    #[test]
    fn rejects_invalid_tokens_and_database_urls() {
        let dir = TestDir::new();
        let mut input = values(&dir);
        for token in ["abc:def", "123:", "12:abc def", "12:abc:def", "１２:abc"] {
            input.insert("TELOXIDE_TOKEN", token.to_owned());
            assert_eq!(load(&input).err().unwrap().variable, "TELOXIDE_TOKEN");
        }
        input.insert("TELOXIDE_TOKEN", "123:secret".to_owned());
        for url in [
            "localhost:5432",
            "postgresql://user@localhost/anime?sslmode=disable&bad=value",
            "postgresql://localhost/anime",
            "postgresql://user@localhost/",
            "postgresql://user@/anime",
            "postgresql://user@localhost/anime?sslmode=require",
        ] {
            input.insert("DATABASE_URL", url.to_owned());
            assert_eq!(
                load(&input).err().unwrap().variable,
                "DATABASE_URL",
                "{url}"
            );
        }
    }

    #[test]
    fn rejects_invalid_artifact_paths_and_log_directives() {
        let dir = TestDir::new();
        let mut input = values(&dir);
        input.insert("ARTIFACTS_DIR", dir.0.join("missing").display().to_string());
        assert_eq!(load(&input).err().unwrap().variable, "ARTIFACTS_DIR");
        let file = dir.0.join("file");
        std::fs::write(&file, "fixture").unwrap();
        input.insert("ARTIFACTS_DIR", file.display().to_string());
        assert_eq!(load(&input).err().unwrap().variable, "ARTIFACTS_DIR");
        input.insert("ARTIFACTS_DIR", dir.0.display().to_string());
        for directive in [
            "",
            "INFO",
            "info,",
            "bot=",
            "=debug",
            "bot=debug/regex",
            "bot=debug=trace",
            "bad-module=info",
            "info,warn",
        ] {
            input.insert("RUST_LOG", directive.to_owned());
            assert_eq!(
                load(&input).err().unwrap().variable,
                "RUST_LOG",
                "{directive}"
            );
        }
    }

    #[test]
    fn errors_never_contain_secret_values() {
        let dir = TestDir::new();
        let mut input = values(&dir);
        let token_secret = "token_SENTINEL_283";
        let password_secret = "password_SENTINEL_731";
        input.insert("TELOXIDE_TOKEN", format!("bad:{token_secret}"));
        input.insert(
            "DATABASE_URL",
            format!("postgresql://user:{password_secret}@localhost/db?sslmode=require"),
        );
        let err = load(&input).err().unwrap();
        for output in [err.to_string(), format!("{err:?}")] {
            assert!(!output.contains(token_secret));
            assert!(!output.contains(password_secret));
        }
        input.insert("TELOXIDE_TOKEN", "123:valid".to_owned());
        let err = load(&input).err().unwrap();
        for output in [err.to_string(), format!("{err:?}")] {
            assert!(!output.contains(password_secret));
        }
    }
}
