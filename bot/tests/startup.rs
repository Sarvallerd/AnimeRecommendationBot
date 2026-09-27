use std::{
    path::Path,
    process::{Command, Output},
};
use tempfile::TempDir;

fn run(args: &[&str], artifacts: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bot"))
        .args(args)
        .env("TELOXIDE_TOKEN", "0:prepare")
        .env(
            "DATABASE_URL",
            "postgresql://anime_bot:dummy@127.0.0.1:1/anime_bot?sslmode=disable",
        )
        .env("ARTIFACTS_DIR", artifacts)
        .env("RUST_LOG", "off")
        .output()
        .unwrap()
}

#[test]
fn check_config_keeps_its_existing_directory_only_semantics() {
    let dir = TempDir::new().unwrap();
    let result = run(&["--check-config"], dir.path());
    assert!(result.status.success());
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "Configuration is valid.\n"
    );
}

#[test]
fn prepare_rejects_invalid_bundle_before_connecting_to_database() {
    let dir = TempDir::new().unwrap();
    let result = run(&["--prepare"], dir.path());
    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(
        stderr.contains("Recommendation bundle failed to load:"),
        "{stderr}"
    );
    assert!(!stderr.contains("Database connection failed"));
}

#[test]
fn invalid_arguments_are_rejected_before_configuration() {
    let dir = TempDir::new().unwrap();
    for args in [&["--prepare", "extra"][..], &["--unknown"][..]] {
        let result = run(args, dir.path());
        assert!(!result.status.success());
        assert_eq!(
            String::from_utf8(result.stderr).unwrap(),
            "Usage: bot [--check-config|--prepare]\n"
        );
    }
}
