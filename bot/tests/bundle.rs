use bot::catalog::Bundle;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};
use tempfile::TempDir;

type Mutation = (&'static str, fn(&mut Value));

fn fixture() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    for name in ["manifest.json", "catalog.json", "neighbors.json"] {
        fs::copy(
            Path::new("../tests/fixtures/bundle").join(name),
            dir.path().join(name),
        )
        .unwrap();
    }
    dir
}
fn write(dir: &Path, name: &str, text: &str) {
    fs::write(dir.join(name), text).unwrap();
}
fn modify(dir: &Path, name: &str, change: impl FnOnce(&mut Value)) {
    let mut value: Value = serde_json::from_slice(&fs::read(dir.join(name)).unwrap()).unwrap();
    change(&mut value);
    write(
        dir,
        name,
        &format!("{}\n", serde_json::to_string(&value).unwrap()),
    );
    if name != "manifest.json" {
        rehash(dir, name);
    }
}
fn rehash(dir: &Path, name: &str) {
    let bytes = fs::read(dir.join(name)).unwrap();
    let hash = format!("{:x}", Sha256::digest(bytes));
    modify(dir, "manifest.json", |manifest| {
        manifest["files"][name]["sha256"] = json!(hash)
    });
}
fn fails(dir: &Path, expected: &str) {
    let error = Bundle::load(dir).unwrap_err().to_string();
    assert!(
        error.contains(expected),
        "expected {expected:?} in {error:?}"
    );
}

#[test]
fn fixture_exposes_numeric_ids_order_and_immutable_views() {
    let dir = fixture();
    let bundle = Bundle::load(dir.path()).unwrap();
    assert_eq!(
        bundle.identity(),
        "sha256:907e00dcd591838876febdd60d022318752f224a317816cf0a1b2a5b3df31fdb"
    );
    assert_eq!(bundle.catalog().len(), 8);
    assert!(!bundle.catalog().is_empty());
    assert_eq!(
        bundle
            .catalog()
            .iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        [1, 2, 3, 10, 11, 12, 20, 99]
    );
    assert_eq!(
        bundle.catalog().get(1).unwrap().title,
        bundle.catalog().get(2).unwrap().title
    );
    assert_eq!(
        bundle.catalog().get(3).unwrap().aliases,
        ["Тэнку но рессха", "天空の列車"]
    );
    assert_eq!(
        bundle
            .catalog()
            .get(1)
            .unwrap()
            .episodes
            .as_ref()
            .unwrap()
            .as_str(),
        "12"
    );
    assert!(bundle.catalog().get(99).unwrap().score.is_none());
    assert_eq!(
        bundle
            .neighbors(1)
            .unwrap()
            .iter()
            .map(|n| n.mal_id)
            .collect::<Vec<_>>(),
        [10, 11, 12, 2, 3]
    );
    assert_eq!(bundle.neighbors(99), Some(&[][..]));
    assert_eq!(bundle.neighbors(100), None);
}
#[test]
fn io_bytes_hash_and_identity() {
    for name in ["manifest.json", "catalog.json", "neighbors.json"] {
        let dir = fixture();
        fs::remove_file(dir.path().join(name)).unwrap();
        fails(dir.path(), name);
    }
    let dir = fixture();
    fs::write(dir.path().join("catalog.json"), b"changed\n").unwrap();
    fails(dir.path(), "hash mismatch");
    let dir = fixture();
    let old = Bundle::load(dir.path()).unwrap().identity().to_owned();
    modify(dir.path(), "catalog.json", |v| {
        v["anime"]["1"]["title"] = json!("Новый заголовок")
    });
    assert_ne!(old, Bundle::load(dir.path()).unwrap().identity());
    let new = Bundle::load(dir.path()).unwrap().identity().to_owned();
    modify(dir.path(), "manifest.json", |v| {
        v["sources"][0]["version"] = json!("future-v2")
    });
    assert_ne!(new, Bundle::load(dir.path()).unwrap().identity());
}
#[test]
fn strict_json_and_bytes() {
    for (text, expected) in [
        (
            "{\"schema_version\":1,\"schema_version\":1}\n",
            "duplicate key",
        ),
        ("{\"a\":{\"x\":1,\"\\u0078\":2}}\n", "duplicate key"),
        ("{\"a\":{\"x\":1,\"x\":2}}\n", "duplicate key"),
        ("{\"a\":NaN}\n", "invalid JSON"),
        ("{\"a\":Infinity}\n", "invalid JSON"),
        ("{\"a\":1e999}\n", "nonfinite number"),
        ("{\"a\":1}\n\n", "exactly one LF"),
        ("{\"a\":1}\r\n", "no CRLF"),
        ("{\"a\":1}", "exactly one LF"),
        ("{", "exactly one LF"),
    ] {
        let dir = fixture();
        write(dir.path(), "manifest.json", text);
        fails(dir.path(), expected);
    }
    let dir = fixture();
    fs::write(dir.path().join("manifest.json"), b"\xef\xbb\xbf{}\n").unwrap();
    fails(dir.path(), "BOM");
    let dir = fixture();
    fs::write(dir.path().join("manifest.json"), b"\xff\n").unwrap();
    fails(dir.path(), "UTF-8");
}
#[test]
fn duplicates_inside_free_objects_are_rejected() {
    let dir = fixture();
    let manifest = fs::read_to_string(dir.path().join("manifest.json")).unwrap();
    write(
        dir.path(),
        "manifest.json",
        &manifest.replace("\"parameters\":{}", "\"parameters\":{\"x\":1,\"x\":2}"),
    );
    fails(
        dir.path(),
        "manifest.json.algorithm.parameters.x: duplicate key",
    );
    let dir = fixture();
    let manifest = fs::read_to_string(dir.path().join("manifest.json")).unwrap();
    write(
        dir.path(),
        "manifest.json",
        &manifest.replace("\"dimensions\":2", "\"dimensions\":2,\"dimensions\":3"),
    );
    fails(
        dir.path(),
        "manifest.json.sources[0].metadata.dimensions: duplicate key",
    );
}
#[test]
fn manifest_rules() {
    let cases: &[Mutation] = &[
        ("schema_version", |v| v["schema_version"] = json!(1.0)),
        ("algorithm.name", |v| {
            v["algorithm"]["name"] = json!("other")
        }),
        ("algorithm.version", |v| {
            v["algorithm"]["version"] = json!(" ")
        }),
        ("algorithm.max_neighbors", |v| {
            v["algorithm"]["max_neighbors"] = json!(6)
        }),
        ("algorithm.similarity", |v| {
            v["algorithm"]["similarity"] = json!("dot")
        }),
        ("algorithm.parameters", |v| {
            v["algorithm"]["parameters"] = json!([])
        }),
        ("sources", |v| v["sources"] = json!([])),
        ("sources[0].sha256", |v| {
            v["sources"][0]["sha256"] = json!("ABC")
        }),
        ("sources[1]", |v| {
            let copy = v["sources"][0].clone();
            v["sources"].as_array_mut().unwrap().push(copy);
        }),
        ("files", |v| {
            v["files"]["other"] = json!({"sha256":"x"});
        }),
        ("files.catalog.json.foo", |v| {
            v["files"]["catalog.json"]["foo"] = json!(1)
        }),
        ("algorithm.foo", |v| v["algorithm"]["foo"] = json!(1)),
    ];
    for (expected, change) in cases {
        let dir = fixture();
        modify(dir.path(), "manifest.json", *change);
        fails(dir.path(), expected);
    }
    let dir = fixture();
    modify(dir.path(), "manifest.json", |v| {
        v["sources"] = json!([
            {"name":"z","version":"1","sha256":"0".repeat(64),"metadata":{}},
            v["sources"][0].clone()
        ]);
    });
    fails(dir.path(), "sources[1]");
}
#[test]
fn catalog_field_rules_and_huge_episodes() {
    let cases: &[Mutation] = &[
        ("title", |v| v["anime"]["1"]["title"] = json!(" ")),
        ("aliases[1]", |v| {
            v["anime"]["1"]["aliases"] = json!(["a", "a"])
        }),
        ("genres[0]", |v| v["anime"]["1"]["genres"] = json!([" "])),
        ("score", |v| v["anime"]["1"]["score"] = json!(10.1)),
        ("year", |v| v["anime"]["1"]["year"] = json!(1.0)),
        ("type", |v| v["anime"]["1"]["type"] = json!("")),
        ("episodes", |v| v["anime"]["1"]["episodes"] = json!(0)),
        ("synopsis", |v| v["anime"]["1"]["synopsis"] = json!(" ")),
        ("extra", |v| v["anime"]["1"]["extra"] = json!(1)),
        ("score", |v| {
            v["anime"]["1"].as_object_mut().unwrap().remove("score");
        }),
        ("anime", |v| v["anime"] = json!({})),
        ("anime.01", |v| {
            v["anime"]["01"] = v["anime"]["1"].clone();
        }),
        ("anime.2147483648", |v| {
            v["anime"]["2147483648"] = v["anime"]["1"].clone();
        }),
    ];
    for (expected, change) in cases {
        let dir = fixture();
        modify(dir.path(), "catalog.json", *change);
        fails(dir.path(), expected);
    }
    let dir = fixture();
    let catalog = fs::read_to_string(dir.path().join("catalog.json")).unwrap();
    let huge = "9".repeat(300);
    write(
        dir.path(),
        "catalog.json",
        &catalog.replacen("\"episodes\":12", &format!("\"episodes\":{huge}"), 1),
    );
    rehash(dir.path(), "catalog.json");
    let bundle = Bundle::load(dir.path()).unwrap();
    assert_eq!(
        bundle
            .catalog()
            .get(1)
            .unwrap()
            .episodes
            .as_ref()
            .unwrap()
            .as_str(),
        huge
    );
}
#[test]
fn neighbor_rules() {
    let cases: &[Mutation] = &[
        ("neighbors.1[0].mal_id", |v| {
            v["neighbors"]["1"][0]["mal_id"] = json!(1)
        }),
        ("neighbors.1[0].mal_id", |v| {
            v["neighbors"]["1"][0]["mal_id"] = json!(777)
        }),
        ("neighbors.1[1].mal_id", |v| {
            v["neighbors"]["1"][1]["mal_id"] = json!(10)
        }),
        ("neighbors.1[0].mal_id", |v| {
            v["neighbors"]["1"][0]["mal_id"] = json!(10.0)
        }),
        ("neighbors.1[0].similarity", |v| {
            v["neighbors"]["1"][0]["similarity"] = json!(0)
        }),
        ("neighbors.1[0].similarity", |v| {
            v["neighbors"]["1"][0]["similarity"] = json!(1.1)
        }),
        ("neighbors.1[1]", |v| {
            v["neighbors"]["1"].as_array_mut().unwrap().swap(0, 1)
        }),
        ("neighbors.1[0]", |v| {
            v["neighbors"]["1"][0]["extra"] = json!(1)
        }),
        ("neighbors.1", |v| {
            v["neighbors"]["1"]
                .as_array_mut()
                .unwrap()
                .push(json!({"mal_id":99,"similarity":0.1}))
        }),
        ("neighbors", |v| {
            v["neighbors"].as_object_mut().unwrap().remove("99");
        }),
        ("neighbors.01", |v| {
            v["neighbors"]["01"] = json!([]);
        }),
        ("neighbors.777", |v| {
            v["neighbors"]["777"] = json!([]);
        }),
    ];
    for (expected, change) in cases {
        let dir = fixture();
        modify(dir.path(), "neighbors.json", *change);
        fails(dir.path(), expected);
    }
}
#[test]
fn cli_and_bot_startup() {
    let dir = fixture();
    let cli = Command::new(env!("CARGO_BIN_EXE_check_bundle"))
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(cli.status.success());
    assert_eq!(
        String::from_utf8(cli.stdout).unwrap(),
        "sha256:907e00dcd591838876febdd60d022318752f224a317816cf0a1b2a5b3df31fdb records=8\n"
    );
    write(dir.path(), "catalog.json", "broken\n");
    let cli = Command::new(env!("CARGO_BIN_EXE_check_bundle"))
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(!cli.status.success());
    assert!(String::from_utf8(cli.stderr)
        .unwrap()
        .contains("hash mismatch"));
    let bot = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_bot"));
        command
            .env("TELOXIDE_TOKEN", "123:abc")
            .env("DATABASE_URL", "postgresql://test@127.0.0.1:1/test")
            .env("ARTIFACTS_DIR", dir.path());
        command
    };
    let config = bot().arg("--check-config").output().unwrap();
    assert!(config.status.success());
    let startup = bot().output().unwrap();
    assert!(!startup.status.success());
    let stderr = String::from_utf8(startup.stderr).unwrap();
    assert!(
        stderr.contains("Recommendation bundle failed to load"),
        "{stderr}"
    );
    assert!(stderr.contains("hash mismatch"), "{stderr}");
}
