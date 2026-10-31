use bot::catalog::Bundle;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};
use tempfile::TempDir;

type Mutation = (&'static str, fn(&mut Value));

fn fixture() -> TempDir {
    copied(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bundle"
    ))
}
fn fixture_v2() -> TempDir {
    copied(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bundle-v2"
    ))
}
fn copied(source: &str) -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    for name in ["manifest.json", "catalog.json", "neighbors.json"] {
        fs::copy(Path::new(source).join(name), dir.path().join(name)).unwrap();
    }
    dir
}
fn ids(neighbors: Option<&[bot::catalog::Neighbor]>) -> Vec<i32> {
    neighbors.unwrap().iter().map(|n| n.mal_id).collect()
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
fn manifest_accepts_itemknn_algorithm() {
    let dir = fixture();
    modify(dir.path(), "manifest.json", |v| {
        v["algorithm"]["name"] = json!("itemknn-shrunk-cosine")
    });
    let bundle = Bundle::load(dir.path()).unwrap();
    assert_eq!(bundle.catalog().len(), 8);
    modify(dir.path(), "manifest.json", |v| {
        v["algorithm"]["name"] = json!("other")
    });
    fails(
        dir.path(),
        "manifest.json.algorithm.name: expected glove-mean-cosine or itemknn-shrunk-cosine",
    );
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
fn python_strip_whitespace_is_blank_without_changing_valid_text() {
    for blank in [
        "\u{001c}",
        "\u{001d}",
        "\u{001e}",
        "\u{001f}",
        " \u{001c}\u{2003}\u{001f} ",
    ] {
        let dir = fixture();
        modify(dir.path(), "catalog.json", |v| {
            v["anime"]["1"]["title"] = json!(blank)
        });
        fails(
            dir.path(),
            "catalog.json.anime.1.title: expected nonblank string",
        );
    }
    let dir = fixture();
    modify(dir.path(), "manifest.json", |v| {
        v["algorithm"]["version"] = json!("\u{001d}")
    });
    fails(
        dir.path(),
        "manifest.json.algorithm.version: expected nonblank string",
    );

    let preserved = " \u{001c}Текст\u{001f} ";
    let dir = fixture();
    modify(dir.path(), "catalog.json", |v| {
        v["anime"]["1"]["title"] = json!(preserved)
    });
    assert_eq!(
        Bundle::load(dir.path())
            .unwrap()
            .catalog()
            .get(1)
            .unwrap()
            .title,
        preserved
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

#[test]
fn v2_fixture_exposes_franchises_and_long_neighbor_lists() {
    let dir = fixture_v2();
    let bundle = Bundle::load(dir.path()).unwrap();
    assert_eq!(
        bundle.identity(),
        "sha256:95e6bd839abc21308c4e2758ce8579086b00919a10889578f701c782fc707412"
    );
    assert_eq!(
        [1, 10, 11, 2, 99].map(|id| bundle.franchise(id)),
        [1, 1, 1, 2, 99]
    );
    assert_eq!(ids(bundle.neighbors(10)), [12, 2, 3]);
    assert_eq!(bundle.neighbors(10).unwrap()[0].similarity, 0.9);
    // A v1 bundle has no franchise map.
    let v1 = fixture();
    assert_eq!(Bundle::load(v1.path()).unwrap().franchise(10), 10);
    // Cards show the first five of up to fifty neighbors.
    modify(dir.path(), "catalog.json", |v| {
        for id in ["100", "101", "102"] {
            v["anime"][id] = v["anime"]["20"].clone();
        }
    });
    modify(dir.path(), "neighbors.json", |v| {
        for id in ["100", "101", "102"] {
            v["neighbors"][id] = json!([]);
        }
        v["neighbors"]["99"] = json!([
            [1, 0.9],
            [2, 0.8],
            [3, 0.7],
            [12, 0.6],
            [20, 0.5],
            [100, 0.4],
            [101, 0.3],
            [102, 0.2]
        ]);
    });
    let bundle = Bundle::load(dir.path()).unwrap();
    assert_eq!(ids(bundle.neighbors(99)).len(), 8);
    assert_eq!(ids(bundle.cards(99)), [1, 2, 3, 12, 20]);
}
#[test]
fn v2_neighbor_franchise_and_version_rules() {
    let neighbors: &[Mutation] = &[
        ("own franchise", |v| {
            v["neighbors"]["10"][0] = json!([1, 0.95])
        }),
        ("not the first entry", |v| {
            v["neighbors"]["12"][0] = json!([10, 0.9])
        }),
        ("expected [mal_id, similarity]", |v| {
            v["neighbors"]["1"][0] = json!({"mal_id": 12, "similarity": 0.9})
        }),
        ("more than 50 neighbors", |v| {
            v["neighbors"]["99"] = json!((100..151).map(|id| json!([id, 0.5])).collect::<Vec<_>>())
        }),
        ("a first entry is not listed", |v| {
            v["franchises"]["12"] = json!(12)
        }),
        ("first entry is not a first entry", |v| {
            v["franchises"]["12"] = json!(10)
        }),
        ("title missing from catalog", |v| {
            v["franchises"]["555"] = json!(1)
        }),
        ("missing field", |v| {
            v.as_object_mut().unwrap().remove("franchises");
        }),
        ("expected version 2", |v| v["schema_version"] = json!(1)),
    ];
    for (expected, change) in neighbors {
        let dir = fixture_v2();
        modify(dir.path(), "neighbors.json", change);
        fails(dir.path(), expected);
    }
    let manifests: &[Mutation] = &[
        ("max_neighbors", |m| {
            m["algorithm"]["max_neighbors"] = json!(5)
        }),
        ("expected version 1 or 2", |m| {
            m["schema_version"] = json!(3)
        }),
        // A v1 manifest does not cover a v2 neighbors file.
        ("unknown field", |m| {
            m["schema_version"] = json!(1);
            m["algorithm"]["max_neighbors"] = json!(5);
        }),
    ];
    for (expected, change) in manifests {
        let dir = fixture_v2();
        modify(dir.path(), "manifest.json", change);
        fails(dir.path(), expected);
    }
}
