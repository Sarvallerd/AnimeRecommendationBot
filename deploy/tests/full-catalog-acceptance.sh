#!/usr/bin/env bash
set -euo pipefail
umask 077
repo=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$repo"
usage() { echo 'Usage: full-catalog-acceptance.sh --project NAME --settings FILE --data-dir DIR --evidence-dir NEW_DIR [--offline]' >&2; exit 2; }
project= settings= data_dir= evidence_dir= offline=0
while (($#)); do
    case "$1" in
        --project|--settings|--data-dir|--evidence-dir)
            (($# >= 2)) || usage
            case "$1" in
                --project) project=$2 ;; --settings) settings=$2 ;;
                --data-dir) data_dir=$2 ;; --evidence-dir) evidence_dir=$2 ;;
            esac
            shift 2 ;;
        --offline) offline=1; shift ;;
        *) usage ;;
    esac
done
[[ -n "$project" && -n "$settings" && -n "$data_dir" && -n "$evidence_dir" ]] || usage
command -v docker >/dev/null
command -v python3 >/dev/null
[[ $(git -c "safe.directory=$repo" rev-parse --show-toplevel) == "$repo" ]] || { echo 'FAIL checkout' >&2; exit 1; }
[[ -z $(git -c "safe.directory=$repo" status --porcelain --untracked-files=normal) ]] || { echo 'FAIL dirty_checkout' >&2; exit 1; }
# All caller-provided Compose and ARB overrides are discarded. DOCKER_CONFIG is preserved.
for variable in "${!ARB_@}" "${!COMPOSE_@}"; do unset "$variable"; done
mapfile -d '' -t settings_values < <(python3 - "$settings" "$project" "$data_dir" "$evidence_dir" "$repo" <<'PY'
import importlib.util, os, pathlib, subprocess, sys
script = pathlib.Path('deploy/tests/acceptance-evidence.py').resolve()
spec = importlib.util.spec_from_file_location('arb020_evidence', script)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
settings_path, project, data_path, evidence_path, repo = map(pathlib.Path, sys.argv[1:])
try:
    values = module.read_settings(settings_path)
    if not module.PROJECT.fullmatch(str(project)) or not settings_path.is_absolute(): raise ValueError()
    if not data_path.is_absolute() or not evidence_path.is_absolute(): raise ValueError()
    def unsafe(path):
        return any(part.is_symlink() for part in (path, *path.parents))
    if any(unsafe(path) for path in (settings_path,data_path,evidence_path,
           pathlib.Path(values['ARB_BOT_ENV']),pathlib.Path(values['ARB_POSTGRES_ENV']))): raise ValueError()
    if data_path.resolve() != pathlib.Path(values['ARB_DATA_DIR']).resolve(): raise ValueError()
    if data_path.resolve() == evidence_path.resolve() or data_path.resolve() in evidence_path.resolve().parents: raise ValueError()
    if evidence_path.resolve() in data_path.resolve().parents: raise ValueError()
    if '.worktrees' in data_path.parts or '.worktrees' in evidence_path.parts: raise ValueError()
    if any(pathlib.Path(values[key]).resolve() == data_path.resolve() or
           data_path.resolve() in pathlib.Path(values[key]).resolve().parents for key in ('ARB_BOT_ENV','ARB_POSTGRES_ENV')): raise ValueError()
    if evidence_path.exists() or not evidence_path.parent.is_dir(): raise ValueError()
    selected = pathlib.Path(values['ARB_BUNDLE_DIR'])
    if not data_path.is_dir() or unsafe(selected) or data_path.resolve() not in selected.resolve().parents:
        raise ValueError()
    if any(unsafe(data_path/part) for part in ('raw','work','bundles')): raise ValueError()
    git_common = subprocess.run(['git','-c',f'safe.directory={repo}',
                                 'rev-parse','--path-format=absolute','--git-common-dir'],
                                cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
    stable_root = pathlib.Path(git_common).resolve().parent
    if stable_root not in data_path.resolve().parents: raise ValueError()
    if stable_root not in evidence_path.resolve().parents: raise ValueError()
    for key in ('ARB_UID','ARB_GID','ARB_BUNDLE_DIR'):
        sys.stdout.buffer.write(values[key].encode()+b'\0')
except (OSError, ValueError, KeyError):
    raise SystemExit('FAIL unsafe_settings')
PY
)
((${#settings_values[@]} == 3)) || { echo 'FAIL unsafe_settings' >&2; exit 1; }
operator_uid=${settings_values[0]}
operator_gid=${settings_values[1]}
[[ $(stat -c %u "$data_dir") == "$operator_uid" && $(stat -c %g "$data_dir") == "$operator_gid" ]] || { echo 'FAIL data_owner' >&2; exit 1; }
[[ -d "$data_dir/raw" && $(stat -c %u "$data_dir/raw") == "$operator_uid" &&
   $(stat -c %g "$data_dir/raw") == "$operator_gid" ]] || { echo 'FAIL raw_owner' >&2; exit 1; }
[[ $(stat -c %a "$(dirname "$evidence_dir")") == 700 ]] || { echo 'FAIL private_parent' >&2; exit 1; }
mkdir -m 700 -- "$evidence_dir"
mkdir -m 700 -- "$evidence_dir/logs" "$evidence_dir/snapshots"
fail() { printf 'FAIL %s; private evidence retained\n' "$1" >&2; exit 1; }
compose() { docker compose --env-file "$settings" -p "$project" -f "$repo/compose.yaml" "$@"; }
phase() {
    local label=$1; shift
    "$@" >"$evidence_dir/logs/$label.stdout" 2>"$evidence_dir/logs/$label.stderr" || fail "$label"
    chmod 600 "$evidence_dir/logs/$label.stdout" "$evidence_dir/logs/$label.stderr"
}
bot_id=$(compose ps -a -q bot 2>"$evidence_dir/logs/compose-ps.stderr") || fail 'compose_ps'
[[ -z $bot_id ]] || {
    [[ $(docker inspect --format '{{.State.Running}}' "$bot_id") == false ]] || fail 'bot_already_running'
}
phase build-images compose --profile tools build builder bot
builder() { compose --profile tools run --rm --no-deps builder "$@"; }
if ((offline)); then phase obtain builder obtain --offline --data-dir /data/raw
else phase obtain builder obtain --data-dir /data/raw; fi
phase normalize builder normalize --anime-csv /data/raw/anime.csv --synopsis-csv /data/raw/anime_with_synopsis.csv --output-dir /data/work/arb020-normalized
for label in A B; do
    block=256; [[ $label == B ]] && block=128
    phase "build-$label" builder build --catalog /data/work/arb020-normalized/catalog.json \
        --normalization-report /data/work/arb020-normalized/normalization-report.json \
        --glove /data/raw/glove.6B.300d.txt --output-dir "/data/work/arb020-build-$label" --block-size "$block"
    phase "export-$label" builder export --catalog /data/work/arb020-normalized/catalog.json \
        --neighbors "/data/work/arb020-build-$label/neighbors.json" \
        --build-report "/data/work/arb020-build-$label/build-report.json" \
        --normalization-report /data/work/arb020-normalized/normalization-report.json \
        --output-dir /data/bundles
    export_line=$(tail -n 1 "$evidence_dir/logs/export-$label.stdout")
    [[ $export_line =~ ^/data/bundles/sha256-([0-9a-f]{64})\ sha256:([0-9a-f]{64})\ records=17562$ ]] || fail "export-$label-format"
    [[ ${BASH_REMATCH[1]} == ${BASH_REMATCH[2]} ]] || fail "export-$label-identity"
    bundle="$data_dir/bundles/sha256-${BASH_REMATCH[1]}"
    [[ -d $bundle && ! -L $bundle ]] || fail "export-$label-path"
    printf '%s\n' "$bundle" >"$evidence_dir/bundle-$label.path"
    chmod 600 "$evidence_dir/bundle-$label.path"
    phase "validate-$label" builder validate "/data/bundles/sha256-${BASH_REMATCH[1]}"
    [[ $(tail -n 1 "$evidence_dir/logs/validate-$label.stdout") == "sha256:${BASH_REMATCH[1]} records=17562" ]] || fail "validate-$label-output"
    chmod 755 "$bundle" || fail "publish-$label"
    chmod 644 "$bundle/catalog.json" "$bundle/neighbors.json" "$bundle/manifest.json" || fail "publish-$label"
done
bundle_a=$(<"$evidence_dir/bundle-A.path")
bundle_b=$(<"$evidence_dir/bundle-B.path")
[[ $bundle_a != "$bundle_b" ]] || fail 'bundle_identity_same'
python3 - "$data_dir" "$bundle_a" "$bundle_b" "$evidence_dir" <<'PY' 2>"$evidence_dir/logs/artifact-provenance.stderr" || fail 'artifact_provenance'
import hashlib, importlib.util, json, pathlib, sys
root = pathlib.Path.cwd(); data, a, b, evidence = map(pathlib.Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('checker', root/'contracts/check_bundle.py')
checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = json.loads((data/'raw/source-manifest.json').read_bytes())
registry = root/'recsys/src/recsys/source_registry.json'
assert source['registry_sha256'] == h(registry)
assert all((data/'raw'/item['filename']).stat().st_size == item['size'] and
           h(data/'raw'/item['filename']) == item['sha256'] for item in source['verified'])
norm = json.loads((data/'work/arb020-normalized/normalization-report.json').read_bytes())
assert norm['registry_sha256'] == h(registry)
assert norm['counts']['retained'] == 17562
reports = []
for label, bundle, block in [('A',a,256),('B',b,128)]:
    report_file = data/f'work/arb020-build-{label}/build-report.json'
    report = json.loads(report_file.read_bytes()); reports.append(report)
    ident, count = checker.check_bundle(bundle)
    assert ident == 'sha256:'+bundle.name[7:] and count == 17562
    assert report['execution']['requested_block_rows'] == block
    assert report['execution']['effective_block_rows'] == block
    assert report['producer']['python_version'] == '3.12.14'
    assert report['producer']['numpy_version'] == '2.5.3'
    assert report['registry_sha256'] == h(registry)
    assert report['inputs']['catalog.json']['sha256'] == h(bundle/'catalog.json')
    assert report['inputs']['normalization-report.json']['sha256'] == h(data/'work/arb020-normalized/normalization-report.json')
    assert report['inputs']['glove.6B.300d.txt']['sha256'] == h(data/'raw/glove.6B.300d.txt')
    assert report['files']['neighbors.json']['sha256'] == h(bundle/'neighbors.json')
    manifest = json.loads((bundle/'manifest.json').read_bytes())
    assert manifest['files'] == report['files']
    assert any(item['name']=='build_report' and item['sha256']==h(report_file) for item in manifest['sources'])
    assert (data/'work/arb020-normalized/catalog.json').read_bytes() == (bundle/'catalog.json').read_bytes()
    neighbors = json.loads((bundle/'neighbors.json').read_bytes())['neighbors']
    assert [r['mal_id'] for r in neighbors['1']] == [400,709,4088,329,1303]
    assert neighbors['40089'] == []
    assert sum(len(v)==5 for v in neighbors.values()) == 17561
    assert sum(len(v)==0 for v in neighbors.values()) == 1
    target = evidence/f'build-report-{label}.json'
    target.write_bytes(report_file.read_bytes()); target.chmod(0o600)
assert (a/'catalog.json').read_bytes() == (b/'catalog.json').read_bytes()
assert (a/'neighbors.json').read_bytes() == (b/'neighbors.json').read_bytes()
left,right=reports
left.pop('execution'); right.pop('execution')
assert left == right
(evidence/'normalization-report.json').write_bytes((data/'work/arb020-normalized/normalization-report.json').read_bytes())
(evidence/'normalization-report.json').chmod(0o600)
PY
# Change only ARB_BUNDLE_DIR. Atomic replace keeps exact other settings and file ownership.
select_bundle() {
    python3 - "$settings" "$1" <<'PY'
import os, pathlib, stat, sys, tempfile
path, selected = map(pathlib.Path, sys.argv[1:])
lines = path.read_text().splitlines(keepends=True)
assert sum(line.startswith('ARB_BUNDLE_DIR=') for line in lines) == 1
assert selected.is_dir() and not selected.is_symlink()
value = ''.join('ARB_BUNDLE_DIR='+str(selected)+'\n' if line.startswith('ARB_BUNDLE_DIR=') else line for line in lines)
old = path.stat()
fd, temporary = tempfile.mkstemp(prefix='.arb020-settings-', dir=path.parent)
try:
    with os.fdopen(fd,'w') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    os.chown(temporary, old.st_uid, old.st_gid)
    os.chmod(temporary, stat.S_IMODE(old.st_mode))
    os.replace(temporary,path)
finally:
    if os.path.exists(temporary): os.unlink(temporary)
PY
}
select_bundle "$bundle_a"
phase check-rust-a compose run --rm --no-deps --entrypoint /usr/local/bin/check_bundle prepare /artifacts
[[ $(tail -n 1 "$evidence_dir/logs/check-rust-a.stdout") == "sha256:${bundle_a##*sha256-} records=17562" ]] || fail 'checker_disagreement'
phase postgres compose up -d --wait postgres
wait_prepare() {
    local id state code
    for _ in {1..120}; do
        id=$(compose ps -a -q prepare)
        if [[ -n $id ]]; then
            read -r state code < <(docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$id")
            [[ $state != exited ]] || { [[ $code == 0 ]] && return 0 || fail 'prepare_failed'; }
        fi
        sleep 2
    done
    fail 'prepare_timeout'
}
sequence=0
for label in A B A; do
    sequence=$((sequence + 1))
    bundle=$bundle_a; [[ $label == B ]] && bundle=$bundle_b
    select_bundle "$bundle"
    phase "check-rust-$sequence-$label" compose run --rm --no-deps --entrypoint /usr/local/bin/check_bundle prepare /artifacts
    [[ $(tail -n 1 "$evidence_dir/logs/check-rust-$sequence-$label.stdout") == "sha256:${bundle##*sha256-} records=17562" ]] || fail 'checker_disagreement'
    phase "prepare-$sequence-$label" compose up -d --no-deps --force-recreate prepare
    wait_prepare
    python3 deploy/tests/acceptance-evidence.py snapshot --project "$project" --settings "$settings" \
        --output "$evidence_dir/snapshots/prepare-$sequence-$label.json" >/dev/null || fail "snapshot-$label"
done
python3 - "$evidence_dir" "$data_dir" "$project" "$bundle_a" "$bundle_b" "$settings" "$repo" <<'PY' 2>"$evidence_dir/logs/prepare-invariants.stderr" || fail 'prepare_invariants'
import hashlib, json, pathlib, subprocess, sys
folder,data,project,a,b,settings,repo = map(pathlib.Path,sys.argv[1:])
shots=[folder/'snapshots'/name for name in ('prepare-1-A.json','prepare-2-B.json','prepare-3-A.json')]
assert len(shots)==3
items=[json.loads(p.read_bytes()) for p in shots]
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
volume=items[0]['runtime']['postgres']['database_volume']
assert volume and items[0]['database']==items[1]['database']==items[2]['database']
assert all(s['runtime']['postgres']['database_volume']==volume and s['runtime']['postgres']['running'] for s in items)
assert all(s['runtime']['bot'] is None or not s['runtime']['bot']['running'] for s in items)
for item, bundle in zip(items,(a,b,a)):
    prepare=item['runtime']['prepare']
    assert prepare['status']=='exited' and prepare['exit_code']==0
    assert prepare['bundle_id']=='sha256:'+bundle.name[7:] and prepare['bundle_count']==17562
    assert prepare['user']=='10001:10001' and prepare['read_only_root']
    assert prepare['artifact_mount']['source']==str(bundle) and not prepare['artifact_mount']['rw']
assert items[0]['database']['migrations'][0]['version']==1 and len(items[0]['database']['migrations'])==1
compose_cmd=['docker','compose','--env-file',str(settings),'-p',str(project),'-f',str(repo/'compose.yaml')]
compose_config=json.loads(subprocess.run(compose_cmd+['config','--format','json'],
                                     capture_output=True,text=True,check=True).stdout)
image_names={key:compose_config['services'][key]['image'] for key in ('builder','bot')}
image_ids={key:subprocess.run(['docker','image','inspect','--format','{{.Id}}',image_names[key]],
                              capture_output=True,text=True,check=True).stdout.strip()
           for key in image_names}
assert image_ids['bot']==items[-1]['runtime']['prepare']['image_id']
source_manifest=json.loads((data/'raw/source-manifest.json').read_bytes())
report={'schema_version':1,'project':str(project),'source_manifest_sha256':h(data/'raw/source-manifest.json'),
        'source_commit':subprocess.run(['git','-c',f'safe.directory={repo}','rev-parse','HEAD'],
                                       cwd=repo,capture_output=True,text=True,check=True).stdout.strip(),
        'source_files':{item['filename']:{'size':item['size'],'sha256':h(data/'raw'/item['filename'])}
                        for item in source_manifest['verified']},
        'tool_sha256':{str(name):h(repo/name) for name in
                       (pathlib.Path('deploy/tests/full-catalog-acceptance.sh'),
                        pathlib.Path('deploy/tests/acceptance-evidence.py'),
                        pathlib.Path('contracts/check_bundle.py'))},
        'registry_sha256':h(repo/'recsys/src/recsys/source_registry.json'),
        'compose_sha256':h(repo/'compose.yaml'),'settings_sha256':h(settings),
        'normalization_report_sha256':h(folder/'normalization-report.json'),
        'database_volume':volume,'prepare_snapshots_sha256':[h(p) for p in shots],
        'bundles':{label:{'identity':'sha256:'+bundle.name[7:],'path':str(bundle),
                          'manifest_sha256':h(bundle/'manifest.json'),
                          'catalog_sha256':h(bundle/'catalog.json'),
                          'neighbors_sha256':h(bundle/'neighbors.json'),
                          'build_report_sha256':h(folder/f'build-report-{label}.json')}
                   for label,bundle in (('A',a),('B',b))},
        'images':{**image_ids,'postgres':items[-1]['runtime']['postgres']['image_id'],
                  'prepare':items[-1]['runtime']['prepare']['image_id']},
        'python_producer':'3.12.14','numpy_producer':'2.5.3'}
path=folder/'build-report.json'
with path.open('x') as stream: json.dump(report,stream,ensure_ascii=False,sort_keys=True,separators=(',',':'))
path.chmod(0o600)
PY
echo 'PASS full-catalog preparation; PostgreSQL retained, A selected, prepare exited 0, bot stopped'
