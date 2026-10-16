#!/usr/bin/env python3
"""Check the recsys import graph without importing or executing recsys."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


IDENT = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
FIELDS = {"schema_version", "subsystems", "exceptions", "existing_cycles", "dynamic_dispatch"}
SUBSYSTEM_FIELDS = {"modules", "depends_on"}
EXCEPTION_FIELDS = {"source", "target", "names", "reason"}
CYCLE_FIELDS = {"members", "edges", "reason"}
DISPATCH_FIELDS = {"source", "function", "registry", "source_sha256", "reason"}
BACKEND_VARIANTS = {"popularity", "content-mean", "content-max", "content-union",
                    "itemknn", "puresvd", "hybrid"}
BACKEND_MODULES = {"research_model_popularity", "research_model_content",
                   "research_model_cf", "research_model_hybrid"}


class PolicyError(ValueError):
    pass


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise PolicyError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _fields(value: object, expected: set[str], where: str) -> dict:
    if type(value) is not dict or set(value) != expected:
        raise PolicyError(f"{where}: expected fields {sorted(expected)}")
    return value


def _identifier(value: object, where: str, *, module: bool = False) -> str:
    parts = value.split(".") if type(value) is str and module else [value]
    if not parts or any(type(p) is not str or not IDENT.fullmatch(p) for p in parts):
        raise PolicyError(f"{where}: invalid identifier")
    if module and parts[0] != "recsys":
        raise PolicyError(f"{where}: expected recsys module")
    return value


def _list(value: object, where: str, *, nonempty: bool = False) -> list:
    if type(value) is not list or (nonempty and not value):
        raise PolicyError(f"{where}: expected {'nonempty ' if nonempty else ''}list")
    return value


def _unique(items: list, where: str) -> None:
    if len(items) != len(set(items)):
        raise PolicyError(f"{where}: duplicate value")


def _reason(value: object, where: str) -> None:
    if type(value) is not str or not value.strip():
        raise PolicyError(f"{where}: empty reason")


def _policy(path: Path) -> tuple[dict, dict[str, str]]:
    policy = _fields(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs), FIELDS, "policy")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise PolicyError("schema_version must be 1")
    subsystems = policy["subsystems"]
    if type(subsystems) is not dict or not subsystems:
        raise PolicyError("subsystems must be a nonempty object")
    owners: dict[str, str] = {}
    for group, raw in subsystems.items():
        _identifier(group, "subsystem")
        item = _fields(raw, SUBSYSTEM_FIELDS, group)
        modules = _list(item["modules"], f"{group}.modules", nonempty=True)
        deps = _list(item["depends_on"], f"{group}.depends_on")
        for name in modules:
            _identifier(name, f"{group}.modules", module=True)
            if name in owners:
                raise PolicyError(f"duplicate module owner: {name}")
            owners[name] = group
        for dep in deps:
            _identifier(dep, f"{group}.depends_on")
        _unique(deps, f"{group}.depends_on")
        if group in deps or any(dep not in subsystems for dep in deps):
            raise PolicyError(f"{group}: self or unknown dependency")
    exceptions = _list(policy["exceptions"], "exceptions")
    seen_exceptions = set()
    for index, raw in enumerate(exceptions):
        item = _fields(raw, EXCEPTION_FIELDS, f"exceptions[{index}]")
        src = _identifier(item["source"], "exception.source", module=True)
        dst = _identifier(item["target"], "exception.target", module=True)
        names = _list(item["names"], "exception.names", nonempty=True)
        for name in names:
            _identifier(name, "exception.name")
        _unique(names, "exception.names")
        _reason(item["reason"], "exception")
        if src not in owners or dst not in owners or src == dst:
            raise PolicyError("exception references unknown or identical modules")
        if owners[src] == owners[dst] or owners[dst] in subsystems[owners[src]]["depends_on"]:
            raise PolicyError(f"redundant exception: {src} -> {dst}")
        if (src, dst) in seen_exceptions:
            raise PolicyError(f"duplicate exception: {src} -> {dst}")
        seen_exceptions.add((src, dst))
    cycles = _list(policy["existing_cycles"], "existing_cycles")
    seen_cycles = set()
    for index, raw in enumerate(cycles):
        item = _fields(raw, CYCLE_FIELDS, f"existing_cycles[{index}]")
        members = _list(item["members"], "cycle.members", nonempty=True)
        edges = _list(item["edges"], "cycle.edges", nonempty=True)
        _reason(item["reason"], "cycle")
        for member in members:
            _identifier(member, "cycle.member", module=True)
            if member not in owners or member in seen_cycles:
                raise PolicyError(f"duplicate or unknown cycle member: {member}")
            seen_cycles.add(member)
        _unique(members, "cycle.members")
        edge_set = set()
        for edge in edges:
            if type(edge) is not list or len(edge) != 2 or any(type(v) is not str for v in edge):
                raise PolicyError("cycle.edge: expected source/target pair")
            src, dst = edge
            if src not in members or dst not in members or (src, dst) in edge_set:
                raise PolicyError("cycle.edge: duplicate or external endpoint")
            edge_set.add((src, dst))
    dispatch = _fields(policy["dynamic_dispatch"], DISPATCH_FIELDS, "dynamic_dispatch")
    for key in ("source", "function", "registry"):
        _identifier(dispatch[key], f"dynamic_dispatch.{key}", module=key == "source")
    if dispatch["source"] not in owners:
        raise PolicyError("dynamic_dispatch.source unknown")
    digest = dispatch["source_sha256"]
    if type(digest) is not str or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise PolicyError("dynamic_dispatch.source_sha256 invalid")
    _reason(dispatch["reason"], "dynamic_dispatch")
    return policy, owners


def _inventory(package: Path) -> dict[str, Path]:
    if package.is_symlink() or not package.is_dir():
        raise PolicyError("package root missing or symlink")
    result = {}
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise PolicyError(f"symlink in package: {path.relative_to(package)}")
        if path.suffix != ".py":
            continue
        rel = path.relative_to(package)
        if path.name == "__init__.py":
            name = ".".join(("recsys", *rel.parts[:-1]))
        else:
            name = ".".join(("recsys", *rel.with_suffix("").parts))
        if name in result:
            raise PolicyError(f"module/package collision for {name}: "
                              f"{result[name].relative_to(package)} and {rel}")
        result[name] = path
    return result


def _attribute_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _resolve_relative(package: str, module: str | None, level: int) -> str:
    if level == 0:
        return module or ""
    parts = package.split(".")
    if level > len(parts):
        return ""
    return ".".join(parts[: len(parts) - level + 1] + ([module] if module else []))


def _literal_loader(node: ast.Call, kind: str, package: str) -> str | None:
    if kind == "builtin":
        if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Constant):
            return None
        value = node.args[0].value
        return value if type(value) is str else None
    if len(node.args) not in (1, 2) or node.keywords or not isinstance(node.args[0], ast.Constant):
        return None
    value = node.args[0].value
    if type(value) is not str:
        return None
    if value.startswith("."):
        if len(node.args) != 2:
            return None
        base = node.args[1]
        if isinstance(base, ast.Name) and base.id == "__package__":
            anchor = package
        elif isinstance(base, ast.Constant) and type(base.value) is str:
            anchor = base.value
        else:
            return None
        levels = len(value) - len(value.lstrip("."))
        parts = anchor.split(".")
        if levels > len(parts):
            return ""
        suffix = value[levels:]
        return ".".join(parts[: len(parts) - levels + 1] + ([suffix] if suffix else []))
    if len(node.args) != 1:
        return None
    return value


def _dispatch_targets(tree: ast.Module, policy: dict, raw: bytes) -> set[str]:
    cfg = policy["dynamic_dispatch"]
    if hashlib.sha256(raw).hexdigest() != cfg["source_sha256"]:
        raise PolicyError("dynamic_dispatch_changed: audited source SHA-256 changed")
    assignments = [node for node in tree.body if isinstance(node, ast.Assign) and
                   len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and
                   node.targets[0].id == cfg["registry"]]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == cfg["function"]]
    if len(assignments) != 1 or len(functions) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise PolicyError("dynamic_dispatch_changed: registry or function shape changed")
    registry = assignments[0].value
    keys, values = [], []
    for key, value in zip(registry.keys, registry.values):
        if not isinstance(key, ast.Constant) or type(key.value) is not str or not isinstance(value, ast.Constant) or type(value.value) is not str:
            raise PolicyError("dynamic_dispatch_changed: registry must be a literal string dictionary")
        _identifier(value.value, "dynamic_dispatch.registry value")
        keys.append(key.value)
        values.append(value.value)
    if (len(keys) != len(set(keys)) or set(keys) != BACKEND_VARIANTS or
            set(values) != BACKEND_MODULES):
        raise PolicyError("dynamic_dispatch_changed: expected seven approved variants and four backend modules")
    calls = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"]
    if len(calls) != 1:
        raise PolicyError("dynamic_dispatch_changed: expected one loader call in backend function")
    call = calls[0]
    loader_aliases = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.Import)
                      for alias in node.names if alias.name == "importlib"}
    if not isinstance(call.func.value, ast.Name) or call.func.value.id not in loader_aliases:
        raise PolicyError("dynamic_dispatch_changed: loader is not imported importlib")
    if len(call.args) != 2 or call.keywords or not isinstance(call.args[1], ast.Name) or call.args[1].id != "__package__":
        raise PolicyError("dynamic_dispatch_changed: loader package argument changed")
    expression = call.args[0]
    if not (isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add) and
            isinstance(expression.left, ast.Constant) and expression.left.value == "." and
            isinstance(expression.right, ast.Subscript) and isinstance(expression.right.value, ast.Name) and
            expression.right.value.id == cfg["registry"] and isinstance(expression.right.slice, ast.Name) and
            expression.right.slice.id == "variant_id"):
        raise PolicyError("dynamic_dispatch_changed: loader expression changed")
    return {"recsys." + name for name in values}


def _scc(nodes: set[str], edges: set[tuple[str, str]]) -> list[set[str]]:
    graph = {node: set() for node in nodes}
    for src, dst in edges:
        graph[src].add(dst)
    index = 0
    stack: list[str] = []
    active: set[str] = set()
    numbers: dict[str, int] = {}
    lows: dict[str, int] = {}
    result: list[set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        numbers[node] = lows[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for child in sorted(graph[node]):
            if child not in numbers:
                visit(child)
                lows[node] = min(lows[node], lows[child])
            elif child in active:
                lows[node] = min(lows[node], numbers[child])
        if lows[node] == numbers[node]:
            component = set()
            while True:
                child = stack.pop()
                active.remove(child)
                component.add(child)
                if child == node:
                    break
            result.append(component)

    for node in sorted(nodes):
        if node not in numbers:
            visit(node)
    return result


def check_repository(root: Path, stats: dict[str, int] | None = None) -> list[str]:
    """Return stable diagnostics; source is parsed only, never imported."""
    errors: list[str] = []
    try:
        if not root.is_absolute():
            raise PolicyError("root must be absolute")
        policy, owners = _policy(root / "scripts/module_boundaries.json")
        inventory = _inventory(root / "src/recsys")
    except (OSError, ValueError, PolicyError) as exc:
        return [f"config_invalid: {exc}"]
    if stats is not None:
        stats.update(modules=len(inventory), subsystems=len(policy["subsystems"]))
    for name in sorted(set(inventory) ^ set(owners)):
        errors.append(f"inventory_mismatch: {name}: {'undeclared source' if name in inventory else 'missing source'}")
    trees: dict[str, ast.Module] = {}
    raw_sources: dict[str, bytes] = {}
    for name, path in sorted(inventory.items()):
        try:
            raw = path.read_bytes()
            trees[name] = ast.parse(raw, filename=str(path))
            raw_sources[name] = raw
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f"config_invalid: {path.relative_to(root)}: {exc}")
    if errors:
        return sorted(errors)
    package_attrs = {name: _attribute_names(tree) for name, tree in trees.items()
                     if inventory[name].name == "__init__.py"}
    exceptions = {(item["source"], item["target"]): set(item["names"]) for item in policy["exceptions"]}
    used_exceptions: dict[tuple[str, str], set[str]] = {key: set() for key in exceptions}
    edges: set[tuple[str, str]] = set()
    cfg = policy["dynamic_dispatch"]
    try:
        targets = _dispatch_targets(trees[cfg["source"]], policy, raw_sources[cfg["source"]])
    except (KeyError, PolicyError) as exc:
        errors.append(f"dynamic_dispatch_changed: {exc}")
        targets = set()

    for source, tree in sorted(trees.items()):
        rel = inventory[source].relative_to(root)
        package = source if inventory[source].name == "__init__.py" else source.rpartition(".")[0]
        imports = set()
        builtins_roots = set()
        loaders = set()
        builtin_loaders = {"__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Without `as`, `import importlib.util` binds the root
                    # `importlib`, while `import importlib.util as util` binds
                    # the submodule to `util`.
                    if alias.name == "importlib" or (alias.name.startswith("importlib.") and not alias.asname):
                        imports.add(alias.asname or "importlib")
                    elif alias.name == "builtins" or (alias.name.startswith("builtins.") and not alias.asname):
                        builtins_roots.add(alias.asname or "builtins")
            elif isinstance(node, ast.ImportFrom) and node.module == "importlib" and node.level == 0:
                for alias in node.names:
                    if alias.name == "import_module":
                        loaders.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "builtins" and node.level == 0:
                for alias in node.names:
                    if alias.name == "__import__":
                        builtin_loaders.add(alias.asname or alias.name)
        # A direct local alias is common in optional imports. No general dataflow
        # or evaluation is attempted: only this statically visible assignment.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            value = node.value
            if isinstance(value, ast.Name) and value.id in loaders | builtin_loaders:
                loaders.add(node.targets[0].id)
                if value.id in builtin_loaders:
                    builtin_loaders.add(node.targets[0].id)
            elif (isinstance(value, ast.Attribute) and value.attr == "import_module" and
                  isinstance(value.value, ast.Name) and value.value.id in imports):
                loaders.add(node.targets[0].id)
            elif (isinstance(value, ast.Attribute) and value.attr == "__import__" and
                  isinstance(value.value, ast.Name) and value.value.id in builtins_roots):
                builtin_loaders.add(node.targets[0].id)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        def loader_reference(node: ast.AST) -> bool:
            return (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and
                    node.id in loaders | builtin_loaders) or (
                isinstance(node, ast.Attribute) and node.attr == "import_module" and
                isinstance(node.value, ast.Name) and node.value.id in imports) or (
                isinstance(node, ast.Attribute) and node.attr == "__import__" and
                isinstance(node.value, ast.Name) and node.value.id in builtins_roots)
        def add(target: str, node: ast.AST, names: set[str] | None = None) -> None:
            if not target.startswith("recsys"):
                return
            location = f"{rel}:{node.lineno} {source} -> {target}"
            if target not in inventory:
                errors.append(f"unknown_import: {location}: missing internal module")
                return
            edges.add((source, target))
            if source == target:
                return
            if source not in owners:
                return
            src_group, dst_group = owners[source], owners[target]
            if dst_group == src_group or dst_group in policy["subsystems"][src_group]["depends_on"]:
                return
            permitted = exceptions.get((source, target))
            if permitted is not None and names is not None and names <= permitted:
                used_exceptions[(source, target)].update(names)
                return
            errors.append(f"forbidden_dependency: {location}: {src_group} cannot import {dst_group}" +
                          (" without listed symbols" if permitted is not None else ""))

        for node in ast.walk(tree):
            if loader_reference(node):
                parent = parents.get(node)
                direct_call = isinstance(parent, ast.Call) and parent.func is node
                direct_alias = (isinstance(parent, ast.Assign) and parent.value is node and
                                len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name))
                if not (direct_call or direct_alias):
                    errors.append(f"unsupported_dynamic_import: {rel}:{node.lineno} {source}: loader callable used as value")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "recsys" or alias.name.startswith("recsys."):
                        add(alias.name, node)
            elif isinstance(node, ast.ImportFrom):
                target = _resolve_relative(package, node.module, node.level)
                if not target.startswith("recsys") and not (node.level and not target):
                    continue
                if not target or (target != "recsys" and not target.startswith("recsys.")):
                    errors.append(f"unknown_import: {rel}:{node.lineno} {source}: relative import escapes recsys")
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        errors.append(f"unknown_import: {rel}:{node.lineno} {source} -> {target}: internal wildcard")
                        continue
                    if target not in inventory:
                        add(target, node)
                    elif target in package_attrs:
                        child = target + "." + alias.name
                        if child in inventory:
                            add(child, node)
                        elif alias.name in package_attrs[target]:
                            add(target, node, {alias.name})
                        else:
                            errors.append(f"unknown_import: {rel}:{node.lineno} {source} -> {child}: unknown package member")
                    else:
                        add(target, node, {alias.name})
            elif isinstance(node, ast.Call):
                kind = None
                if isinstance(node.func, ast.Name) and node.func.id in builtin_loaders:
                    kind = "builtin"
                elif (isinstance(node.func, ast.Attribute) and node.func.attr == "__import__" and
                      isinstance(node.func.value, ast.Name) and node.func.value.id in builtins_roots):
                    kind = "builtin"
                elif isinstance(node.func, ast.Name) and node.func.id in loaders:
                    kind = "module"
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module" and isinstance(node.func.value, ast.Name) and node.func.value.id in imports:
                    kind = "module"
                if kind is None:
                    continue
                target = _literal_loader(node, kind, package)
                if target is not None:
                    if target == "":
                        errors.append(f"unknown_import: {rel}:{node.lineno} {source}: relative dynamic import escapes recsys")
                    elif target == "recsys" or target.startswith("recsys."):
                        add(target, node)
                    continue
                audited = (source == cfg["source"] and targets and
                           isinstance(node.func, ast.Attribute) and
                           isinstance(node.func.value, ast.Name) and node.func.value.id in imports and
                           any(node is candidate for func in tree.body
                               if isinstance(func, ast.FunctionDef) and func.name == cfg["function"]
                               for candidate in ast.walk(func)))
                if audited:
                    for target in sorted(targets):
                        add(target, node)
                else:
                    errors.append(f"unsupported_dynamic_import: {rel}:{node.lineno} {source}: computed loader call")
    for (source, target), names in sorted(used_exceptions.items()):
        missing = exceptions[(source, target)] - names
        if missing:
            errors.append(f"stale_exception: {source} -> {target}: unused symbols {','.join(sorted(missing))}")
    actual_cycles: dict[frozenset[str], set[tuple[str, str]]] = {}
    for component in _scc(set(inventory), edges):
        internal = {(src, dst) for src, dst in edges if src in component and dst in component}
        if len(component) > 1 or any(src == dst for src, dst in internal):
            actual_cycles[frozenset(component)] = internal
    expected_cycles = {frozenset(item["members"]): {tuple(edge) for edge in item["edges"]}
                       for item in policy["existing_cycles"]}
    for members, expected in expected_cycles.items():
        actual = actual_cycles.get(members)
        if actual is None:
            errors.append(f"stale_exception: cycle {','.join(sorted(members))}: component no longer present")
        elif actual != expected:
            added = sorted(actual - expected)
            removed = sorted(expected - actual)
            errors.append(f"new_cycle: {','.join(sorted(members))}: added={added} missing={removed}")
    for members in actual_cycles.keys() - expected_cycles.keys():
        errors.append(f"new_cycle: unapproved component {','.join(sorted(members))}")
    if stats is not None:
        stats.update(edges=len(edges), cycles=len(actual_cycles))
    return sorted(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    stats: dict[str, int] = {}
    errors = check_repository(args.root, stats)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"PASS Python module boundaries: {stats['modules']} modules, "
          f"{stats['subsystems']} subsystems, {stats['edges']} imports, {stats['cycles']} existing cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
