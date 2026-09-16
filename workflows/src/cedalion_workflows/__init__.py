from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import itertools
import json
import warnings
from dataclasses import dataclass, field
from importlib.metadata import entry_points as _entry_points
from typing import Any, Callable, Iterable, Iterator
import re

DEFAULT_GROUP = "cedalion.steps"



# Discovery & resolution
def _eps(group: str):
    """entry_points(group=...) across 3.9 (dict API) and 3.10+ (selectable)."""
    try:
        return list(_entry_points(group=group))  # Python 3.10+
    except TypeError:  # Python 3.9
        return list(_entry_points().get(group, []))


def discover_steps(group: str = DEFAULT_GROUP) -> dict[str, Any]:
    """Map step-name -> EntryPoint by reading installed-package metadata.

    Does NOT import the target modules (use .load() for that). Reflects exactly
    what is installed in the *current* environment, so each per-rule process
    sees its own steps and nothing leaks across environments.
    """
    return {ep.name: ep for ep in _eps(group)}


def resolve(spec: str, group: str = DEFAULT_GROUP) -> Callable:
    """Resolve a step to a callable.

    `spec` is either
      * an entry-point name registered under `group`  ('name_of_registered_func'), or
      * an explicit dotted path                       ('pkg.module:func').

    A spec containing ':' is always treated as a dotted path. Otherwise it is
    looked up among registered entry points. This lets registered steps use
    short names while ad-hoc / unregistered functions stay fully addressable.
    """
    if ":" in spec:
        module_name, _, func_name = spec.partition(":")
        module = importlib.import_module(module_name)
        try:
            return getattr(module, func_name)
        except AttributeError as exc:
            raise ImportError(
                f"{module_name!r} has no attribute {func_name!r}"
            ) from exc

    steps = discover_steps(group)
    if spec in steps:
        return steps[spec].load()
    raise KeyError(
        f"{spec!r} is not a registered step in group {group!r}. "
        f"Known: {sorted(steps)}. "
        f"For an unregistered function use an explicit 'pkg.module:func' path."
    )



# Signature-driven validation & binding

def check_config(
    func: Callable,
    config: dict[str, Any],
    external: Iterable[str] = (),
) -> None:
    """Structural validation of `config` against `func`'s signature.

    Checks for unexpected keys and missing required arguments. Does NOT bind or
    call, so it is safe when some required arguments are supplied elsewhere
    (the `external` set) and are therefore absent from `config` — e.g. when
    validating a YAML section up front, before Snakemake has provided I/O paths.

    Type checking/coercion is the function's own concern (pydantic.validate_call,
    typeguard, ...) and fires when the function is actually called.
    """
    external = frozenset(external)
    params = inspect.signature(func).parameters
    accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())

    bindable = {
        n
        for n, p in params.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    required = {
        n
        for n, p in params.items()
        if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }

    errors = []
    unknown = set(config) - bindable
    if unknown and not accepts_kwargs:
        errors.append(f"unexpected keys: {sorted(unknown)}")
    missing = required - set(config) - external
    if missing:
        errors.append(f"missing required keys: {sorted(missing)}")

    if errors:
        sig_str = ", ".join(str(p) for p in params.values())
        raise ValueError(
            f"Invalid config for {func.__module__}.{func.__qualname__}:\n  "
            + "\n  ".join(errors)
            + f"\n  signature: ({sig_str})"
        )


def bind_config(
    func: Callable,
    config: dict[str, Any],
    external: Iterable[str] = (),
) -> inspect.BoundArguments:
    """check_config + produce bound arguments ready to call.

    Use at call time, when the config already contains every required argument
    (algorithmic params from YAML plus any `external` I/O merged in). Defaults
    are applied so YAML need only specify overrides.
    """
    check_config(func, config, external=external)
    sig = inspect.signature(func)
    params = sig.parameters
    accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    bindable = {
        n
        for n, p in params.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    usable = {k: v for k, v in config.items() if k in bindable or accepts_kwargs}
    bound = sig.bind(**usable)
    bound.apply_defaults()
    return bound


def call_from_config(
    func: Callable,
    config: dict[str, Any],
    external: Iterable[str] = (),
) -> Any:
    """bind_config + invoke."""
    bound = bind_config(func, config, external=external)
    return func(*bound.args, **bound.kwargs)


# Whole-config (launch-time) validation

def validate_bindings(
    config: dict[str, Any],
    bindings: dict[str, dict],
    group: str = DEFAULT_GROUP,
    strict: bool = False,
) -> None:
    """Best-effort validation of every configured step at launch time.

    `bindings` maps a config-section name to a dict with keys:
        func     : entry-point name or dotted path (required)
        external : iterable of runtime-supplied arg names (optional)

    Behaviour for a step whose library cannot be imported in the *current*
    (launcher) environment:
        strict=False -> warn and skip (it will be validated in-env by the
                        wrapper just before the call — the real backstop);
        strict=True  -> treat as a failure.

    Raises SystemExit listing all problems if any are found, so Snakemake aborts
    before scheduling jobs.
    """
    problems: dict[str, str] = {}
    for section, binding in bindings.items():
        func_spec = binding["func"]
        external = binding.get("external", ())
        params = config.get(section)
        if params is None:
            problems[section] = f"[{section}] missing config section"
            continue
        try:
            func = resolve(func_spec, group)
        except (ImportError, KeyError, ModuleNotFoundError) as exc:
            msg = (
                f"[{section}] cannot resolve {func_spec!r} in the launcher "
                f"environment: {exc}"
            )
            if strict:
                problems[section] = msg
            else:
                warnings.warn(msg + "  (skipped; validated in-env at run time)")
            continue
        try:
            check_config(func, dict(params), external=external)
        except ValueError as exc:
            problems[section] = str(exc)

    if problems:
        raise SystemExit("Config validation failed:\n" + "\n".join(problems.values()))


# ensemble support

_RESERVED = ("sweep", "scenarios")


@dataclass(frozen=True)
class Variant:
    id: str  # path-safe wildcard value
    params: dict[str, Any]  # base + overrides (what the step gets)
    varied_params: dict[str, Any] = field(default_factory=dict)  # only the varied keys


def _base_params(section: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in section.items() if k not in _RESERVED}


def _short_hash(d: dict[str, Any]) -> str:
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=4).hexdigest()  # 8 hex chars


_SEG = re.compile(r"([^.\[\]]+)|\[([^\]]+)\]")   # 'foo' -> key ;  '[bar]' -> name-selector

def _parse_path(path):
    # "steps[od_ff].params.fmin" -> [("key","steps"),("name","od_ff"),("key","params"),("key","fmin")]
    segs = []
    for m in _SEG.finditer(path):
        if m.group(1) is not None:
            segs.append(("key", m.group(1)))
        else:
            segs.append(("name", m.group(2)))
    return segs

def _walk_to_parent(root, segs):
    node = root
    for kind, val in segs[:-1]:
        if kind == "key":
            node = node[val]
        else:  # name-selector into a list of records
            matches = [e for e in node if isinstance(e, dict) and e.get("name") == val]
            if len(matches) != 1:
                raise KeyError(
                    f"name-selector [{val}] matched {len(matches)} elements "
                    "(need exactly 1)"
                )
            node = matches[0]
    return node



def _deep_set(d : dict, path : str, value):
    """Sets values in nested dicts with dotted key notation.

    _deep_set(d, "a.b.c", v) is equivalent to d["a"]["b"]["c"] = v
    """
    segs = _parse_path(path)
    parent = _walk_to_parent(d, segs)
    last_kind, last_val = segs[-1]
    if last_kind == "name":               # selecting a list element as the final target
        for e in parent:
            if isinstance(e, dict) and e.get("name") == last_val:
                e.clear(); e.update(value); return
        raise KeyError(f"name [{last_val}] not found at leaf")
    parent[last_val] = value              # ordinary key assignment


def ensemble(section: dict[str, Any]) -> list[Variant]:
    """Expand a config section into its list of ensemble members."""
    base_params = _base_params(section)
    sweep = section.get("sweep") or {}
    scenarios = section.get("scenarios") or {}

    if not sweep and not scenarios:
        return [Variant(id="base", params=base_params, varied_params={})]

    variants: list[Variant] = []

    for name, varied_params in scenarios.items():
        variants.append(
            Variant(
                id=f"{name}-{_short_hash(varied_params)}",
                params={**base_params, **varied_params},
                varied_params=dict(varied_params),
            )
        )

    if sweep:
        keys = list(sweep)
        for combo in itertools.product(*(sweep[k] for k in keys)):
            varied_params = dict(zip(keys, combo))
            params = copy.deepcopy(base_params)

            for dotted_key, value in varied_params.items():
                _deep_set(params, dotted_key, value)

            variants.append(
                Variant(
                    id=_short_hash(varied_params),
                    params=params,
                    varied_params=varied_params,
                )
            )
    return variants


def ensemble_ids(section: dict[str, Any]) -> list[str]:
    """Member ids, for ``expand(..., run=ensemble_ids(section))`` in targets."""
    return [m.id for m in ensemble(section)]


def variant_params(section: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Merged params for one member, for a rule's ``params.config`` function."""
    for m in ensemble(section):
        if m.id == run_id:
            return m.params
    raise KeyError(f"no ensemble member with id {run_id!r} in section {section!r}")


def manifest_rows(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Long-format provenance rows (step, run, param, value) for every section
    that declares an ensemble. Easy to load with pandas and pivot/join against
    your results when estimating the effect of each parameter."""
    for step, section in config.items():
        if not isinstance(section, dict):
            continue
        if not (section.get("sweep") or section.get("scenarios")):
            continue
        for variant in ensemble(section):
            if not variant.varied_params:  # the implicit base member
                yield {"step": step, "run": variant.id, "param": "", "value": ""}
            for k, v in variant.varied_params.items():
                yield {"step": step, "run": variant.id, "param": k, "value": v}
