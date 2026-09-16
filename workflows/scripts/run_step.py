"""Snakemake wrapper to run individual steps."""

from cedalion_workflows import resolve, bind_config

if "snakemake" not in globals():
    raise RuntimeError("This script must be run through Snakemake.")

func = resolve(snakemake.params.func)                 # noqa: F821 (Snakemake-injected)

cfg = dict(snakemake.params.config)                   # noqa: F821  algorithmic params
cfg.update(dict(snakemake.input.items()))             # noqa: F821  named inputs  -> args
cfg.update(dict(snakemake.output.items()))            # noqa: F821  named outputs -> args

bound = bind_config(func, cfg)                        # validate against the signature
func(*bound.args, **bound.kwargs)
