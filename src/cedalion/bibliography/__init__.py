from pathlib import Path
from pybtex.database import parse_file, BibliographyData
from pybtex.plugin import find_plugin
from importlib.resources import files
from collections import defaultdict
import inspect

BIB_FILE = files("cedalion.bibliography") / "references.bib"
bib_data = parse_file(BIB_FILE)


class Bibliography:
    """Collects citation keys.

    Automatically deduplicates. Dumps as a formatted references section.
    """

    def __init__(self):
        self._refs = defaultdict(set)  # key → set of function names

    def cite(self, bibtex_key):
        # figure out who call us
        stack = inspect.stack()
        frame = stack[1][0]  # get the calling function
        if (
            frame.f_globals.get("__name__") == "cedalion"
            and frame.f_code.co_name == "cite"
        ):
            frame = stack[2][0]  # cedalion.cite wrapper? move up in the stack

        caller_name = frame.f_globals["__name__"] + "." + frame.f_code.co_qualname

        self._refs[bibtex_key].add(caller_name)

    def __len__(self):
        return len(self._refs)

    def clear(self):
        self._refs.clear()

    @property
    def keys(self) -> list[str]:
        return list(self._refs.keys())

    @staticmethod
    def format_entry(key, style_name="plain", backend_name="plaintext"):
        single = BibliographyData(entries={key: bib_data.entries[key]})
        style = find_plugin("pybtex.style.formatting", style_name)()
        backend = find_plugin("pybtex.backends", backend_name)()
        formatted = style.format_bibliography(single)
        return list(formatted)[0].text.render(backend)

    # --- flush targets ---

    def dump_to_string(self) -> str:
        lines = ["Methods & References", "=" * 40]
        for i, (key, fn_names) in enumerate(self._refs.items(), 1):
            fn_names = ", ".join(sorted(fn_names))
            rendered = self.format_entry(key)
            lines.append(f"[{i}] {key} — {fn_names}")
            lines.append(f"    {rendered}")
        return "\n".join(lines)

    def dump_to_file(self, path, mode="a", clear=False):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open(mode) as fout:
            fout.write(self.dump_to_string() + "\n")
        if clear:
            self.clear()

    def dump_to_notebook(self, title="Methods used", clear=False, show_functions=True):
        try:
            from IPython.display import HTML, display
        except ImportError:
            print(self.dump_to_string())
            return

        _st_num = "color:#888;padding-right:16px;white-space:nowrap;vertical-align:top"
        _st_key = (
            "font-family:monospace;color:#0d6efd;"
            "padding-right:16px;white-space:nowrap;vertical-align:top"
        )
        _st_lbl = (
            "padding-right:16px;vertical-align:top;word-break:break-word;width:50%"
        )
        _st_ref = (
            "color:#444;vertical-align:top;word-break:break-word;"
            + ("width:50%" if show_functions else "width:100%")
        )

        def merge_fn_names(fn_names):
            return ", ".join(sorted(fn_names))

        def make_row(i, key, fn_names):
            fn_col = (
                f'<td style="{_st_lbl}">{merge_fn_names(fn_names)}</td>'
                if show_functions else ""
            )
            return (
                f"<tr>"
                f'<td style="{_st_num}">[{i}]</td>'
                f'<td style="{_st_key}">{key}</td>'
                f"{fn_col}"
                f'<td style="{_st_ref}">'
                f"{self.format_entry(key, backend_name='html')}</td>"
                f"</tr>"
            )

        rows = "".join(
            make_row(i, key, fn_names)
            for i, (key, fn_names) in enumerate(self._refs.items(), 1)
        )
        display(
            HTML(
                f"<div style='border:1px solid #dee2e6;border-radius:6px;"
                f"padding:10px 14px;background:#f8f9fa'>"
                f"<h4 style='margin:0 0 8px'>{title}</h4>"
                f"<table style='border-collapse:collapse;font-size:0.9em'>{rows}</table>"
                f"</div>"
            )
        )
        if clear:
            self.clear()
