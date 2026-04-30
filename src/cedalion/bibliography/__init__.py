from pathlib import Path
from pybtex.database import parse_file, BibliographyData
from pybtex.plugin import find_plugin
from importlib.resources import files

BIB_FILE = files("cedalion.bibliography") / "references.bib"
bib_data = parse_file(BIB_FILE)


class Bibliography:
    """Collects citation keys.

    Automatically deduplicates. Dumps as a formatted references section.
    """

    def __init__(self):
        self._refs: dict[str, str] = {}  # key → human label

    def cite(self, bibtex_key, label):
        self._refs.setdefault(bibtex_key, label)  # first occurrence wins

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
        for i, (key, label) in enumerate(self._refs.items(), 1):
            rendered = self.format_entry(key)
            lines.append(f"[{i}] {key} — {label}")
            lines.append(f"    {rendered}")
        return "\n".join(lines)

    def dump_to_file(self, path, mode="a", clear=False):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open(mode) as fout:
            fout.write(self.dump_to_string() + "\n")
        if clear:
            self.clear()

    def dump_to_notebook(self, title="Methods used", clear=False):
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
        _st_lbl = "padding-right:16px;white-space:nowrap;vertical-align:top"
        rows = "".join(
            f"<tr>"
            f'<td style="{_st_num}">[{i}]</td>'
            f'<td style="{_st_key}">{key}</td>'
            f'<td style="{_st_lbl}">{label}</td>'
            f'<td style="color:#444">'
            f'{self.format_entry(key, backend_name="html")}</td>'
            f"</tr>"
            for i, (key, label) in enumerate(self._refs.items(), 1)
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
