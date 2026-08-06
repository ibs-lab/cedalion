/**
 * rtd-version-shim.js
 *
 * Emulates the Read the Docs Addons "readthedocs-addons-data-ready" CustomEvent
 * for self-hosted Sphinx documentation using sphinx_rtd_theme >= 3.0.
 *
 * The RTD theme v3 removed its own version display in favour of listening for
 * a CustomEvent that the Read the Docs hosting platform fires. When you
 * self-host, that event never fires and the version selector stays empty.
 *
 * This script bridges the gap: it fetches a versions.json you maintain,
 * detects the current version from the URL, and dispatches the exact event
 * the theme expects — so the built-in version_selector "just works."
 *
 * ── Setup ──────────────────────────────────────────────────────────────────
 *
 * 1. Create a `versions.json` at a stable URL (see example below).
 *
 * 2. Copy this file into your Sphinx project's `_static/` folder.
 *
 * 3. In your conf.py:
 *
 *    html_theme_options = {
 *        "version_selector": True,
 *        "language_selector": False,
 *        # …
 *    }
 *
 *    html_js_files = [
 *        "rtd-version-shim.js",
 *    ]
 *
 * ── versions.json example ──────────────────────────────────────────────────
 *
 * Place this at e.g. https://server.tld/docs/versions.json
 *
 *   [
 *     { "slug": "latest",  "url": "/docs/latest/"  },
 *     { "slug": "2.1.0",   "url": "/docs/2.1.0/"   },
 *     { "slug": "2.0.0",   "url": "/docs/2.0.0/"   },
 *     { "slug": "1.9.0",   "url": "/docs/1.9.0/"   }
 *   ]
 *
 *   Each entry needs at minimum:
 *     - slug: the version identifier (shown in the dropdown)
 *     - url:  the base URL for that version's docs
 *
 *   Optional fields:
 *     - name: display name override (e.g. "2.1.0 (stable)")
 *     - hidden: true to hide from the dropdown (but still navigable)
 *
 * ── Configuration ──────────────────────────────────────────────────────────
 *
 * Set these variables BEFORE this script loads (e.g. in an earlier <script>
 * tag or via html_context / a tiny inline script):
 *
 *   DEFINED_RTD_SHIM_VERSIONS_URL  — URL to versions.json
 *                                   Default: auto-detected as
 *                                   <docs_base>/versions.json
 *
 *   DEFINED_RTD_SHIM_DOCS_BASE     — The path prefix before the version
 *                                   segment, e.g. "/docs/"
 *                                   Default: auto-detected from the first
 *                                   path segment
 *
 *   DEFINED_RTD_SHIM_PROJECT_SLUG  — Project name for the Addons data
 *                                   Default: "project"
 */

(function () {
  "use strict";

  // ── Helpers ────────────────────────────────────────────────────────────

  /**
   * Detect the docs base path from the URL.
   * For /docs/2.1.0/api/index.html → "/docs/"
   */
  function detectDocsBase() {
    var match = window.location.pathname.match(/^(\/[^/]+\/)/);
    return match ? match[1] : "/";
  }

  /**
   * Extract the current version slug from the URL.
   * For /docs/2.1.0/api/index.html with base "/docs/" → "2.1.0"
   */
  function detectCurrentVersion(docsBase) {
    var base = docsBase.replace(/\/$/, "");
    var rest = window.location.pathname.substring(base.length).replace(/^\//, "");
    var segments = rest.split("/");
    return segments[0] || null;
  }

  // ── Config ─────────────────────────────────────────────────────────────

  //var docsBase =
  //  window.DEFINED_RTD_SHIM_DOCS_BASE || detectDocsBase();
  var docsBase = "/cedalion/doc/";
  var versionsUrl =
    window.DEFINED_RTD_SHIM_VERSIONS_URL || docsBase + "versions.json";
  var projectSlug =
    window.DEFINED_RTD_SHIM_PROJECT_SLUG || "project";

  var currentSlug = detectCurrentVersion(docsBase);

  if (!currentSlug) {
    console.warn(
      "[rtd-version-shim] Could not detect current version from URL:",
      window.location.pathname
    );
    return;
  }

  // ── Fetch and dispatch ─────────────────────────────────────────────────

  fetch(versionsUrl)
    .then(function (res) {
      if (!res.ok)
        throw new Error(
          "Failed to fetch " + versionsUrl + " (HTTP " + res.status + ")"
        );
      return res.json();
    })
    .then(function (versions) {
      // Build the active versions array in the shape the RTD theme expects.
      var activeVersions = versions
        .filter(function (v) {
          return !v.hidden;
        })
        .map(function (v) {
          return {
            slug: v.slug,
            verbose_name: v.name || v.slug,
            active: true,
            hidden: false,
            urls: {
              documentation: v.url,
            },
          };
        });

      // Find the current version object.
      var currentEntry = versions.find(function (v) {
        return v.slug === currentSlug;
      });

      if (!currentEntry) {
        console.warn(
          "[rtd-version-shim] Current version '" +
            currentSlug +
            "' not found in versions.json. The selector may not highlight correctly."
        );
        currentEntry = { slug: currentSlug, url: window.location.pathname };
      }

      var currentVersion = {
        slug: currentEntry.slug,
        verbose_name: currentEntry.name || currentEntry.slug,
        active: true,
        hidden: !!currentEntry.hidden,
        urls: {
          documentation: currentEntry.url,
        },
      };

      // Assemble the full addons data payload that the RTD theme expects.
      var addonsData = {
        api_version: "1",
        projects: {
          current: {
            slug: projectSlug,
            language: { code: "en" },
            versioning_scheme: "multiple_versions_without_translations",
            programming_language: { code: "py" },
            urls: {
              documentation: docsBase,
              home: docsBase,
            },
          },
          translations: [],
        },
        versions: {
          current: currentVersion,
          active: activeVersions,
        },
        addons: {
          flyout: { enabled: true, position: null },
          search: { enabled: false },
          analytics: { enabled: false },
          notifications: {
            enabled: false,
            show_on_latest: false,
            show_on_non_stable: false,
            show_on_external: false,
          },
          hotkeys: { enabled: false },
          linkpreviews: { enabled: false },
          filetreediff: { enabled: false },
          ethicalads: { enabled: false },
        },
      };

      // Dispatch the CustomEvent the RTD theme listens for.
      var event = new CustomEvent("readthedocs-addons-data-ready", {
        detail: {
          data: function () {
            return addonsData;
          },
        },
      });

      // Also set the global so late-loading scripts can access the data.
      window.ReadTheDocsEventData = {
        data: function () {
          return addonsData;
        },
      };

      document.dispatchEvent(event);
    })
    .catch(function (err) {
      console.warn("[rtd-version-shim]", err.message);
    });
})();
