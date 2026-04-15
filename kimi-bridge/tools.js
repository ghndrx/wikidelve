// External tools exposed to kimi-cli via @moonshot-ai/kimi-agent-sdk.
//
// Three Playwright-backed tools that run inside the sidecar container:
//   browser_screenshot      — fetch a remote URL, return PNG (base64) + title
//   browser_snapshot        — fetch a remote URL, return accessibility tree + links
//   browser_validate_local  — render a file in the session workdir and report
//                             console errors + a screenshot of the result
//
// All handlers share a single Chromium instance via getBrowser() so we
// don't pay browser-launch cost on every tool call.

import { chromium } from "playwright";
import { z } from "zod";
import { createExternalTool } from "@moonshot-ai/kimi-agent-sdk";
import path from "node:path";
import fs from "node:fs/promises";

let _browser;
async function getBrowser() {
  if (_browser && _browser.isConnected()) return _browser;
  _browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  return _browser;
}

export async function closeBrowser() {
  if (_browser) {
    try { await _browser.close(); } catch (_) { /* ignore */ }
    _browser = undefined;
  }
}

const VIEWPORTS = {
  desktop: { width: 1440, height: 900 },
  tablet: { width: 834, height: 1112 },
  mobile: { width: 390, height: 844 },
};

function pickViewport(v) {
  if (v && typeof v === "object" && v.width && v.height) return v;
  return VIEWPORTS[v] || VIEWPORTS.desktop;
}

// Screenshots go to disk inside the workdir, not into the tool response —
// base64-encoding a 1.5MB PNG into the stream devours 500K+ tokens per
// call and will blow past Kimi's 262K context window after 2-3 calls.
// Tools return the file path; kimi can read/display it from disk if needed.
import { randomUUID } from "node:crypto";

async function withPage(viewport, fn) {
  const browser = await getBrowser();
  const ctx = await browser.newContext({
    viewport: pickViewport(viewport),
    userAgent: "wikidelve-kimi-bridge/0.1 (+playwright)",
  });
  const page = await ctx.newPage();
  try {
    return await fn(page);
  } finally {
    await ctx.close().catch(() => {});
  }
}

// Pull up to `limit` {text, href} pairs from a loaded page.
async function collectLinks(page, limit) {
  return page.evaluate((n) => {
    const out = [];
    const anchors = document.querySelectorAll("a[href]");
    for (let i = 0; i < anchors.length && out.length < n; i++) {
      const a = anchors[i];
      out.push({
        text: (a.textContent || "").trim().slice(0, 120),
        href: a.getAttribute("href") || "",
      });
    }
    return out;
  }, limit);
}

export function buildTools({ workdirRoot }) {
  // workdirRoot constrains validate_local to sandbox paths so an agent
  // can't read /etc/passwd through the file:// loader.
  const safeLocalPath = (rel) => {
    const abs = path.resolve(workdirRoot, rel);
    if (!abs.startsWith(path.resolve(workdirRoot) + path.sep) && abs !== path.resolve(workdirRoot)) {
      throw new Error(`path escapes workdir: ${rel}`);
    }
    return abs;
  };

  const browser_screenshot = createExternalTool({
    name: "browser_screenshot",
    description:
      "Fetch a remote URL and return a PNG screenshot (base64). Use this to " +
      "study real-world reference UIs before building. Returns title, final " +
      "URL after redirects, and a base64-encoded PNG.",
    parameters: z.object({
      url: z.string().url().describe("The URL to load"),
      viewport: z.union([
        z.enum(["desktop", "tablet", "mobile"]),
        z.object({ width: z.number().int().min(200).max(3840), height: z.number().int().min(200).max(2160) }),
      ]).optional().default("desktop"),
      fullPage: z.boolean().optional().default(false).describe("Capture the full scrollable page (slower)"),
      waitMs: z.number().int().min(0).max(10_000).optional().default(800).describe("Extra wait after load for JS-driven UIs"),
    }),
    handler: async ({ url, viewport, fullPage, waitMs }) => {
      return withPage(viewport, async (page) => {
        const resp = await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30_000 });
        if (waitMs) await page.waitForTimeout(waitMs);
        const title = (await page.title().catch(() => "")) || "";
        const finalUrl = page.url();
        const fname = `.screenshots/ext-${randomUUID().slice(0, 8)}.png`;
        const abs = path.resolve(workdirRoot, fname);
        await fs.mkdir(path.dirname(abs), { recursive: true });
        await page.screenshot({ type: "png", fullPage, path: abs });
        const { size } = await fs.stat(abs);
        return {
          output: JSON.stringify({
            title, finalUrl, status: resp?.status() ?? null,
            screenshot_path: fname, size_bytes: size,
          }),
          message: `Screenshotted ${finalUrl} → ${fname} (${size}B)`,
        };
      });
    },
  });

  const browser_snapshot = createExternalTool({
    name: "browser_snapshot",
    description:
      "Fetch a remote URL and return its accessibility tree + top-level " +
      "links. Better than raw DOM for understanding page structure. Use to " +
      "study how reference sites are organized before building.",
    parameters: z.object({
      url: z.string().url(),
      waitMs: z.number().int().min(0).max(10_000).optional().default(800),
      maxLinks: z.number().int().min(1).max(200).optional().default(40),
    }),
    handler: async ({ url, waitMs, maxLinks }) => {
      return withPage("desktop", async (page) => {
        const resp = await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30_000 });
        if (waitMs) await page.waitForTimeout(waitMs);
        const title = (await page.title().catch(() => "")) || "";
        const snapshot = await page.accessibility.snapshot({ interestingOnly: true }).catch(() => null);
        const links = await collectLinks(page, maxLinks);
        return {
          output: JSON.stringify({
            title, finalUrl: page.url(), status: resp?.status() ?? null,
            accessibility: snapshot, links,
          }),
          message: `Snapshotted ${page.url()} (${links.length} links)`,
        };
      });
    },
  });

  const browser_validate_local = createExternalTool({
    name: "browser_validate_local",
    description:
      "Render a file from the current working directory and capture console " +
      "errors + a screenshot of the result. Use this to self-validate your " +
      "scaffold output — render index.html, see what it looks like, iterate.",
    parameters: z.object({
      path: z.string().describe("Relative path inside the workdir, e.g. 'index.html'"),
      viewport: z.union([
        z.enum(["desktop", "tablet", "mobile"]),
        z.object({ width: z.number().int(), height: z.number().int() }),
      ]).optional().default("desktop"),
      waitMs: z.number().int().min(0).max(10_000).optional().default(500),
    }),
    handler: async ({ path: rel, viewport, waitMs }) => {
      const abs = safeLocalPath(rel);
      try { await fs.access(abs); }
      catch { return { output: JSON.stringify({ error: "file not found", path: rel }), message: "file missing" }; }

      return withPage(viewport, async (page) => {
        const consoleMsgs = [];
        const errors = [];
        page.on("console", (msg) => {
          if (msg.type() === "error" || msg.type() === "warning") {
            consoleMsgs.push({ type: msg.type(), text: msg.text().slice(0, 500) });
          }
        });
        page.on("pageerror", (err) => errors.push(String(err.message || err).slice(0, 500)));

        await page.goto("file://" + abs, { waitUntil: "domcontentloaded", timeout: 15_000 });
        if (waitMs) await page.waitForTimeout(waitMs);
        const title = (await page.title().catch(() => "")) || "";
        const shotName = `.screenshots/self-${randomUUID().slice(0, 8)}.png`;
        const shotAbs = path.resolve(workdirRoot, shotName);
        await fs.mkdir(path.dirname(shotAbs), { recursive: true });
        await page.screenshot({ type: "png", fullPage: true, path: shotAbs });
        const { size } = await fs.stat(shotAbs);
        return {
          output: JSON.stringify({
            path: rel, title, console: consoleMsgs, errors,
            screenshot_path: shotName, size_bytes: size,
          }),
          message: `Validated ${rel} → ${shotName} (${errors.length} errors, ${consoleMsgs.length} console)`,
        };
      });
    },
  });

  // --- Research tools: HTTP callbacks into the wikidelve app ---------------
  const APP_URL = (process.env.WIKIDELVE_APP_URL || "http://app:8888").replace(/\/$/, "");
  const BRIDGE_SECRET = process.env.KIMI_BRIDGE_SECRET || "";

  async function callApp(path, body) {
    const res = await fetch(`${APP_URL}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(BRIDGE_SECRET ? { "x-kimi-bridge-secret": BRIDGE_SECRET } : {}),
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30_000),
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`app ${res.status}: ${txt.slice(0, 300)}`);
    }
    return res.json();
  }

  const search_web = createExternalTool({
    name: "search_web",
    description:
      "Google-quality web search (Serper). Returns ranked results with " +
      "titles, URLs, and snippets. Use to find reference UIs, component " +
      "libraries, real-world examples before building.",
    parameters: z.object({
      query: z.string().min(1),
      num_results: z.number().int().min(1).max(20).optional().default(8),
    }),
    handler: async ({ query, num_results }) => {
      const data = await callApp("/api/internal/search_web", { query, num_results });
      return {
        output: JSON.stringify(data),
        message: `search_web "${query}" → ${(data.results || []).length} results`,
      };
    },
  });

  const read_webpage = createExternalTool({
    name: "read_webpage",
    description:
      "Extract the main text content from a webpage URL (Readability). " +
      "Returns up to 12k chars of clean text. Use to read specific reference " +
      "articles, documentation, or design writeups in depth.",
    parameters: z.object({
      url: z.string().url(),
    }),
    handler: async ({ url }) => {
      const data = await callApp("/api/internal/read_webpage", { url });
      return {
        output: JSON.stringify(data),
        message: `read_webpage ${url} → ${(data.text || "").length} chars`,
      };
    },
  });

  const search_kb = createExternalTool({
    name: "search_kb",
    description:
      "Search the WikiDelve knowledge base for existing internal articles " +
      "matching a query (hybrid FTS + vector + graph). Use FIRST before " +
      "web research to reuse patterns already documented in the wiki.",
    parameters: z.object({
      query: z.string().min(1),
      kb: z.string().optional().default("personal"),
      limit: z.number().int().min(1).max(30).optional().default(10),
    }),
    handler: async ({ query, kb, limit }) => {
      const data = await callApp("/api/internal/search_kb", { query, kb, limit });
      return {
        output: JSON.stringify(data),
        message: `search_kb "${query}" in ${kb} → ${(data.results || []).length} hits`,
      };
    },
  });

  return [
    browser_screenshot, browser_snapshot, browser_validate_local,
    search_web, read_webpage, search_kb,
  ];
}
