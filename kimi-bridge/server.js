// kimi-bridge — thin HTTP wrapper around @moonshot-ai/kimi-agent-sdk so our
// Python worker can delegate scaffold jobs to a logged-in `kimi` CLI.
//
// One endpoint: POST /agent/run { prompt, workDir, model?, thinking?, timeoutMs? }
// Returns: { status, steps, transcript, tool_calls, error? }
//
// Auth lives in ~/.kimi/credentials/ inside the container — the host's
// credential dir is bind-mounted read/write so /login state persists.

import express from "express";
import { createSession } from "@moonshot-ai/kimi-agent-sdk";
import { randomUUID } from "node:crypto";
import { buildTools, closeBrowser } from "./tools.js";

const app = express();
app.use(express.json({ limit: "2mb" }));

const PORT = parseInt(process.env.PORT || "5555", 10);
const SHARED_SECRET = (process.env.KIMI_BRIDGE_SECRET || "").trim();
const DEFAULT_MODEL = process.env.KIMI_MODEL || "kimi-latest";
const DEFAULT_TIMEOUT_MS = parseInt(process.env.KIMI_TURN_TIMEOUT_MS || "1500000", 10); // 25 min

function requireAuth(req, res, next) {
  if (!SHARED_SECRET) return next(); // dev mode — no secret set
  const got = (req.get("x-kimi-bridge-secret") || "").trim();
  if (got !== SHARED_SECRET) return res.status(401).json({ error: "unauthorized" });
  next();
}

app.get("/healthz", (_req, res) => res.json({ ok: true }));

app.post("/agent/run", requireAuth, async (req, res) => {
  const reqId = randomUUID().slice(0, 8);
  const { prompt, workDir, model, thinking, timeoutMs } = req.body || {};
  if (!prompt || typeof prompt !== "string") return res.status(400).json({ error: "prompt required" });
  if (!workDir || typeof workDir !== "string") return res.status(400).json({ error: "workDir required" });

  const turnTimeout = Math.min(Math.max(parseInt(timeoutMs || DEFAULT_TIMEOUT_MS, 10), 30_000), 3_600_000);
  const t0 = Date.now();
  console.log(`[${reqId}] run start workDir=${workDir} model=${model || DEFAULT_MODEL} promptLen=${prompt.length}`);

  let session;
  try {
    session = createSession({
      workDir,
      model: model || DEFAULT_MODEL,
      thinking: thinking !== false, // default on
      yoloMode: true, // auto-approve file writes; workDir is an ephemeral scaffold dir
      externalTools: buildTools({ workdirRoot: workDir }),
    });
  } catch (err) {
    console.error(`[${reqId}] session create failed`, err);
    return res.status(500).json({ error: "session_create_failed", detail: String(err?.message || err) });
  }

  const transcript = [];
  const toolCalls = [];
  let steps = 0;
  let status = "unknown";
  let errorMsg;

  const turn = session.prompt(prompt);
  const timer = setTimeout(() => {
    console.warn(`[${reqId}] turn timeout after ${turnTimeout}ms — interrupting`);
    turn.interrupt().catch(() => {});
  }, turnTimeout);

  try {
    for await (const event of turn) {
      switch (event.type) {
        case "StepBegin":
          steps = event.payload?.n ?? steps + 1;
          break;
        case "ContentPart": {
          const p = event.payload;
          if (p?.type === "text" && p.text) transcript.push({ kind: "text", text: p.text });
          else if (p?.type === "think" && p.think) transcript.push({ kind: "think", text: p.think });
          break;
        }
        case "ToolCall": {
          const tc = event.payload;
          toolCalls.push({
            id: tc.id,
            name: tc.function?.name,
            arguments: tc.function?.arguments,
          });
          break;
        }
        case "ToolResult": {
          const tr = event.payload;
          const last = toolCalls.find((x) => x.id === tr.tool_call_id);
          if (last) {
            last.is_error = !!tr.return_value?.is_error;
            last.message = tr.return_value?.message;
          }
          break;
        }
        case "ApprovalRequest": {
          // yoloMode should suppress most of these, but fail loud if one leaks.
          const id = event.payload?.id;
          if (id) await turn.approve(id, "approve_for_session").catch(() => {});
          break;
        }
        default:
          break;
      }
    }
    const result = await turn.result;
    status = result.status || "finished";
    steps = result.steps ?? steps;
  } catch (err) {
    console.error(`[${reqId}] turn error`, err);
    status = "error";
    errorMsg = String(err?.message || err);
  } finally {
    clearTimeout(timer);
    try { await session.close(); } catch (_) { /* ignore */ }
  }

  const dtMs = Date.now() - t0;
  console.log(`[${reqId}] run done status=${status} steps=${steps} tools=${toolCalls.length} dt=${dtMs}ms`);

  res.json({
    request_id: reqId,
    status,
    steps,
    duration_ms: dtMs,
    transcript,
    tool_calls: toolCalls,
    ...(errorMsg ? { error: errorMsg } : {}),
  });
});

const server = app.listen(PORT, "0.0.0.0", () => {
  console.log(`kimi-bridge listening on :${PORT} (auth=${SHARED_SECRET ? "on" : "off"})`);
});

async function shutdown(signal) {
  console.log(`[shutdown] ${signal} — closing browser + server`);
  await closeBrowser();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 5000).unref();
}
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
