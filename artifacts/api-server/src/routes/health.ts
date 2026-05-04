import { Router, type IRouter } from "express";
import { pool } from "@workspace/db";
import { logger } from "../lib/logger";
import { FREE_TEXT_MODEL, FREE_VISION_MODEL } from "../lib/models";

const router: IRouter = Router();

type CheckStatus = "ok" | "fail" | "skipped";

interface CheckResult {
  status: CheckStatus;
  message?: string;
  latencyMs?: number;
}

async function checkDatabase(): Promise<CheckResult> {
  if (!process.env["DATABASE_URL"]) {
    return { status: "fail", message: "DATABASE_URL is not set" };
  }
  const start = Date.now();
  try {
    await pool.query("SELECT 1");
    return { status: "ok", latencyMs: Date.now() - start };
  } catch (err) {
    return {
      status: "fail",
      message: err instanceof Error ? err.message : "Database query failed",
      latencyMs: Date.now() - start,
    };
  }
}

function checkAiProvider(): CheckResult {
  const apiKey =
    process.env["OPENROUTER_API_KEY"] ||
    process.env["OPENAI_API_KEY1"] ||
    process.env["OPENAI_API_KEY"] ||
    process.env["AI_INTEGRATIONS_OPENAI_API_KEY"];
  if (!apiKey) {
    return {
      status: "fail",
      message: "AI provider is not configured. Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY in your environment.",
    };
  }
  const baseUrl =
    process.env["AI_INTEGRATIONS_OPENAI_BASE_URL"] ||
    process.env["OPENROUTER_BASE_URL"] ||
    "https://openrouter.ai/api/v1";
  try {
    new URL(baseUrl);
  } catch {
    return { status: "fail", message: `Invalid AI base URL: ${baseUrl}` };
  }
  return { status: "ok" };
}

router.get("/model-info", (_req, res) => {
  const textModel  = FREE_TEXT_MODEL;
  const visionModel = FREE_VISION_MODEL;
  const isFree = (m: string) => /:free$/.test(m) || /free/i.test(m.split("/").pop() ?? "");
  res.json({
    textModel,
    visionModel,
    textFree:   isFree(textModel),
    visionFree: isFree(visionModel),
    sameModel:  textModel === visionModel,
  });
});

router.get("/healthz", async (_req, res) => {
  const [database, ai] = await Promise.all([
    checkDatabase(),
    Promise.resolve(checkAiProvider()),
  ]);

  const allOk = database.status === "ok" && ai.status === "ok";
  const status: "ok" | "degraded" = allOk ? "ok" : "degraded";

  if (!allOk) {
    logger.warn(
      { database, ai },
      "Health check reported degraded dependencies",
    );
  }

  res.status(allOk ? 200 : 503).json({
    status,
    checks: {
      database,
      ai,
    },
    uptimeSeconds: Math.round(process.uptime()),
    timestamp: new Date().toISOString(),
  });
});

export default router;
