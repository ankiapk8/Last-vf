/**
 * AI model selection.
 *
 * Priority:
 *  1. Explicit non-empty env override (AI_TEXT_MODEL / AI_VISION_MODEL)
 *  2. OPENROUTER_API_KEY present → free Gemini 2.0 Flash on OpenRouter
 *  3. Replit modelFarm fallback → gpt-4o-mini (works with AI_INTEGRATIONS_OPENAI_API_KEY)
 */

const isOpenRouter = !!process.env.OPENROUTER_API_KEY;

const envText   = process.env.AI_TEXT_MODEL?.trim()   || null;
const envVision = process.env.AI_VISION_MODEL?.trim()  || null;

export const FREE_TEXT_MODEL =
  envText   ?? (isOpenRouter ? "google/gemini-2.0-flash-exp:free" : "gpt-4o-mini");

export const FREE_VISION_MODEL =
  envVision ?? (isOpenRouter ? "google/gemini-2.0-flash-exp:free" : "gpt-4o-mini");
