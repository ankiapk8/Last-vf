/**
 * AI model selection.
 *
 * Priority:
 *  1. Explicit non-empty env override (AI_TEXT_MODEL / AI_VISION_MODEL)
 *  2. OPENROUTER_API_KEY present → best available free models on OpenRouter
 *  3. Replit modelFarm fallback → gpt-4o-mini (works with AI_INTEGRATIONS_OPENAI_API_KEY)
 *
 * Feature-specific overrides (OpenRouter only):
 *  - Explain features → deepseek/deepseek-v3:free
 *  - Visual Card Detection → google/gemini-2.5-flash-preview:free (vision-capable)
 */

const isOpenRouter = !!process.env.OPENROUTER_API_KEY;

const envText   = process.env.AI_TEXT_MODEL?.trim()   || null;
const envVision = process.env.AI_VISION_MODEL?.trim()  || null;

export const FREE_TEXT_MODEL =
  envText   ?? (isOpenRouter ? "google/gemini-2.5-flash-preview:free" : "gpt-4o-mini");

export const FREE_VISION_MODEL =
  envVision ?? (isOpenRouter ? "google/gemini-2.5-flash-preview:free" : "gpt-4o-mini");

export const EXPLAIN_MODEL =
  isOpenRouter ? "deepseek/deepseek-v3:free" : FREE_TEXT_MODEL;

export const VISUAL_DETECTION_MODEL =
  isOpenRouter ? "google/gemini-2.5-flash-preview:free" : FREE_VISION_MODEL;
