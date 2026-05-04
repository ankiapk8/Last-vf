/**
 * Default AI model constants.
 *
 * All models are available on OpenRouter (openrouter.ai).
 * Override at runtime by setting the corresponding env var.
 *
 * google/gemini-2.5-flash-preview
 *   - Latest Gemini 2.5 Flash: 1M context, vision support,
 *     superior structured-JSON output, up to 65k output tokens.
 *   - Ideal for: flashcard generation, visual detection,
 *     explanations, mind-map generation, QBank creation.
 *
 * Fallback: google/gemini-2.0-flash-001 (stable, always available)
 */

export const FREE_TEXT_MODEL =
  process.env.AI_TEXT_MODEL ?? "google/gemini-2.5-flash-preview";

export const FREE_VISION_MODEL =
  process.env.AI_VISION_MODEL ?? "google/gemini-2.5-flash-preview";
