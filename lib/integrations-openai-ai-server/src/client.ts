import OpenAI from "openai";

const apiKey =
  process.env.OPENROUTER_API_KEY ||
  process.env.OPENAI_API_KEY1 ||
  process.env.OPENAI_API_KEY ||
  process.env.AI_INTEGRATIONS_OPENAI_API_KEY;

if (!apiKey) {
  throw new Error(
    "OPENROUTER_API_KEY must be set. Get one at https://openrouter.ai/keys",
  );
}

const baseURL =
  process.env.AI_INTEGRATIONS_OPENAI_BASE_URL ||
  process.env.OPENROUTER_BASE_URL ||
  "https://openrouter.ai/api/v1";

export const openai = new OpenAI({
  apiKey,
  baseURL,
  defaultHeaders: {
    "HTTP-Referer":
      process.env.OPENROUTER_HTTP_REFERER || "https://anki-generator.local",
    "X-Title": process.env.OPENROUTER_APP_TITLE || "Anki Card Generator",
  },
});
