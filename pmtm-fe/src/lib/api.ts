const FALLBACK_API_BASE_URL = "http://localhost:8100";

export function getApiBaseUrl() {
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? FALLBACK_API_BASE_URL;
}
