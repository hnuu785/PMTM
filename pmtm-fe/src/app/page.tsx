"use client";

import { FormEvent, useMemo, useState } from "react";
import { getApiBaseUrl } from "@/lib/api";

type LyricResponse = {
  title: string;
  lyrics: string;
  bpm: number;
  llm: LyricModel;
  notes: string[];
};

type LyricModel = "qwen-local" | "qwen-exp-001-sft" | "qwen-exp-001-grpo" | "openai";

const BPM_PRESETS = [80, 90, 120, 140];
const LLM_OPTIONS: Array<{ value: LyricModel; label: string; detail: string }> = [
  {
    value: "qwen-local",
    label: "Qwen local",
    detail: "Qwen2.5-1.5B",
  },
  {
    value: "qwen-exp-001-sft",
    label: "exp-001 SFT",
    detail: "Qwen + SFT adapter",
  },
  {
    value: "qwen-exp-001-grpo",
    label: "exp-001 GRPO",
    detail: "Qwen + GRPO adapter",
  },
  {
    value: "openai",
    label: "OpenAI",
    detail: "gpt-5-mini",
  },
];

export default function Home() {
  const [bpm, setBpm] = useState("90");
  const [llm, setLlm] = useState<LyricModel>("qwen-local");
  const [result, setResult] = useState<LyricResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [copyLabel, setCopyLabel] = useState("Copy");

  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);

  async function handleGenerate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const parsedBpm = Number(bpm);
    if (!Number.isInteger(parsedBpm) || parsedBpm < 40 || parsedBpm > 220) {
      setError("BPM은 40부터 220 사이의 정수로 입력해주세요.");
      return;
    }

    setIsLoading(true);
    setError("");
    setCopyLabel("Copy");

    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/lyrics/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ bpm: parsedBpm, llm }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "가사 생성 요청에 실패했습니다.");
      }

      const data = (await response.json()) as LyricResponse;
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleCopy() {
    if (!result) {
      return;
    }

    await navigator.clipboard.writeText(result.lyrics);
    setCopyLabel("Copied");
    window.setTimeout(() => setCopyLabel("Copy"), 1400);
  }

  return (
    <main className="min-h-screen bg-[#f6f3ed] text-zinc-950">
      <section className="mx-auto grid min-h-screen w-full max-w-6xl gap-8 px-5 py-6 md:grid-cols-[360px_1fr] md:px-8 md:py-10">
        <aside className="flex flex-col justify-between gap-8 border-b border-zinc-300 pb-6 md:border-r md:border-b-0 md:pr-8 md:pb-0">
          <div className="space-y-8">
            <div>
              <p className="text-xs font-semibold tracking-[0.22em] text-zinc-500 uppercase">
                PMTM
              </p>
              <h1 className="mt-3 text-3xl leading-tight font-semibold">
                BPM으로 랩 벌스 만들기
              </h1>
            </div>

            <form onSubmit={handleGenerate} className="space-y-5">
              <label className="block">
                <span className="text-sm font-medium text-zinc-700">BPM</span>
                <input
                  value={bpm}
                  onChange={(event) => setBpm(event.target.value)}
                  inputMode="numeric"
                  className="mt-2 h-14 w-full border border-zinc-400 bg-white px-4 text-2xl font-semibold outline-none transition focus:border-zinc-950"
                  placeholder="90"
                  aria-label="BPM"
                />
              </label>

              <div className="grid grid-cols-4 gap-2">
                {BPM_PRESETS.map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setBpm(String(preset))}
                    className="h-10 border border-zinc-300 bg-white text-sm font-medium text-zinc-700 transition hover:border-zinc-950 hover:text-zinc-950"
                  >
                    {preset}
                  </button>
                ))}
              </div>

              <fieldset className="space-y-2">
                <legend className="text-sm font-medium text-zinc-700">LLM</legend>
                <div className="grid gap-2">
                  {LLM_OPTIONS.map((option) => (
                    <label
                      key={option.value}
                      className={`flex cursor-pointer items-center justify-between border px-3 py-3 transition ${
                        llm === option.value
                          ? "border-zinc-950 bg-zinc-950 text-white"
                          : "border-zinc-300 bg-white text-zinc-800 hover:border-zinc-950"
                      }`}
                    >
                      <span>
                        <span className="block text-sm font-semibold">{option.label}</span>
                        <span
                          className={`block text-xs ${
                            llm === option.value ? "text-zinc-300" : "text-zinc-500"
                          }`}
                        >
                          {option.detail}
                        </span>
                      </span>
                      <input
                        type="radio"
                        name="llm"
                        value={option.value}
                        checked={llm === option.value}
                        onChange={() => setLlm(option.value)}
                        className="h-4 w-4 accent-zinc-950"
                      />
                    </label>
                  ))}
                </div>
              </fieldset>

              <button
                type="submit"
                disabled={isLoading}
                className="h-12 w-full bg-zinc-950 px-4 text-sm font-semibold text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-400"
              >
                {isLoading ? "Generating" : "Generate"}
              </button>
            </form>
          </div>

          <p className="text-xs leading-5 text-zinc-500">
            API: <span className="break-all font-mono">{apiBaseUrl}</span>
          </p>
        </aside>

        <section className="flex min-h-[520px] flex-col">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-300 pb-4">
            <div>
              <p className="text-sm font-medium text-zinc-500">Generated verse</p>
              <h2 className="mt-1 text-xl font-semibold">
                {result ? result.title : "결과가 여기에 표시됩니다"}
              </h2>
            </div>
            <button
              type="button"
              onClick={handleCopy}
              disabled={!result}
              className="h-10 min-w-24 border border-zinc-950 bg-white px-4 text-sm font-semibold text-zinc-950 transition hover:bg-zinc-950 hover:text-white disabled:cursor-not-allowed disabled:border-zinc-300 disabled:text-zinc-400 disabled:hover:bg-white"
            >
              {copyLabel}
            </button>
          </div>

          {error ? (
            <div className="mt-5 border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          ) : null}

          <div className="mt-5 flex-1 border border-zinc-300 bg-white">
            <pre className="min-h-[420px] whitespace-pre-wrap px-5 py-5 font-mono text-sm leading-7 text-zinc-900">
              {result?.lyrics ??
                "BPM을 입력하고 Generate를 누르면 8마디 벌스가 생성됩니다."}
            </pre>
          </div>

          <div className="mt-4 min-h-12 space-y-1 text-sm text-zinc-500">
            {result?.notes.map((note) => (
              <p key={note}>{note}</p>
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}
