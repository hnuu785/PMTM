"use client";

import Image from "next/image";
import { FormEvent, useMemo, useState } from "react";
import { getApiBaseUrl } from "@/lib/api";

type LyricResponse = {
  title: string;
  lyrics: string;
  bpm: number;
  llm: LyricModel;
  notes: string[];
};

type LyricModel =
  | "qwen-local"
  | "qwen-exp-001-sft"
  | "qwen-exp-001-grpo"
  | "qwen-exp-002-sft"
  | "qwen-exp-002-grpo"
  | "openai";

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
    value: "qwen-exp-002-sft",
    label: "exp-002 SFT",
    detail: "Qwen + SFT adapter",
  },
  {
    value: "qwen-exp-002-grpo",
    label: "exp-002 GRPO",
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
    <main className="pmtm-stage min-h-screen overflow-hidden text-[#fff6df]">
      <section className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:px-8 lg:py-6">
        <header className="flex flex-col gap-3 border-b border-[#f5b950]/25 pb-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex min-w-0 items-center gap-4">
            <div className="relative h-16 w-16 shrink-0 overflow-hidden rounded-sm border border-[#f7c76d]/45 bg-black/55 shadow-[0_0_34px_rgba(245,106,32,0.34)] sm:h-20 sm:w-20">
              <Image
                src="/brand/pmtm-icon.png"
                alt="프메더머니 로고"
                fill
                priority
                sizes="96px"
                className="object-cover"
              />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-semibold tracking-[0.24em] text-[#52d4c8] uppercase">
                PMTM
              </p>
              <h1 className="mt-1.5 text-2xl leading-tight font-black text-[#fff3ca] sm:text-3xl">
                BPM으로 랩 벌스 만들기
              </h1>
              <p className="mt-1.5 max-w-xl text-sm leading-6 text-[#d8b993]">
                비트의 속도와 모델을 고르면 바로 8마디 초안을 생성합니다.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs font-semibold tracking-[0.16em] text-[#f5b950] uppercase">
            <span className="h-2 w-2 rounded-full bg-[#ff5a1f] shadow-[0_0_16px_rgba(255,90,31,0.9)]" />
            Beat Draft
          </div>
        </header>

        <div className="grid flex-1 gap-4 lg:grid-cols-[380px_1fr]">
          <aside className="pmtm-panel flex flex-col justify-between gap-5 p-4 sm:p-5">
            <div className="space-y-4">
              <div>
                <p className="text-xs font-semibold tracking-[0.18em] text-[#f5b950] uppercase">
                  Generator
                </p>
                <h2 className="mt-2 text-xl font-bold text-[#fff6df]">생성 조건</h2>
              </div>

              <form onSubmit={handleGenerate} className="space-y-3">
                <label className="block">
                  <span className="text-sm font-semibold text-[#d8b993]">BPM</span>
                  <input
                    value={bpm}
                    onChange={(event) => setBpm(event.target.value)}
                    inputMode="numeric"
                    className="mt-2 h-16 w-full border border-[#f5b950]/45 bg-[#130806]/88 px-4 text-4xl font-black text-[#fff3ca] outline-none transition placeholder:text-[#7b5130] focus:border-[#ffb23f] focus:shadow-[0_0_0_3px_rgba(255,178,63,0.18)]"
                    placeholder="90"
                    aria-label="BPM"
                  />
                </label>

                <div className="grid grid-cols-4 gap-2 rounded-sm border border-[#f5b950]/25 bg-black/25 p-1">
                  {BPM_PRESETS.map((preset) => (
                    <button
                      key={preset}
                      type="button"
                      onClick={() => setBpm(String(preset))}
                      className={`h-10 border text-sm font-bold transition ${
                        bpm === String(preset)
                          ? "border-[#ffb23f] bg-[#ff5a1f] text-white shadow-[0_0_18px_rgba(255,90,31,0.34)]"
                          : "border-transparent bg-[#23100b] text-[#d8b993] hover:border-[#f5b950]/55 hover:text-[#fff3ca]"
                      }`}
                    >
                      {preset}
                    </button>
                  ))}
                </div>

                <fieldset className="space-y-2">
                  <legend className="text-sm font-semibold text-[#d8b993]">LLM</legend>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {LLM_OPTIONS.map((option) => (
                      <label
                        key={option.value}
                        className={`flex min-h-[52px] cursor-pointer items-start justify-between gap-2 border px-3 py-2 transition ${
                          llm === option.value
                            ? "border-[#f5b950] bg-[#f5b950] text-[#170906]"
                            : "border-[#f5b950]/22 bg-[#130806]/82 text-[#fff6df] hover:border-[#f5b950]/60"
                        }`}
                      >
                        <span>
                          <span className="block text-sm leading-5 font-semibold">{option.label}</span>
                          <span
                            className={`block text-[11px] leading-4 ${
                              llm === option.value ? "text-[#5f260d]" : "text-[#b9865f]"
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
                          className="mt-1 h-4 w-4 shrink-0 accent-[#ff5a1f]"
                        />
                      </label>
                    ))}
                  </div>
                </fieldset>

                <button
                  type="submit"
                  disabled={isLoading}
                  className="h-12 w-full border border-[#ffd78a]/55 bg-[#ff5a1f] px-4 text-sm font-black tracking-[0.08em] text-white uppercase shadow-[0_14px_34px_rgba(255,90,31,0.28)] transition hover:bg-[#ff7a28] disabled:cursor-not-allowed disabled:border-[#6d4530] disabled:bg-[#6d4530] disabled:text-[#c39a75] disabled:shadow-none"
                >
                  {isLoading ? "Generating" : "Generate"}
                </button>
              </form>
            </div>

            <p className="border-t border-[#f5b950]/20 pt-4 text-xs leading-5 text-[#a97859]">
              API: <span className="break-all font-mono text-[#d8b993]">{apiBaseUrl}</span>
            </p>
          </aside>

          <section className="pmtm-panel flex min-h-[560px] flex-col p-4 sm:p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#f5b950]/25 pb-4">
              <div>
                <p className="text-sm font-semibold text-[#52d4c8]">Generated verse</p>
                <h2 className="mt-1 text-xl font-black text-[#fff3ca]">
                  {result ? result.title : "결과가 여기에 표시됩니다"}
                </h2>
                <p className="mt-1 text-xs font-semibold tracking-[0.14em] text-[#b9865f] uppercase">
                  {bpm || "--"} BPM · {llm}
                </p>
              </div>
              <button
                type="button"
                onClick={handleCopy}
                disabled={!result}
                className="h-10 min-w-24 border border-[#f5b950]/75 bg-[#fff3ca] px-4 text-sm font-black text-[#170906] transition hover:bg-[#f5b950] disabled:cursor-not-allowed disabled:border-[#6d4530] disabled:bg-transparent disabled:text-[#8b674c]"
              >
                {copyLabel}
              </button>
            </div>

            {error ? (
              <div className="mt-5 border border-[#ff6b4a]/55 bg-[#2b0c08] px-4 py-3 text-sm font-medium text-[#ffb6a2]">
                {error}
              </div>
            ) : null}

            <div className="lyric-paper mt-5 flex-1 border border-[#f5b950]/35">
              <pre className="min-h-[420px] whitespace-pre-wrap px-5 py-5 font-mono text-sm leading-8 text-[#fff6df] sm:px-7 sm:py-6">
                {isLoading
                  ? "비트에 맞춰 벌스를 구성하는 중..."
                  : result?.lyrics ??
                    "BPM을 입력하고 Generate를 누르면 8마디 벌스가 생성됩니다."}
              </pre>
            </div>

            <div className="mt-4 min-h-12 space-y-1 text-sm text-[#b9865f]">
              {result?.notes.map((note) => (
                <p key={note}>{note}</p>
              ))}
            </div>
          </section>
        </div>
      </section>
    </main>
  );
}
