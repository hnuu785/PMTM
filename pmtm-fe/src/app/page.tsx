"use client";

import Image from "next/image";
import Link from "next/link";
import { SubmitEvent, useEffect, useMemo, useState } from "react";
import { getApiBaseUrl } from "@/lib/api";

type LyricResponse = {
  title: string;
  lyrics: string;
  bpm: number;
  llm: LyricModel;
  notes: string[];
  rhymeAnalysis: RhymeLineAnalysis[];
};

type DemoStatus =
  | "queued"
  | "analyzing"
  | "writing"
  | "planning"
  | "voicing"
  | "rendering"
  | "converting_rvc"
  | "mixing"
  | "succeeded"
  | "failed";

type DemoGenerateResponse = {
  jobId: string;
  status: DemoStatus;
};

type DemoStatusResponse = {
  jobId: string;
  status: DemoStatus;
  progress: number;
  workerAvailable: boolean;
  workerCount: number;
  bpm: number | null;
  lyrics: string | null;
  notes: string[];
  error: string | null;
  audioUrl: string | null;
  vocalUrl: string | null;
  flowPlanUrl: string | null;
  voicebank: string | null;
};

type BeatAnalysis = {
  tempo: number;
  firstBarStartSec: number;
};

type VoicebankInfo = {
  id: string;
  label: string;
  available: boolean;
};

type RhymeLineAnalysis = {
  text: string;
  rhymeGroup: number | null;
  score: number;
  highlightStart: number | null;
  highlightEnd: number | null;
  highlightRanges?: Array<{ start: number; end: number }>;
};

type LyricModel = string;

type ApiErrorResponse = {
  detail?: unknown;
};

type GenerateMode = "beat" | "manual";

const BPM_PRESETS = [80, 90, 120, 140];
const DIFFSINGER_VOICEBANKS = [
  { value: "potg", label: "POTG" },
  { value: "kitane", label: "KITANE" },
  { value: "rang", label: "RANG" },
  { value: "lunar", label: "LUNAR" },
];
const RHYME_COLORS = [
  { background: "rgba(82, 212, 200, 0.28)", border: "rgba(82, 212, 200, 0.74)", color: "#d7fffb" },
  { background: "rgba(255, 90, 31, 0.28)", border: "rgba(255, 90, 31, 0.74)", color: "#ffe2d4" },
  { background: "rgba(245, 185, 80, 0.28)", border: "rgba(245, 185, 80, 0.74)", color: "#fff3ca" },
  { background: "rgba(150, 124, 255, 0.28)", border: "rgba(150, 124, 255, 0.74)", color: "#ece7ff" },
  { background: "rgba(74, 222, 128, 0.24)", border: "rgba(74, 222, 128, 0.70)", color: "#dcfce7" },
];
const DEFAULT_LLM_OPTIONS: Array<{ value: string; label: string; detail: string }> = [
  {
    value: "qwen-local",
    label: "Qwen local",
    detail: "Qwen2.5-3B-Instruct",
  },
  {
    value: "openai",
    label: "OpenAI",
    detail: "gpt-5-mini",
  },
];
const TOPIC_OPTIONS = [
  { value: "", label: "전체 (자율)" },
  { value: "자신감/성공", label: "자신감/성공" },
  { value: "삶/성찰", label: "삶/성찰" },
  { value: "사랑", label: "사랑" },
  { value: "이별", label: "이별" },
  { value: "유흥/파티", label: "유흥/파티" },
  { value: "비판/디스", label: "비판/디스" },
];

export default function Home() {
  const [mode, setMode] = useState<GenerateMode>("beat");
  const [beatFile, setBeatFile] = useState<File | null>(null);
  const [beatAnalysis, setBeatAnalysis] = useState<BeatAnalysis | null>(null);
  const [firstBarStartSec, setFirstBarStartSec] = useState("0");
  const [bpm, setBpm] = useState("90");
  const [topic, setTopic] = useState("");
  const [llm, setLlm] = useState<LyricModel>("qwen-local");
  const [llmOptions, setLlmOptions] = useState<Array<{ value: string; label: string; detail: string }>>(
    DEFAULT_LLM_OPTIONS,
  );
  const [voicebank, setVoicebank] = useState("potg");
  const [voicebankOptions, setVoicebankOptions] = useState<VoicebankInfo[]>(
    DIFFSINGER_VOICEBANKS.map((option) => ({ id: option.value, label: option.label, available: true })),
  );
  const [result, setResult] = useState<LyricResponse | null>(null);
  const [demoJob, setDemoJob] = useState<DemoStatusResponse | null>(null);
  const [lyricLines, setLyricLines] = useState<string[]>([]);
  const [rhymeAnalysis, setRhymeAnalysis] = useState<RhymeLineAnalysis[]>([]);
  const [error, setError] = useState("");
  const [rhymeError, setRhymeError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzingBeat, setIsAnalyzingBeat] = useState(false);
  const [isGeneratingDemo, setIsGeneratingDemo] = useState(false);
  const [isAnalyzingRhyme, setIsAnalyzingRhyme] = useState(false);
  const [copyLabel, setCopyLabel] = useState("Copy");
  const [editingLineIndex, setEditingLineIndex] = useState<number | null>(null);

  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);

  useEffect(() => {
    const controller = new AbortController();
    void fetch(`${apiBaseUrl}/api/v1/guide-demos/voicebanks`, { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error("voicebank lookup failed"))))
      .then((items: VoicebankInfo[]) => {
        setVoicebankOptions(items);
        const firstAvailable = items.find((item) => item.available);
        setVoicebank((current) =>
          firstAvailable && !items.some((item) => item.id === current && item.available)
            ? firstAvailable.id
            : current,
        );
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [apiBaseUrl]);

  useEffect(() => {
    const controller = new AbortController();
    void fetch(`${apiBaseUrl}/api/v1/lyrics/models`, { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error("models lookup failed"))))
      .then((items: Array<{ id: string; name: string; detail: string; type: string }>) => {
        const options = items.map((item) => ({
          value: item.id,
          label: item.name,
          detail: item.detail,
        }));
        setLlmOptions(options);
        setLlm((current) => {
          const hasCurrent = options.some((opt) => opt.value === current);
          if (!hasCurrent && options.length > 0) {
            const defaultOpt = options.find((opt) => opt.value === "qwen-local") || options[0];
            return defaultOpt.value;
          }
          return current;
        });
      })
      .catch(() => {
        setLlmOptions(DEFAULT_LLM_OPTIONS);
      });
    return () => controller.abort();
  }, [apiBaseUrl]);

  useEffect(() => {
    if (
      !demoJob ||
      demoJob.status === "succeeded" ||
      demoJob.status === "failed" ||
      (demoJob.status === "queued" && demoJob.workerAvailable === false)
    ) {
      return;
    }

    const controller = new AbortController();
    const intervalId = window.setInterval(async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/v1/demos/${demoJob.jobId}`, {
          signal: controller.signal,
        });

        if (!response.ok) {
          const message = await readErrorMessage(response);
          throw new Error(message || "데모 상태 조회에 실패했습니다.");
        }

        const data = (await response.json()) as DemoStatusResponse;
        setDemoJob(data);
        if (data.status === "queued" && data.workerAvailable === false) {
          setIsGeneratingDemo(false);
        }
        if (data.status === "succeeded") {
          setIsGeneratingDemo(false);
        }
        if (data.status === "failed") {
          setError(data.error || "데모 생성에 실패했습니다.");
          setIsGeneratingDemo(false);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "데모 상태 조회 중 오류가 발생했습니다.");
          setIsGeneratingDemo(false);
        }
      }
    }, 1500);

    return () => {
      controller.abort();
      window.clearInterval(intervalId);
    };
  }, [apiBaseUrl, demoJob]);

  useEffect(() => {
    if (!result || lyricLines.length === 0) {
      return;
    }

    const controller = new AbortController();
    const timeoutId = window.setTimeout(async () => {
      setIsAnalyzingRhyme(true);
      setRhymeError("");

      try {
        const response = await fetch(`${apiBaseUrl}/api/v1/lyrics/analyze-rhyme`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ lines: lyricLines }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const message = await readErrorMessage(response);
          throw new Error(message || "라임 분석 요청에 실패했습니다.");
        }

        const data = (await response.json()) as RhymeLineAnalysis[];
        setRhymeAnalysis(data);
      } catch (err) {
        if (!controller.signal.aborted) {
          setRhymeError(err instanceof Error ? err.message : "라임 분석 중 오류가 발생했습니다.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsAnalyzingRhyme(false);
        }
      }
    }, 250);

    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
    };
  }, [apiBaseUrl, lyricLines, result]);

  async function handleGenerate(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();

    if (mode === "beat" && !beatFile) {
      setError("비트 파일을 선택해주세요.");
      return;
    }

    const parsedBpm = Number(bpm);
    if (mode === "manual" && (!Number.isInteger(parsedBpm) || parsedBpm < 40 || parsedBpm > 220)) {
      setError("BPM은 40부터 220 사이의 정수로 입력해주세요.");
      return;
    }

    setIsLoading(true);
    setError("");
    setDemoJob(null);
    setCopyLabel("Copy");
    setEditingLineIndex(null);

    try {
      const response =
        mode === "beat"
          ? await requestBeatGeneration()
          : await requestManualGeneration(parsedBpm);

      if (!response.ok) {
        const message = await readErrorMessage(response);
        throw new Error(message || "가사 생성 요청에 실패했습니다.");
      }

      const data = (await response.json()) as LyricResponse;
      setResult(data);
      setDemoJob(null);
      setLyricLines(parseLyricLines(data.lyrics));
      setRhymeAnalysis(data.rhymeAnalysis ?? []);
      setRhymeError("");
      setEditingLineIndex(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  }

  function handleModeChange(nextMode: GenerateMode) {
    setMode(nextMode);
    setResult(null);
    setDemoJob(null);
    setLyricLines([]);
    setRhymeAnalysis([]);
    setError("");
    setRhymeError("");
    setCopyLabel("Copy");
    setEditingLineIndex(null);
  }

  async function handleBeatFileChange(file: File | null) {
    setBeatFile(file);
    setBeatAnalysis(null);
    setResult(null);
    setDemoJob(null);
    setLyricLines([]);
    setRhymeAnalysis([]);
    setError("");
    setRhymeError("");
    setEditingLineIndex(null);
    if (!file) {
      return;
    }

    setIsAnalyzingBeat(true);
    const body = new FormData();
    body.append("beat", file);
    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/beats/analyze`, { method: "POST", body });
      if (!response.ok) {
        throw new Error((await readErrorMessage(response)) || "비트 분석에 실패했습니다.");
      }
      const analysis = (await response.json()) as BeatAnalysis;
      setBeatAnalysis(analysis);
      setBpm(String(Math.round(analysis.tempo)));
      setFirstBarStartSec(String(analysis.firstBarStartSec));
    } catch (err) {
      setError(err instanceof Error ? err.message : "비트 분석 중 오류가 발생했습니다.");
    } finally {
      setIsAnalyzingBeat(false);
    }
  }

  async function requestBeatGeneration() {
    const body = new FormData();
    body.append("beat", beatFile as File);
    body.append("llm", llm);
    if (topic) {
      body.append("topic", topic);
    }
    if (beatAnalysis?.tempo) {
      body.append("bpm", String(Math.round(beatAnalysis.tempo)));
    }

    return fetch(`${apiBaseUrl}/api/v1/lyrics/generate-from-beat`, {
      method: "POST",
      body,
    });
  }

  async function handleGuideDemo() {
    if (!beatFile || !result || (lyricLines.length !== 8 && lyricLines.length !== 16) || lyricLines.some((line) => !line.trim())) {
      setError("비트와 비어 있지 않은 가사(8줄 또는 16줄)가 필요합니다.");
      return;
    }
    const parsedStart = Number(firstBarStartSec);
    if (!Number.isFinite(parsedStart) || parsedStart < 0) {
      setError("첫 마디 시작 시각을 0 이상의 초 단위 값으로 입력해주세요.");
      return;
    }

    const body = new FormData();
    body.append("beat", beatFile as File);
    body.append("lyrics", ["[Verse]", ...lyricLines].join("\n"));
    body.append("bpm", String(result.bpm));
    body.append("firstBarStartSec", String(parsedStart));
    body.append("voicebank", voicebank);

    setError("");
    setIsGeneratingDemo(true);
    setDemoJob(null);
    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/guide-demos`, { method: "POST", body });
      if (!response.ok) {
        throw new Error((await readErrorMessage(response)) || "8마디 가이드 랩 생성 요청에 실패했습니다.");
      }
      const data = (await response.json()) as DemoGenerateResponse;
      setDemoJob({
        jobId: data.jobId,
        status: data.status,
        progress: 0,
        workerAvailable: true,
        workerCount: 0,
        bpm: result.bpm,
        lyrics: ["[Verse]", ...lyricLines].join("\n"),
        notes: ["8마디 SVS 작업이 대기열에 등록되었습니다."],
        error: null,
        audioUrl: null,
        vocalUrl: null,
        flowPlanUrl: null,
        voicebank,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "가이드 랩 생성 중 오류가 발생했습니다.");
      setIsGeneratingDemo(false);
    }
  }

  async function requestManualGeneration(parsedBpm: number) {
    return fetch(`${apiBaseUrl}/api/v1/lyrics/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ bpm: parsedBpm, llm, topic: topic || undefined }),
    });
  }

  async function handleCopy() {
    if (!result) {
      return;
    }

    await navigator.clipboard.writeText(["[Verse]", ...lyricLines].join("\n"));
    setCopyLabel("Copied");
    window.setTimeout(() => setCopyLabel("Copy"), 1400);
  }

  function updateLyricLine(index: number, value: string) {
    setLyricLines((current) => current.map((line, lineIndex) => (lineIndex === index ? value : line)));
    setCopyLabel("Copy");
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
                랩 벌스 만들기
              </h1>
              <p className="mt-1.5 max-w-xl text-sm leading-6 text-[#d8b993]">
                비트 분석 또는 직접 조건 입력으로 8마디 초안을 생성합니다.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/flow-test"
              className="border border-[#52d4c8]/55 px-3 py-2 text-xs font-bold tracking-[0.08em] text-[#d7fffb] transition hover:border-[#72ebd8] hover:bg-[#0b2023]"
            >
              플로우 테스트
            </Link>
            <Link
              href="/beat-analysis"
              className="border border-[#f5b950]/55 px-3 py-2 text-xs font-bold tracking-[0.08em] text-[#fff3ca] transition hover:border-[#ffb23f] hover:bg-[#23100b]"
            >
              Librosa 분석
            </Link>
            <div className="flex items-center gap-2 text-xs font-semibold tracking-[0.16em] text-[#f5b950] uppercase">
              <span className="h-2 w-2 rounded-full bg-[#ff5a1f] shadow-[0_0_16px_rgba(255,90,31,0.9)]" />
              Beat Draft
            </div>
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
                <div className="grid grid-cols-2 gap-2 rounded-sm border border-[#f5b950]/25 bg-black/25 p-1">
                  <button
                    type="button"
                    onClick={() => handleModeChange("beat")}
                    className={`h-10 border text-sm font-bold transition ${
                      mode === "beat"
                        ? "border-[#ffb23f] bg-[#ff5a1f] text-white shadow-[0_0_18px_rgba(255,90,31,0.34)]"
                        : "border-transparent bg-[#23100b] text-[#d8b993] hover:border-[#f5b950]/55 hover:text-[#fff3ca]"
                    }`}
                  >
                    비트 분석
                  </button>
                  <button
                    type="button"
                    onClick={() => handleModeChange("manual")}
                    className={`h-10 border text-sm font-bold transition ${
                      mode === "manual"
                        ? "border-[#ffb23f] bg-[#ff5a1f] text-white shadow-[0_0_18px_rgba(255,90,31,0.34)]"
                        : "border-transparent bg-[#23100b] text-[#d8b993] hover:border-[#f5b950]/55 hover:text-[#fff3ca] disabled:cursor-not-allowed disabled:text-[#6d4530]"
                    }`}
                  >
                    직접 입력
                  </button>
                </div>

                {mode === "beat" ? (
                  <>
                    <label className="block">
                      <span className="text-sm font-semibold text-[#d8b993]">Beat file</span>
                      <input
                        key="beat-file-input"
                        type="file"
                        accept="audio/*"
                        onChange={(event) => void handleBeatFileChange(event.target.files?.[0] ?? null)}
                        className="mt-2 block w-full border border-[#f5b950]/45 bg-[#130806]/88 px-3 py-3 text-sm font-semibold text-[#fff3ca] outline-none transition file:mr-4 file:border-0 file:bg-[#f5b950] file:px-3 file:py-2 file:text-sm file:font-black file:text-[#170906] hover:border-[#f5b950]/70 focus:border-[#ffb23f] focus:shadow-[0_0_0_3px_rgba(255,178,63,0.18)]"
                        aria-label="Beat file"
                      />
                    </label>
                    <p className="min-h-5 text-xs font-semibold text-[#b9865f]">
                      {isAnalyzingBeat
                        ? "BPM과 첫 마디 시작점을 분석하는 중..."
                        : beatAnalysis
                          ? `${beatFile?.name} · ${beatAnalysis.tempo.toFixed(1)} BPM · 첫 마디 ${beatAnalysis.firstBarStartSec.toFixed(3)}s`
                          : beatFile?.name ?? "MP3, WAV, M4A, AAC, FLAC"}
                    </p>
                  </>
                ) : (
                  <>
                    <label className="block">
                      <span className="text-sm font-semibold text-[#d8b993]">BPM</span>
                      <input
                        key="manual-bpm-input"
                        value={bpm}
                        onChange={(event) => {
                          setBpm(event.target.value);
                          setResult(null);
                          setLyricLines([]);
                          setRhymeAnalysis([]);
                          setError("");
                          setRhymeError("");
                          setEditingLineIndex(null);
                        }}
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
                          onClick={() => {
                            setBpm(String(preset));
                            setResult(null);
                            setLyricLines([]);
                            setRhymeAnalysis([]);
                            setError("");
                            setRhymeError("");
                          }}
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
                  </>
                )}

                <fieldset className="space-y-2">
                  <legend className="text-sm font-semibold text-[#d8b993]">가사 주제</legend>
                  <div className="flex flex-wrap gap-1.5 rounded-sm border border-[#f5b950]/25 bg-black/25 p-1.5">
                    {TOPIC_OPTIONS.map((opt) => (
                      <button
                        key={opt.value}
                        type="button"
                        onClick={() => setTopic(opt.value)}
                        className={`h-9 rounded-sm border px-3 text-xs font-bold transition ${
                          topic === opt.value
                            ? "border-[#ffb23f] bg-[#ff5a1f] text-white shadow-[0_0_14px_rgba(255,90,31,0.34)]"
                            : "border-transparent bg-[#23100b] text-[#d8b993] hover:border-[#f5b950]/55 hover:text-[#fff3ca]"
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </fieldset>

                <fieldset className="space-y-2">
                  <legend className="text-sm font-semibold text-[#d8b993]">LLM</legend>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {llmOptions.map((option) => (
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
                  disabled={isLoading || isAnalyzingBeat || (mode === "beat" && !beatFile)}
                  className="h-12 w-full border border-[#ffd78a]/55 bg-[#ff5a1f] px-4 text-sm font-black tracking-[0.08em] text-white uppercase shadow-[0_14px_34px_rgba(255,90,31,0.28)] transition hover:bg-[#ff7a28] disabled:cursor-not-allowed disabled:border-[#6d4530] disabled:bg-[#6d4530] disabled:text-[#c39a75] disabled:shadow-none"
                >
                  {isLoading ? "Generating" : "Generate 8 Bars"}
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
                  {demoJob?.bpm ? `${demoJob.bpm} BPM` : result ? `${result.bpm} BPM` : "-- BPM"} · {llm}
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
              <div className="min-h-[420px] px-5 py-5 sm:px-7 sm:py-6">
                {isLoading ? (
                  <div className="space-y-3 font-mono text-sm leading-8 text-[#fff6df]">
                    <p>
                      {mode === "beat"
                        ? "비트를 분석하고 8마디 벌스를 구성하는 중..."
                        : "입력 조건에 맞춰 8마디 벌스를 구성하는 중..."}
                    </p>
                  </div>
                ) : lyricLines.length > 0 ? (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between gap-3 text-xs font-semibold tracking-[0.12em] text-[#b9865f] uppercase">
                      <span>[Verse]</span>
                      <span>{isAnalyzingRhyme ? "Analyzing rhyme" : "Rhyme view"}</span>
                    </div>
                    {lyricLines.map((line, index) => {
                      const analysis = rhymeAnalysis[index];
                      const isEditing = editingLineIndex === index;

                      return (
                        <div
                          key={index}
                          className="border border-[#f5b950]/20 bg-[#130806]/58 px-3 py-3"
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            {isEditing ? (
                              <textarea
                                value={line}
                                rows={1}
                                onChange={(event) => updateLyricLine(index, event.target.value)}
                                className="block min-h-10 min-w-0 flex-1 resize-y border border-[#f5b950]/25 bg-black/30 px-3 py-2 font-mono text-sm leading-6 text-[#fff6df] outline-none transition focus:border-[#ffb23f] focus:shadow-[0_0_0_3px_rgba(255,178,63,0.14)]"
                                aria-label={`Lyric line ${index + 1}`}
                              />
                            ) : (
                              <div className="min-w-0 flex-1 font-mono text-sm leading-7 text-[#fff6df]">
                                {renderHighlightedLine(line, analysis)}
                              </div>
                            )}
                            <span className="shrink-0 border border-[#f5b950]/25 bg-black/25 px-2 py-1 text-[11px] font-bold text-[#b9865f]">
                              {analysis?.rhymeGroup == null
                                ? `score ${formatScore(analysis?.score)}`
                                : `R${analysis.rhymeGroup + 1} · ${formatScore(analysis.score)}`}
                            </span>
                            <button
                              type="button"
                              onClick={() => setEditingLineIndex(isEditing ? null : index)}
                              className="h-8 shrink-0 border border-[#f5b950]/45 px-3 text-xs font-bold text-[#fff3ca] transition hover:border-[#ffb23f] hover:bg-[#23100b]"
                            >
                              {isEditing ? "완료" : "수정"}
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="font-mono text-sm leading-8 text-[#fff6df]">
                    생성 방식을 선택하고 Generate를 누르면 8마디 벌스가 생성됩니다.
                  </p>
                )}
              </div>
            </div>

            {result && mode === "beat" ? (
              <div className="mt-4 space-y-3 border border-[#52d4c8]/35 bg-[#071b1a]/70 p-4">
                <div className="flex flex-wrap items-end gap-3">
                  <label className="min-w-40 flex-1">
                    <span className="text-xs font-semibold tracking-[0.12em] text-[#52d4c8] uppercase">
                      DiffSinger voicebank
                    </span>
                    <select
                      value={voicebank}
                      onChange={(event) => setVoicebank(event.target.value)}
                      className="mt-2 h-10 w-full border border-[#52d4c8]/35 bg-[#130806]/88 px-3 text-sm font-semibold text-[#fff3ca] outline-none focus:border-[#52d4c8]"
                    >
                      {voicebankOptions.map((option) => (
                        <option key={option.id} value={option.id} disabled={!option.available}>
                          {option.label}{option.available ? "" : " (not installed)"}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="w-44">
                    <span className="text-xs font-semibold tracking-[0.12em] text-[#52d4c8] uppercase">
                      First bar (sec)
                    </span>
                    <input
                      value={firstBarStartSec}
                      onChange={(event) => setFirstBarStartSec(event.target.value)}
                      inputMode="decimal"
                      className="mt-2 h-10 w-full border border-[#52d4c8]/35 bg-[#130806]/88 px-3 text-sm font-semibold text-[#fff3ca] outline-none focus:border-[#52d4c8]"
                      aria-label="First bar start seconds"
                    />
                  </label>
                  <button
                    type="button"
                    onClick={() => void handleGuideDemo()}
                    disabled={
                      isGeneratingDemo ||
                      !beatAnalysis ||
                      !voicebankOptions.some((option) => option.id === voicebank && option.available) ||
                      (lyricLines.length !== 8 && lyricLines.length !== 16) ||
                      lyricLines.some((line) => !line.trim())
                    }
                    className="h-10 border border-[#8af5eb]/55 bg-[#169c91] px-4 text-xs font-black tracking-[0.08em] text-white uppercase transition hover:bg-[#20b9ac] disabled:cursor-not-allowed disabled:border-[#315c58] disabled:bg-[#315c58] disabled:text-[#9abdb9]"
                  >
                    {isGeneratingDemo ? "Rendering 8 Bars" : "Make Guide Rap"}
                  </button>
                </div>
                <p className="text-xs leading-5 text-[#8fcac4]">
                  편집된 가사(붐뱁 8줄 / 트랩 16줄)를 사용합니다. 첫 마디 시작점을 직접 보정할 수 있습니다.
                </p>
                {demoJob && demoJob.status !== "succeeded" && demoJob.status !== "failed" ? (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-[#b9eee9]">
                      {formatDemoStatus(demoJob.status)} · {Math.round((demoJob.progress ?? 0) * 100)}%
                    </p>
                    <div className="h-2 w-full border border-[#52d4c8]/25 bg-black/30">
                      <div
                        className="h-full bg-[#52d4c8] transition-all"
                        style={{ width: `${Math.round((demoJob.progress ?? 0) * 100)}%` }}
                      />
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}

            {demoJob?.status === "succeeded" && demoJob.audioUrl ? (
              <div className="mt-4 border border-[#f5b950]/35 bg-[#130806]/78 p-3 shadow-[0_16px_36px_rgba(0,0,0,0.26)]">
                <p className="mb-2 text-xs font-bold tracking-[0.1em] text-[#f5b950] uppercase">Mixed demo</p>
                <audio controls src={toAbsoluteApiUrl(apiBaseUrl, demoJob.audioUrl)} className="w-full" />
                {demoJob.vocalUrl ? (
                  <>
                    <p className="mt-4 mb-2 text-xs font-bold tracking-[0.1em] text-[#52d4c8] uppercase">Dry vocal</p>
                    <audio controls src={toAbsoluteApiUrl(apiBaseUrl, demoJob.vocalUrl)} className="w-full" />
                  </>
                ) : null}
                <div className="mt-3 flex flex-wrap gap-2">
                  <a
                    href={toAbsoluteApiUrl(apiBaseUrl, demoJob.audioUrl)}
                    download
                    className="inline-flex h-10 items-center border border-[#ffd78a]/55 bg-[#ff5a1f] px-4 text-xs font-black tracking-[0.08em] text-white uppercase transition hover:bg-[#ff7a28]"
                  >
                    Download mix
                  </a>
                  {demoJob.vocalUrl ? (
                    <a
                      href={toAbsoluteApiUrl(apiBaseUrl, demoJob.vocalUrl)}
                      download
                      className="inline-flex h-10 items-center border border-[#52d4c8]/55 bg-[#126c65] px-4 text-xs font-black tracking-[0.08em] text-white uppercase transition hover:bg-[#18877e]"
                    >
                      Download vocal
                    </a>
                  ) : null}
                  {demoJob.flowPlanUrl ? (
                    <a
                      href={toAbsoluteApiUrl(apiBaseUrl, demoJob.flowPlanUrl)}
                      download
                      className="inline-flex h-10 items-center border border-[#f5b950]/45 px-4 text-xs font-black tracking-[0.08em] text-[#fff3ca] uppercase transition hover:bg-[#23100b]"
                    >
                      FlowPlan JSON
                    </a>
                  ) : null}
                </div>
              </div>
            ) : null}

            <div className="mt-4 min-h-12 space-y-1 text-sm text-[#b9865f]">
              {rhymeError ? <p className="text-[#ffb6a2]">{rhymeError}</p> : null}
              {demoJob && demoJob.status !== "succeeded" && demoJob.status !== "failed" ? (
                <p>{formatDemoStatus(demoJob.status)} · {Math.round((demoJob.progress ?? 0) * 100)}%</p>
              ) : null}
              {demoJob?.status === "queued" && demoJob.workerAvailable === false ? (
                <p className="text-[#ffb6a2]">
                  RQ worker가 감지되지 않았습니다. 별도 터미널에서 demo-generation worker를 실행해야 합니다.
                </p>
              ) : null}
              {demoJob?.notes.map((note) => (
                <p key={note}>{note}</p>
              ))}
              {!demoJob && result?.notes.map((note) => (
                <p key={note}>{note}</p>
              ))}
            </div>
          </section>
        </div>
      </section>
    </main>
  );
}

async function readErrorMessage(response: Response) {
  const fallback = await response.text();
  if (!fallback) {
    return "";
  }

  try {
    const parsed = JSON.parse(fallback) as ApiErrorResponse;
    if (typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch {
    return fallback;
  }

  return fallback;
}

function parseLyricLines(lyrics: string) {
  const lines = lyrics
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !line.toLowerCase().startsWith("[verse"));

  const targetLen = lines.length > 8 ? 16 : 8;
  const sliced = lines.slice(0, targetLen);
  while (sliced.length < targetLen) {
    sliced.push("");
  }
  return sliced;
}

function renderHighlightedLine(line: string, analysis?: RhymeLineAnalysis) {
  const rawRanges = analysis?.highlightRanges?.length
    ? analysis.highlightRanges
    : analysis?.highlightStart != null && analysis.highlightEnd != null
      ? [{ start: analysis.highlightStart, end: analysis.highlightEnd }]
      : [];
  const highlightRanges = normalizeHighlightRanges(rawRanges, line.length);

  if (
    !analysis ||
    analysis.rhymeGroup == null ||
    highlightRanges.length === 0
  ) {
    return <span>{line || " "}</span>;
  }

  const palette = RHYME_COLORS[analysis.rhymeGroup % RHYME_COLORS.length];
  const parts = [];
  let cursor = 0;

  for (const range of highlightRanges) {
    if (cursor < range.start) {
      parts.push(<span key={`plain-${cursor}`}>{line.slice(cursor, range.start)}</span>);
    }
    parts.push(
      <span
        key={`highlight-${range.start}-${range.end}`}
        className="border px-1 py-0.5 font-bold"
        style={{
          backgroundColor: palette.background,
          borderColor: palette.border,
          color: palette.color,
        }}
      >
        {line.slice(range.start, range.end)}
      </span>,
    );
    cursor = range.end;
  }

  if (cursor < line.length) {
    parts.push(<span key={`plain-${cursor}`}>{line.slice(cursor)}</span>);
  }

  return <>{parts}</>;
}

function normalizeHighlightRanges(ranges: Array<{ start: number; end: number }>, textLength: number) {
  const normalized = ranges
    .map((range) => ({
      start: Math.max(0, Math.min(textLength, range.start)),
      end: Math.max(0, Math.min(textLength, range.end)),
    }))
    .filter((range) => range.end > range.start)
    .sort((left, right) => left.start - right.start);

  const merged: Array<{ start: number; end: number }> = [];
  for (const range of normalized) {
    const previous = merged[merged.length - 1];
    if (previous && range.start <= previous.end) {
      previous.end = Math.max(previous.end, range.end);
    } else {
      merged.push({ ...range });
    }
  }

  return merged;
}

function formatScore(score?: number) {
  if (typeof score !== "number") {
    return "0.00";
  }
  return score.toFixed(2);
}

function formatDemoStatus(status?: DemoStatus) {
  switch (status) {
    case "queued":
      return "대기 중";
    case "analyzing":
      return "비트 분석";
    case "writing":
      return "가사 생성";
    case "planning":
      return "FlowPlan 생성";
    case "voicing":
      return "보컬 생성";
    case "rendering":
      return "DiffSinger 렌더링";
    case "converting_rvc":
      return "RVC 음색 변환";
    case "mixing":
      return "믹싱";
    case "succeeded":
      return "완료";
    case "failed":
      return "실패";
    default:
      return "준비 중";
  }
}

function toAbsoluteApiUrl(apiBaseUrl: string, path: string) {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${apiBaseUrl}${path}`;
}
