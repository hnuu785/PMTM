"use client";

import Image from "next/image";
import Link from "next/link";
import { SubmitEvent, useEffect, useMemo, useState } from "react";
import { getApiBaseUrl } from "@/lib/api";

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

type VoicebankInfo = {
  id: string;
  label: string;
  available: boolean;
};

type RvcModelInfo = {
  id: string;
  label: string;
  available: boolean;
};

type ApiErrorResponse = {
  detail?: unknown;
};

const DEFAULT_LYRICS = `새벽을 가로질러 달리는 이 비트
내 목소리로 채워가는 새로운 시트
두려움 따윈 접어두고 앞으로 가
결국엔 내가 원했던 곳에 닿아
한 번 더 볼륨을 높여서 노래해
어둠을 걷어내고 밝게 빛나네
마지막 마디까지 전부 쏟아내
이제 내 목소리로 세상을 물들여`;

const DIFFSINGER_VOICEBANKS = [
  { value: "potg", label: "POTG" },
  { value: "kitane", label: "KITANE" },
  { value: "rang", label: "RANG" },
  { value: "lunar", label: "LUNAR" },
];

export default function FlowTestPage() {
  const [lyrics, setLyrics] = useState(DEFAULT_LYRICS);
  const [bpm, setBpm] = useState("90");
  const [firstBarStartSec, setFirstBarStartSec] = useState("1.25");
  const [voicebank, setVoicebank] = useState("potg");
  const [rvcModelId, setRvcModelId] = useState("none");
  const [beatFile, setBeatFile] = useState<File | null>(null);
  
  const [voicebankOptions, setVoicebankOptions] = useState<VoicebankInfo[]>(
    DIFFSINGER_VOICEBANKS.map((option) => ({ id: option.value, label: option.label, available: true })),
  );
  const [rvcModelOptions, setRvcModelOptions] = useState<RvcModelInfo[]>([]);

  const [isLoading, setIsLoading] = useState(false);
  const [demoJob, setDemoJob] = useState<DemoStatusResponse | null>(null);
  const [error, setError] = useState("");

  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);

  // Sync voicebanks availability
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

  // Sync RVC models
  useEffect(() => {
    const controller = new AbortController();
    void fetch(`${apiBaseUrl}/api/v1/guide-demos/rvc-models`, { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error("rvc lookup failed"))))
      .then((items: RvcModelInfo[]) => {
        setRvcModelOptions(items);
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [apiBaseUrl]);

  // Poll job status
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
        if (data.status === "succeeded" || data.status === "failed") {
          setIsLoading(false);
          if (data.status === "failed" && data.error) {
            setError(data.error);
          }
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "데모 상태 조회 중 오류가 발생했습니다.");
          setIsLoading(false);
        }
      }
    }, 1500);

    return () => {
      controller.abort();
      window.clearInterval(intervalId);
    };
  }, [apiBaseUrl, demoJob]);

  async function readErrorMessage(response: Response): Promise<string> {
    try {
      const payload = (await response.json()) as ApiErrorResponse;
      if (payload && typeof payload.detail === "string") {
        return payload.detail;
      }
      if (payload && typeof payload.detail === "object" && payload.detail !== null) {
        return JSON.stringify(payload.detail);
      }
    } catch {
      // ignore
    }
    return "";
  }

  // Count active lines (excluding empty lines and verse tags)
  const lineCount = useMemo(() => {
    return lyrics
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.toLowerCase().startsWith("[verse")).length;
  }, [lyrics]);

  async function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    if (!beatFile) {
      setError("테스트할 비트 오디오 파일을 업로드해주세요.");
      return;
    }

    const parsedBpm = Number(bpm);
    if (!Number.isInteger(parsedBpm) || parsedBpm < 40 || parsedBpm > 220) {
      setError("BPM은 40부터 220 사이의 정수여야 합니다.");
      return;
    }

    const parsedStart = Number(firstBarStartSec);
    if (!Number.isFinite(parsedStart) || parsedStart < 0) {
      setError("첫 마디 시작 오프셋은 0 이상의 소수여야 합니다.");
      return;
    }

    if (lineCount !== 8 && lineCount !== 16) {
      setError(`가이드는 8줄(붐뱁) 또는 16줄(트랩)의 가사만 지원합니다. (현재: ${lineCount}줄)`);
      return;
    }

    setIsLoading(true);
    setDemoJob(null);

    const body = new FormData();
    body.append("beat", beatFile);
    // Automatically wrap lyrics with [Verse] tag for the SVS planner
    const formattedLyrics = ["[Verse]", ...lyrics.split("\n").map(l => l.trim()).filter(l => l.length > 0)].join("\n");
    body.append("lyrics", formattedLyrics);
    body.append("bpm", String(parsedBpm));
    body.append("firstBarStartSec", String(parsedStart));
    body.append("voicebank", voicebank);
    if (rvcModelId && rvcModelId !== "none") {
      body.append("rvcModelId", rvcModelId);
    }
    body.append("genre", lineCount === 16 ? "trap" : "boom_bap");

    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/guide-demos`, { method: "POST", body });
      if (!response.ok) {
        throw new Error((await readErrorMessage(response)) || "가이드 랩 생성 요청에 실패했습니다.");
      }
      const data = (await response.json()) as { jobId: string; status: DemoStatus };
      setDemoJob({
        jobId: data.jobId,
        status: data.status,
        progress: 0,
        workerAvailable: true,
        workerCount: 1,
        bpm: parsedBpm,
        lyrics: formattedLyrics,
        notes: ["작업이 대기열에 진입했습니다."],
        error: null,
        audioUrl: null,
        vocalUrl: null,
        flowPlanUrl: null,
        voicebank,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
      setIsLoading(false);
    }
  }

  return (
    <main className="pmtm-stage min-h-screen text-[#fff6df]">
      <section className="mx-auto w-full max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
        {/* Navigation & Header */}
        <header className="flex flex-col gap-4 border-b border-[#f5b950]/25 pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            <div className="relative h-14 w-14 overflow-hidden border border-[#52d4c8]/45 bg-black/55 sm:h-16 sm:w-16">
              <Image src="/brand/pmtm-icon.png" alt="PMTM" fill priority sizes="64px" className="object-cover" />
            </div>
            <div>
              <p className="text-xs font-semibold tracking-[0.24em] text-[#52d4c8] uppercase">PMTM · SVS</p>
              <h1 className="mt-1 text-2xl font-black text-[#fff3ca] sm:text-3xl">플로우 보컬 테스트</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Link href="/beat-analysis" className="border border-[#f5b950]/55 px-4 py-2 text-sm font-bold transition hover:border-[#ffb23f] hover:bg-[#23100b]">
              Librosa 분석
            </Link>
            <Link href="/" className="border border-[#52d4c8]/55 px-4 py-2 text-sm font-bold text-[#d7fffb] transition hover:border-[#72ebd8] hover:bg-[#0b2023]">
              ← 가사 생성으로
            </Link>
          </div>
        </header>

        {/* Layout Grid */}
        <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_450px]">
          {/* Form Side */}
          <form onSubmit={handleSubmit} className="pmtm-panel space-y-5 p-5">
            <h2 className="text-lg font-black text-[#fff3ca]">1. 가이드 조건 입력</h2>

            {/* Lyrics Area */}
            <div>
              <div className="flex justify-between items-end">
                <label htmlFor="lyrics-input" className="text-sm font-bold text-[#d8b993]">랩 가사 (8줄 또는 16줄)</label>
                <span className={`text-xs font-bold ${lineCount === 8 || lineCount === 16 ? "text-[#52d4c8]" : "text-[#ff5a1f]"}`}>
                  현재: {lineCount} / 8 또는 16 줄
                </span>
              </div>
              <textarea
                id="lyrics-input"
                rows={10}
                value={lyrics}
                onChange={(e) => setLyrics(e.target.value)}
                placeholder="1줄이 1마디에 대응됩니다. 한글 및 영문 가사를 입력해주세요."
                className="mt-2 block w-full border border-[#f5b950]/45 bg-[#130806]/88 p-4 text-sm font-semibold leading-relaxed focus:border-[#ff5a1f] focus:outline-none font-mono"
              />
              <p className="mt-1.5 text-xs text-[#b9865f]">
                ※ 한글 및 영문 가사(POTG 발음 음소)를 지원합니다.
              </p>
            </div>


            {/* Grid options */}
            <div className="grid gap-4 sm:grid-cols-4">
              <div>
                <label htmlFor="bpm-input" className="text-sm font-bold text-[#d8b993]">BPM</label>
                <input
                  id="bpm-input"
                  type="number"
                  min="40"
                  max="220"
                  value={bpm}
                  onChange={(e) => setBpm(e.target.value)}
                  className="mt-2 block w-full border border-[#f5b950]/45 bg-[#130806]/88 px-3 py-2 text-sm font-semibold focus:border-[#ff5a1f] focus:outline-none"
                />
              </div>

              <div>
                <label htmlFor="offset-input" className="text-sm font-bold text-[#d8b993]">첫 마디 시작 (초)</label>
                <input
                  id="offset-input"
                  type="number"
                  step="0.01"
                  min="0"
                  value={firstBarStartSec}
                  onChange={(e) => setFirstBarStartSec(e.target.value)}
                  className="mt-2 block w-full border border-[#f5b950]/45 bg-[#130806]/88 px-3 py-2 text-sm font-semibold focus:border-[#ff5a1f] focus:outline-none"
                />
              </div>

              <div>
                <label htmlFor="voicebank-input" className="text-sm font-bold text-[#d8b993]">보이스뱅크 (SVS)</label>
                <select
                  id="voicebank-input"
                  value={voicebank}
                  onChange={(e) => setVoicebank(e.target.value)}
                  className="mt-2 block w-full border border-[#f5b950]/45 bg-[#130806]/88 px-3 py-2 text-sm font-semibold focus:border-[#ff5a1f] focus:outline-none text-white"
                >
                  {voicebankOptions.map((opt) => (
                    <option key={opt.id} value={opt.id} disabled={!opt.available} className="bg-[#130806]">
                      {opt.label} {!opt.available ? "(준비 안 됨)" : ""}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="rvc-model-input" className="text-sm font-bold text-[#d8b993]">RVC 음색 변환</label>
                <select
                  id="rvc-model-input"
                  value={rvcModelId}
                  onChange={(e) => setRvcModelId(e.target.value)}
                  className="mt-2 block w-full border border-[#f5b950]/45 bg-[#130806]/88 px-3 py-2 text-sm font-semibold focus:border-[#ff5a1f] focus:outline-none text-white"
                >
                  <option value="none" className="bg-[#130806]">적용 안 함 (기본)</option>
                  {rvcModelOptions.map((opt) => (
                    <option key={opt.id} value={opt.id} disabled={!opt.available} className="bg-[#130806]">
                      {opt.label} {!opt.available ? "(준비 안 됨)" : ""}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Beat Upload */}
            <div>
              <label htmlFor="beat-input" className="text-sm font-bold text-[#d8b993]">배경 비트 MR (.wav, .mp3 등)</label>
              <input
                id="beat-input"
                type="file"
                accept="audio/*"
                onChange={(e) => setBeatFile(e.target.files?.[0] ?? null)}
                className="mt-2 block w-full border border-[#f5b950]/45 bg-[#130806]/88 px-3 py-2 text-sm font-semibold file:mr-4 file:border-0 file:bg-[#f5b950] file:px-3 file:py-1 file:font-black file:text-[#170906]"
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="border border-[#ff6b4a]/55 bg-[#2b0c08] px-4 py-3 text-sm text-[#ffb6a2]">
                {error}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading || (lineCount !== 8 && lineCount !== 16) || !beatFile}
              className="w-full h-12 border border-[#ffd78a]/55 bg-[#ff5a1f] font-black text-white transition hover:bg-[#ff7a28] disabled:cursor-not-allowed disabled:opacity-40 shadow-[0_4px_20px_rgba(255,90,31,0.25)]"
            >
              {isLoading ? "렌더링 중..." : "가이드 랩 보컬 생성"}
            </button>
          </form>

          {/* Result Side */}
          <div className="space-y-6">
            {/* Info panel */}
            <div className="pmtm-panel p-5 border border-[#52d4c8]/25 bg-black/15">
              <h2 className="text-md font-black text-[#52d4c8] uppercase tracking-wider">💡 안내 및 규칙</h2>
              <ul className="mt-3 text-xs leading-relaxed text-[#d8b993] list-disc list-inside space-y-2">
                <li>가사는 **8줄(붐뱁) 또는 16줄(트랩)**이어야 하며, 공백 라인은 제외됩니다.</li>
                <li>**1줄은 정확히 1마디(1 Bar)**로 자동 변환되어 리듬을 타고 노래합니다.</li>

                <li>마디 당 음절 수에 따라 **스윙 그루브 템플릿**이 동적으로 적용됩니다.</li>
                <li>로컬 서버의 렌더링 워커(`rq worker`)가 실행 중이어야 결과물이 나옵니다.</li>
              </ul>
            </div>

            {/* Status & Player Panel */}
            {demoJob && (
              <div className="pmtm-panel p-5 space-y-4">
                <h2 className="text-lg font-black text-[#fff3ca] flex items-center gap-2">
                  <span>작업 현황</span>
                  <span className={`inline-block h-2 w-2 rounded-full ${
                    demoJob.status === "succeeded" ? "bg-[#52d4c8]" : demoJob.status === "failed" ? "bg-[#ff5a1f]" : "bg-[#f5b950] animate-pulse"
                  }`} />
                </h2>

                <div className="space-y-2 text-xs font-mono text-[#d8b993]">
                  <div>작업 ID: <span className="text-white">{demoJob.jobId}</span></div>
                  <div>현재 상태: <span className={`font-bold ${
                    demoJob.status === "succeeded" ? "text-[#52d4c8]" : demoJob.status === "failed" ? "text-[#ff5a1f]" : "text-[#f5b950]"
                  }`}>{demoJob.status.toUpperCase()}</span></div>
                  
                  {/* Progress bar */}
                  <div className="w-full bg-black/55 h-1.5 rounded-full overflow-hidden mt-1 border border-white/5">
                    <div
                      className="bg-gradient-to-r from-[#ff5a1f] to-[#52d4c8] h-full transition-all duration-300"
                      style={{ width: `${demoJob.progress * 100}%` }}
                    />
                  </div>
                </div>

                {/* API Notes */}
                {demoJob.notes && demoJob.notes.length > 0 && (
                  <div className="bg-black/35 p-3 rounded text-[11px] font-mono leading-relaxed text-[#d7fffb] border border-white/5 max-h-32 overflow-y-auto">
                    {demoJob.notes.map((note, idx) => (
                      <div key={idx}>• {note}</div>
                    ))}
                  </div>
                )}

                {/* Audio Players when Succeeded */}
                {demoJob.status === "succeeded" && (
                  <div className="space-y-4 pt-3 border-t border-white/10">
                    <div>
                      <label className="block text-xs font-bold text-[#d8b993] mb-1.5">🎵 완성된 가이드 랩 데모 (보컬 + 비트)</label>
                      <audio controls src={`${apiBaseUrl}${demoJob.audioUrl}`} className="w-full mt-1 accent-[#ff5a1f]" />
                    </div>

                    <div>
                      <label className="block text-xs font-bold text-[#d8b993] mb-1.5">🎙️ 드라이 보컬 전용 (아카펠라)</label>
                      <audio controls src={`${apiBaseUrl}${demoJob.vocalUrl}`} className="w-full mt-1 accent-[#52d4c8]" />
                    </div>

                    <div className="pt-2">
                      <a
                        href={`${apiBaseUrl}${demoJob.flowPlanUrl}`}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center justify-center gap-1.5 border border-[#52d4c8]/55 px-3 py-2 text-xs font-black text-[#d7fffb] transition hover:bg-[#52d4c8]/10 w-full"
                      >
                        📄 플로우 계획 데이터 다운로드 (flow-plan.json)
                      </a>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
