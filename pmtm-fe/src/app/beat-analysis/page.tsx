"use client";

import Image from "next/image";
import Link from "next/link";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { getApiBaseUrl } from "@/lib/api";

type TimedValue = { time: number; value: number };

type BeatAnalysis = {
  fileName: string;
  durationSec: number;
  sampleRate: number;
  sampleCount: number;
  tempo: number;
  timeSignature: string;
  timeSignatureSource: string;
  introStartSec: number;
  introEndSec: number;
  drumEntrySec: number;
  firstBeatSec: number;
  firstBarStartSec: number;
  firstBarEndSec: number;
  firstBarBeatTimes: number[];
  beatTimes: number[];
  onsetTimes: number[];
  waveform: TimedValue[];
  rms: TimedValue[];
  onsetStrength: TimedValue[];
  spectral: Record<string, number>;
  chroma: number[];
  mfcc: number[];
};

const CHROMA_LABELS = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const SPECTRAL_LABELS: Record<string, string> = {
  rmsMean: "평균 RMS",
  rmsMax: "최대 RMS",
  zeroCrossingRateMean: "평균 영교차율",
  centroidMeanHz: "평균 스펙트럼 중심",
  bandwidthMeanHz: "평균 대역폭",
  rolloffMeanHz: "평균 롤오프",
};

export default function BeatAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<BeatAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);
  const audioUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;

    setIsLoading(true);
    setError("");
    setAnalysis(null);
    const body = new FormData();
    body.append("beat", file);

    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/beats/analyze`, { method: "POST", body });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(payload?.detail || "비트 분석에 실패했습니다.");
      }
      setAnalysis((await response.json()) as BeatAnalysis);
    } catch (err) {
      setError(err instanceof Error ? err.message : "비트 분석에 실패했습니다.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="pmtm-stage min-h-screen text-[#fff6df]">
      <section className="mx-auto w-full max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 border-b border-[#f5b950]/25 pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            <div className="relative h-14 w-14 overflow-hidden border border-[#f7c76d]/45 bg-black/55 sm:h-16 sm:w-16">
              <Image src="/brand/pmtm-icon.png" alt="PMTM" fill priority sizes="64px" className="object-cover" />
            </div>
            <div>
              <p className="text-xs font-semibold tracking-[0.24em] text-[#52d4c8] uppercase">PMTM · Librosa</p>
              <h1 className="mt-1 text-2xl font-black text-[#fff3ca] sm:text-3xl">비트 분석 리포트</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Link href="/flow-test" className="border border-[#52d4c8]/55 px-4 py-2 text-sm font-bold text-[#d7fffb] transition hover:border-[#72ebd8] hover:bg-[#0b2023]">
              플로우 테스트
            </Link>
            <Link href="/" className="border border-[#f5b950]/55 px-4 py-2 text-sm font-bold transition hover:border-[#ffb23f] hover:bg-[#23100b]">
              ← 가사 생성으로
            </Link>
          </div>
        </header>

        <form onSubmit={handleSubmit} className="pmtm-panel mt-5 grid gap-4 p-4 sm:grid-cols-[1fr_auto] sm:items-end sm:p-5">
          <div>
            <label htmlFor="analysis-beat" className="text-sm font-bold text-[#d8b993]">분석할 비트 파일</label>
            <input
              id="analysis-beat"
              type="file"
              accept="audio/*"
              onChange={(event) => {
                setFile(event.target.files?.[0] ?? null);
                setAnalysis(null);
                setError("");
              }}
              className="mt-2 block w-full border border-[#f5b950]/45 bg-[#130806]/88 px-3 py-3 text-sm font-semibold file:mr-4 file:border-0 file:bg-[#f5b950] file:px-3 file:py-2 file:font-black file:text-[#170906]"
            />
            <p className="mt-2 text-xs text-[#b9865f]">최대 20MB · 분석 구간은 앞 60초 · MP3, WAV, M4A, AAC, FLAC</p>
          </div>
          <button
            type="submit"
            disabled={!file || isLoading}
            className="h-12 border border-[#ffd78a]/55 bg-[#ff5a1f] px-7 font-black text-white transition hover:bg-[#ff7a28] disabled:cursor-not-allowed disabled:opacity-45"
          >
            {isLoading ? "분석 중..." : "전체 분석"}
          </button>
        </form>

        {file && audioUrl ? <audio controls src={audioUrl} className="mt-4 w-full" /> : null}
        {error ? <p className="mt-4 border border-[#ff6b4a]/55 bg-[#2b0c08] px-4 py-3 text-sm text-[#ffb6a2]">{error}</p> : null}

        {analysis ? <AnalysisReport analysis={analysis} /> : !isLoading ? (
          <div className="mt-5 border border-dashed border-[#f5b950]/35 bg-black/20 px-6 py-16 text-center text-sm leading-7 text-[#b9865f]">
            비트를 올리면 리듬, 파형, 에너지, 음색 특징을 한 페이지에서 확인할 수 있습니다.
          </div>
        ) : null}
      </section>
    </main>
  );
}

function AnalysisReport({ analysis }: { analysis: BeatAnalysis }) {
  return (
    <div className="mt-5 space-y-4">
      <div className="border border-[#f5b950]/30 bg-black/25 px-4 py-3 font-mono text-sm text-[#d7fffb]">
        {analysis.fileName}
      </div>
      <StructureSummary analysis={analysis} />
      <FirstBarResult analysis={analysis} />
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Metric label="분석 길이" value={`${analysis.durationSec.toFixed(2)} sec`} />
        <Metric label="박자 수" value={`${analysis.beatTimes.length}`} />
        <Metric label="온셋 수" value={`${analysis.onsetTimes.length}`} />
        <Metric label="샘플레이트" value={`${analysis.sampleRate.toLocaleString()} Hz`} />
        <Metric label="샘플 수" value={analysis.sampleCount.toLocaleString()} />
      </section>

      <ChartPanel title="Waveform" description="주황색 영역이 첫 마디 후보이며 굵은 선은 시작점입니다.">
        <LineChart
          data={analysis.waveform}
          markers={analysis.beatTimes}
          highlightRange={[analysis.firstBarStartSec, analysis.firstBarEndSec]}
          primaryMarker={analysis.firstBarStartSec}
          color="#f5b950"
          symmetric
        />
      </ChartPanel>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartPanel title="RMS Energy" description="시간에 따른 음량 에너지">
          <LineChart data={analysis.rms} color="#52d4c8" />
        </ChartPanel>
        <ChartPanel title="Onset Strength" description="타격과 음의 시작 강도">
          <LineChart data={analysis.onsetStrength} markers={analysis.onsetTimes} color="#ff7a28" />
        </ChartPanel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartPanel title="Chroma" description="12개 피치 클래스의 평균 에너지">
          <BarChart values={analysis.chroma} labels={CHROMA_LABELS} color="#52d4c8" />
        </ChartPanel>
        <ChartPanel title="MFCC" description="13개 음색 계수의 평균값">
          <BarChart values={analysis.mfcc} labels={analysis.mfcc.map((_, index) => String(index + 1))} color="#ff7a28" signed />
        </ChartPanel>
      </div>

      <section className="pmtm-panel p-4 sm:p-5">
        <h2 className="text-lg font-black text-[#fff3ca]">Spectral Summary</h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(analysis.spectral).map(([key, value]) => (
            <Metric key={key} label={SPECTRAL_LABELS[key] ?? key} value={`${formatNumber(value)}${key.endsWith("Hz") ? " Hz" : ""}`} />
          ))}
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <TimeList title="Beat Times" values={analysis.beatTimes} />
        <TimeList title="Onset Times" values={analysis.onsetTimes} />
      </div>

      <details className="pmtm-panel p-4 sm:p-5">
        <summary className="cursor-pointer font-black text-[#fff3ca]">원본 분석 JSON 보기</summary>
        <pre className="mt-4 max-h-[520px] overflow-auto bg-black/35 p-4 text-xs leading-5 text-[#d7fffb]">{JSON.stringify(analysis, null, 2)}</pre>
      </details>
    </div>
  );
}

function StructureSummary({ analysis }: { analysis: BeatAnalysis }) {
  const duration = Math.max(analysis.durationSec, 0.001);
  const introWidth = Math.min(100, (analysis.introEndSec / duration) * 100);
  const drumPosition = Math.min(100, (analysis.drumEntrySec / duration) * 100);
  const firstBeatPosition = Math.min(100, (analysis.firstBeatSec / duration) * 100);

  return (
    <section className="pmtm-panel p-4 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-xs font-bold tracking-[0.14em] text-[#52d4c8] uppercase">Song Structure</p>
          <h2 className="mt-1 text-xl font-black text-[#fff3ca]">리듬 구조</h2>
        </div>
        <p className="text-xs text-[#b9865f]">박자표는 기본 가정, 시간 위치는 Librosa 기반 추정입니다.</p>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StructureMetric label="박자표" value={analysis.timeSignature} badge={analysis.timeSignatureSource === "assumed" ? "가정" : "추정"} />
        <StructureMetric label="BPM" value={analysis.tempo.toFixed(2)} badge="Librosa" accent />
        <StructureMetric label="인트로" value={`${analysis.introStartSec.toFixed(2)}s – ${analysis.introEndSec.toFixed(2)}s`} badge="추정" />
        <StructureMetric label="첫 마디 시작" value={`${analysis.firstBeatSec.toFixed(2)}s`} badge="후보" accent />
      </div>

      <div className="mt-6">
        <div className="relative h-12 overflow-hidden border border-[#f5b950]/30 bg-[#130806]">
          <div className="absolute inset-y-0 left-0 bg-[#52d4c8]/25" style={{ width: `${introWidth}%` }} />
          <div className="absolute inset-y-0 w-px bg-[#f5b950]" style={{ left: `${drumPosition}%` }} />
          <div className="absolute inset-y-0 w-0.5 bg-[#ff5a1f] shadow-[0_0_12px_rgba(255,90,31,.8)]" style={{ left: `${firstBeatPosition}%` }} />
          <span className="absolute left-2 top-1/2 -translate-y-1/2 text-xs font-bold text-[#d7fffb]">INTRO</span>
          <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs font-bold text-[#d8b993]">MAIN BEAT</span>
        </div>
        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs text-[#b9865f]">
          <span><b className="text-[#f5b950]">드럼 진입</b> {analysis.drumEntrySec.toFixed(2)}s</span>
          <span><b className="text-[#ff7a28]">첫 마디 시작 후보</b> {analysis.firstBeatSec.toFixed(2)}s</span>
        </div>
      </div>
    </section>
  );
}

function FirstBarResult({ analysis }: { analysis: BeatAnalysis }) {
  return (
    <section className="border border-[#ff7a28]/60 bg-[linear-gradient(135deg,rgba(255,90,31,.18),rgba(19,8,6,.92))] p-4 shadow-[0_18px_50px_rgba(255,90,31,.12)] sm:p-5">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-black tracking-[0.16em] text-[#ff9a55] uppercase">First Bar Candidate</p>
          <h2 className="mt-1 text-2xl font-black text-[#fff3ca]">첫 마디가 들어갈 자리</h2>
        </div>
        <p className="font-mono text-xl font-black text-[#ff9a55]">
          {analysis.firstBarStartSec.toFixed(3)}s – {analysis.firstBarEndSec.toFixed(3)}s
        </p>
      </div>
      <p className="mt-3 text-sm leading-6 text-[#d8b993]">
        지속적인 드럼 진입점을 찾고 가장 가까운 박자를 1박으로 잡은 4/4 첫 마디 후보입니다.
      </p>
      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {analysis.firstBarBeatTimes.map((time, index) => (
          <div key={index} className={`border p-3 ${index === 0 ? "border-[#ff7a28] bg-[#ff5a1f]/20" : "border-[#f5b950]/25 bg-black/25"}`}>
            <p className="text-xs font-bold text-[#b9865f]">{index + 1}박</p>
            <p className="mt-1 font-mono text-lg font-black text-[#fff3ca]">{time.toFixed(3)}s</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function StructureMetric({ label, value, badge, accent = false }: { label: string; value: string; badge: string; accent?: boolean }) {
  return (
    <div className="border border-[#f5b950]/30 bg-black/25 p-4">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-bold tracking-[0.12em] text-[#b9865f] uppercase">{label}</p>
        <span className="border border-[#f5b950]/30 px-1.5 py-0.5 text-[10px] font-bold text-[#d8b993]">{badge}</span>
      </div>
      <p className={`mt-3 text-2xl font-black ${accent ? "text-[#ff7a28]" : "text-[#fff3ca]"}`}>{value}</p>
    </div>
  );
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div className="border border-[#f5b950]/30 bg-[#130806]/86 p-4"><p className="text-xs font-bold tracking-[0.12em] text-[#b9865f] uppercase">{label}</p><p className={`mt-2 text-xl font-black ${accent ? "text-[#ff7a28]" : "text-[#fff3ca]"}`}>{value}</p></div>;
}

function ChartPanel({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return <section className="pmtm-panel p-4 sm:p-5"><h2 className="text-lg font-black text-[#fff3ca]">{title}</h2><p className="mt-1 text-xs text-[#b9865f]">{description}</p><div className="mt-4">{children}</div></section>;
}

function LineChart({ data, markers = [], highlightRange, primaryMarker, color, symmetric = false }: { data: TimedValue[]; markers?: number[]; highlightRange?: [number, number]; primaryMarker?: number; color: string; symmetric?: boolean }) {
  if (data.length === 0) return <p className="text-sm text-[#b9865f]">데이터 없음</p>;
  const width = 900;
  const height = 220;
  const duration = data[data.length - 1]?.time || 1;
  const values = data.map((point) => point.value);
  const max = symmetric ? Math.max(...values.map(Math.abs), 0.000001) : Math.max(...values, 0.000001);
  const min = symmetric ? -max : Math.min(0, ...values);
  const range = max - min || 1;
  const points = data.map((point) => `${(point.time / duration) * width},${height - ((point.value - min) / range) * height}`).join(" ");
  return <svg viewBox={`0 0 ${width} ${height}`} className="h-52 w-full bg-black/25" role="img">
    {highlightRange ? <rect x={(highlightRange[0] / duration) * width} y="0" width={Math.max(0, ((highlightRange[1] - highlightRange[0]) / duration) * width)} height={height} fill="rgba(255,90,31,.16)" /> : null}
    <line x1="0" y1={height - ((0 - min) / range) * height} x2={width} y2={height - ((0 - min) / range) * height} stroke="rgba(245,185,80,.2)" />
    {markers.map((time, index) => <line key={index} x1={(time / duration) * width} x2={(time / duration) * width} y1="0" y2={height} stroke="rgba(255,90,31,.45)" strokeWidth="1" />)}
    {primaryMarker == null ? null : <line x1={(primaryMarker / duration) * width} x2={(primaryMarker / duration) * width} y1="0" y2={height} stroke="#ff5a1f" strokeWidth="4" />}
    <polyline points={points} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
  </svg>;
}

function BarChart({ values, labels, color, signed = false }: { values: number[]; labels: string[]; color: string; signed?: boolean }) {
  const max = Math.max(...values.map(Math.abs), 0.000001);
  return <div className="flex h-52 items-stretch gap-1 border-b border-[#f5b950]/20 bg-black/25 px-2 pt-3">
    {values.map((value, index) => {
      const height = `${Math.max(2, (Math.abs(value) / max) * (signed ? 46 : 88))}%`;
      return <div key={index} className="flex min-w-0 flex-1 flex-col items-center justify-end gap-1" title={`${labels[index]}: ${value}`}>
        {signed && value >= 0 ? <div className="flex h-1/2 w-full items-end"><div className="w-full" style={{ height, background: color }} /></div> : null}
        {signed && value < 0 ? <div className="h-1/2 w-full"><div className="w-full" style={{ height, background: color }} /></div> : null}
        {!signed ? <div className="w-full" style={{ height, background: color }} /> : null}
        <span className="text-[10px] text-[#b9865f]">{labels[index]}</span>
      </div>;
    })}
  </div>;
}

function TimeList({ title, values }: { title: string; values: number[] }) {
  return <section className="pmtm-panel p-4 sm:p-5"><div className="flex items-end justify-between"><h2 className="text-lg font-black text-[#fff3ca]">{title}</h2><span className="text-xs text-[#b9865f]">{values.length} points</span></div><div className="mt-4 max-h-48 overflow-auto"><div className="flex flex-wrap gap-2">{values.map((value, index) => <span key={index} className="border border-[#f5b950]/25 bg-black/25 px-2 py-1 font-mono text-xs text-[#d7fffb]">{value.toFixed(3)}s</span>)}</div></div></section>;
}

function formatNumber(value: number) {
  return Math.abs(value) >= 100 ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}
