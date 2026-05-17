import { getApiBaseUrl } from "@/lib/api";

const pillars = [
  "장르와 감정에 맞는 가사 초안 생성",
  "후렴구, 벌스, 브리지 단위 재작성",
  "LLM 연결 전 프롬프트 실험용 초기 UI",
];

export default function Home() {
  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_#fff2d6,_#f3ebe0_45%,_#e1d8cb_100%)] text-stone-900">
      <section className="mx-auto flex min-h-screen max-w-6xl flex-col justify-center gap-10 px-6 py-16 md:px-10">
        <div className="inline-flex w-fit rounded-full border border-stone-400/70 bg-white/70 px-4 py-2 text-sm tracking-[0.18em] text-stone-600 uppercase backdrop-blur">
          PMTM Lyric Lab
        </div>

        <div className="grid gap-8 md:grid-cols-[1.2fr_0.8fr] md:items-end">
          <div className="space-y-6">
            <h1 className="max-w-3xl text-5xl leading-tight font-semibold md:text-7xl">
              작사 아이디어를
              <br />
              제품으로 바꾸는
              <br />
              초기 골격
            </h1>
            <p className="max-w-2xl text-lg leading-8 text-stone-700">
              프론트엔드는 Next.js, 백엔드는 FastAPI로 분리했습니다. 지금은
              기능보다 흐름 검증을 우선하는 상태입니다.
            </p>
          </div>

          <div className="rounded-[2rem] border border-stone-300/80 bg-stone-950 p-6 text-stone-50 shadow-2xl shadow-stone-600/10">
            <p className="text-sm text-stone-400">Backend endpoint</p>
            <p className="mt-2 break-all font-mono text-sm">{getApiBaseUrl()}</p>
            <div className="mt-6 grid gap-3">
              {pillars.map((item) => (
                <div
                  key={item}
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-stone-200"
                >
                  {item}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
