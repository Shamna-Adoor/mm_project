import { useState, useEffect, useCallback } from 'react'
import MiniGame from './MiniGame'

/* ── Fun facts fallback (used if API is unreachable) ───────────────────── */
const FALLBACK_FACTS = [
  "YouTube users upload 500 hours of video every single minute.",
  "The human brain processes an image in just 13 milliseconds — 60,000× faster than reading text.",
  "Netflix saves $1 billion per year by using AI to pick personalized thumbnails.",
  "Whisper, the AI powering this app's transcription, was trained on 680,000 hours of audio.",
  "The first video ever uploaded to YouTube on April 23, 2005 was just 18 seconds long.",
  "60% of all internet traffic worldwide is video streaming.",
  "Viewers retain 95% of a message delivered in video, versus only 10% in text.",
  "The word 'broadcast' originally meant scattering seeds widely across a field.",
  "AI has transcribed speech more accurately than professional human transcribers since 2017.",
  "A 1-minute video contains as much information as 1.8 million words of text.",
  "The human eye can detect up to 1,000 frames per second — far more than any screen shows.",
  "The first film with synchronized sound ('The Jazz Singer') debuted in 1927.",
  "Silent film studios painted actors' lips dark red — it showed up better on black-and-white film.",
  "TikTok's recommendation algorithm processes signals in under 200 milliseconds.",
  "The average YouTube video is 7 minutes long, but average watch time is only 4 minutes.",
]

async function fetchFunFact(): Promise<string> {
  try {
    const res = await fetch('https://uselessfacts.jsph.pl/api/v2/facts/random?language=en', {
      signal: AbortSignal.timeout(4000),
    })
    if (!res.ok) throw new Error('status ' + res.status)
    const data = await res.json()
    const text = (data.text as string).trim()
    return text.length > 20 ? text : FALLBACK_FACTS[Math.floor(Math.random() * FALLBACK_FACTS.length)]
  } catch {
    return FALLBACK_FACTS[Math.floor(Math.random() * FALLBACK_FACTS.length)]
  }
}

/* ── Stage icons & labels ──────────────────────────────────────────────── */
const STAGE_META: Record<number, { icon: string; label: string; short: string }> = {
  1: { icon: '🎵', label: 'Extracting audio',             short: 'Audio' },
  2: { icon: '🗣️', label: 'Transcribing speech with AI', short: 'Transcribe' },
  3: { icon: '🔇', label: 'Detecting silence',            short: 'Silence' },
  4: { icon: '🎼', label: 'Analyzing music',              short: 'Music' },
  5: { icon: '🤖', label: 'AI classifying segments',      short: 'Classify' },
  6: { icon: '🎞️', label: 'Extracting video frames',     short: 'Frames' },
  7: { icon: '🎬', label: 'Detecting scene changes',      short: 'Scenes' },
  8: { icon: '🔍', label: 'OCR on video frames',          short: 'OCR' },
  9: { icon: '🧩', label: 'Fusing all signals',           short: 'Fusion' },
}

/* ── Hourglass SVG ─────────────────────────────────────────────────────── */
function HourglassSVG({ pct }: { pct: number }) {
  const particles = [
    { cx: 52, delay: 0,    size: 2.5, color: '#3b82f6' },
    { cx: 44, delay: 0.4,  size: 2,   color: '#06b6d4' },
    { cx: 60, delay: 0.7,  size: 2,   color: '#8b5cf6' },
    { cx: 48, delay: 1.0,  size: 1.8, color: '#3b82f6' },
    { cx: 56, delay: 1.4,  size: 1.5, color: '#06b6d4' },
    { cx: 50, delay: 1.8,  size: 2.2, color: '#a78bfa' },
  ]

  // Fill level in bottom chamber based on pct (0-1)
  const fillLevel = Math.min(0.95, pct)

  return (
    <svg
      viewBox="0 0 104 136"
      width="130"
      height="170"
      className="hourglass-glow"
      style={{ overflow: 'visible' }}
    >
      {/* Outer dashed orbit ring */}
      <circle
        cx="52" cy="68" r="58"
        fill="none"
        stroke="#1e3a5f"
        strokeWidth="1"
        strokeDasharray="6 4"
        style={{ animation: 'scanSweep 12s linear infinite', transformOrigin: '52px 68px' }}
      />

      {/* Scanning arc */}
      <path
        d="M 52 10 A 58 58 0 0 1 110 68"
        fill="none"
        stroke="url(#scanGrad)"
        strokeWidth="2"
        strokeLinecap="round"
        style={{ animation: 'scanSweep 3s linear infinite', transformOrigin: '52px 68px' }}
      />

      {/* Orbit dots */}
      {[0, 72, 144, 216, 288].map((deg, i) => (
        <circle
          key={i}
          cx="52" cy="16"
          r={i % 2 === 0 ? 2.5 : 1.8}
          fill={i % 3 === 0 ? '#3b82f6' : i % 3 === 1 ? '#06b6d4' : '#8b5cf6'}
          style={{
            '--start-deg': `${deg}deg`,
            animation: 'orbitDot 4s linear infinite',
            animationDelay: `${-i * 0.8}s`,
            transformOrigin: '52px 68px',
          } as React.CSSProperties}
        />
      ))}

      {/* Hourglass frame — top half */}
      <polygon
        points="8,6 96,6 52,68"
        fill="rgba(15,23,42,0.9)"
        stroke="#3b82f6"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      {/* Hourglass frame — bottom half */}
      <polygon
        points="8,130 96,130 52,68"
        fill="rgba(15,23,42,0.9)"
        stroke="#3b82f6"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />

      {/* Top & bottom caps */}
      <rect x="6" y="3" width="92" height="6" rx="2" fill="#1e40af" />
      <rect x="6" y="127" width="92" height="6" rx="2" fill="#1e40af" />

      {/* Sand fill in bottom chamber (grows with progress) */}
      <clipPath id="bottomChamberClip">
        <polygon points="8,130 96,130 52,68" />
      </clipPath>
      <rect
        x="8" y={130 - (130 - 68) * fillLevel}
        width="88" height={(130 - 68) * fillLevel}
        fill="url(#sandGrad)"
        opacity="0.6"
        clipPath="url(#bottomChamberClip)"
      />

      {/* Sand particles falling in top chamber */}
      {particles.map((p, i) => (
        <circle
          key={i}
          cx={p.cx}
          cy="18"
          r={p.size}
          fill={p.color}
          className="sand-particle"
          style={{ animationDelay: `${p.delay}s`, opacity: 0 }}
        />
      ))}

      {/* Neck glow point */}
      <circle cx="52" cy="68" r="3.5" fill="#06b6d4" opacity="0.9"
        style={{ animation: 'glowPulse 1.2s ease-in-out infinite' }} />
      <circle cx="52" cy="68" r="1.5" fill="white" />

      {/* Gradient defs */}
      <defs>
        <linearGradient id="scanGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#3b82f6" stopOpacity="0" />
          <stop offset="100%" stopColor="#06b6d4" stopOpacity="1" />
        </linearGradient>
        <linearGradient id="sandGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.7" />
        </linearGradient>
      </defs>
    </svg>
  )
}

/* ── Fact card ─────────────────────────────────────────────────────────── */
function FactCard({ fact, countdown }: { fact: string; countdown: number }) {
  return (
    <div
      key={fact}
      className="fact-card w-full rounded-xl px-4 py-3 text-left"
      style={{
        background: 'linear-gradient(135deg, rgba(30,58,95,0.6), rgba(15,23,42,0.8))',
        border: '1px solid rgba(59,130,246,0.25)',
        boxShadow: '0 0 24px rgba(59,130,246,0.08)',
      }}
    >
      <div className="flex items-center gap-1.5 mb-1.5">
        <span className="text-sm">💡</span>
        <span className="text-[10px] font-bold uppercase tracking-widest text-blue-400">Did You Know?</span>
      </div>
      <p className="text-xs text-slate-200 leading-relaxed">"{fact}"</p>
      <div className="mt-2 h-0.5 rounded-full overflow-hidden" style={{ background: '#1e3a5f' }}>
        <div
          className="h-full rounded-full"
          style={{
            width: `${(countdown / 15) * 100}%`,
            background: 'linear-gradient(90deg, #3b82f6, #06b6d4)',
            transition: 'width 1s linear',
          }}
        />
      </div>
    </div>
  )
}

/* ── Main export ───────────────────────────────────────────────────────── */
interface LoadingScreenProps {
  message:     string
  stageNum:    number
  totalStages: number
  percent:     number
  videoName:   string
}

export default function LoadingScreen({
  message, stageNum, totalStages, percent, videoName,
}: LoadingScreenProps) {
  const [fact,      setFact]      = useState<string | null>(null)
  const [countdown, setCountdown] = useState(15)
  const [showGame,  setShowGame]  = useState(false)

  const loadFact = useCallback(async () => {
    const f = await fetchFunFact()
    setFact(f)
    setCountdown(15)
  }, [])

  /* Load first fact */
  useEffect(() => { loadFact() }, [loadFact])

  /* Rotate every 15 s */
  useEffect(() => {
    const id = setInterval(() => {
      setCountdown(c => {
        if (c <= 1) { loadFact(); return 15 }
        return c - 1
      })
    }, 1000)
    return () => clearInterval(id)
  }, [loadFact])

  const stage = STAGE_META[stageNum] ?? { icon: '⚙️', label: message }

  return (
    <div
      className="absolute inset-0 flex flex-col items-center justify-center gap-5 px-6 py-8 overflow-y-auto"
      style={{
        background: 'radial-gradient(ellipse at 50% 30%, #1e3a5f 0%, #0f172a 70%)',
        zIndex: 10,
      }}
    >
      {/* Grid dot background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: 'radial-gradient(circle, rgba(59,130,246,0.08) 1px, transparent 1px)',
          backgroundSize: '28px 28px',
        }}
      />

      {/* Hourglass graphic */}
      <div className="relative flex items-center justify-center" style={{ zIndex: 1 }}>
        <div className="scale-75">
          <HourglassSVG pct={percent / 100} />
        </div>
      </div>

      {/* Title */}
      <div className="text-center -mt-2" style={{ zIndex: 1 }}>
        <div className="shimmer-text text-lg font-bold mb-0.5">Analyzing your video…</div>
        {videoName && (
          <div className="text-xs text-slate-500 truncate max-w-xs">{videoName}</div>
        )}
      </div>

      {/* Stage + progress */}
      <div className="w-full max-w-sm" style={{ zIndex: 1 }}>
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-sm">{stage.icon}</span>
          <span className="text-sm text-slate-200 font-medium">{stage.label}</span>
          <span className="ml-auto text-xs font-mono text-blue-400 font-bold">
            {stageNum}/{totalStages}
          </span>
        </div>
        <div className="h-2 rounded-full overflow-hidden" style={{ background: '#1e293b' }}>
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${percent}%`,
              background: 'linear-gradient(90deg, #1d4ed8, #3b82f6, #06b6d4)',
              boxShadow: '0 0 12px rgba(59,130,246,0.5)',
            }}
          />
        </div>
        <div className="flex justify-between text-[11px] mt-1">
          <span className="text-slate-500 truncate max-w-[70%]">{message}</span>
          <span className="font-mono text-cyan-400 font-bold">{percent}%</span>
        </div>
      </div>

      {/* Stage pills */}
      <div className="grid grid-cols-3 gap-2 w-full max-w-sm" style={{ zIndex: 1 }}>
        {Object.entries(STAGE_META).map(([n, s]) => {
          const num  = parseInt(n)
          const done = num < stageNum
          const curr = num === stageNum
          return (
            <div
              key={n}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg transition-all ${
                curr  ? 'bg-blue-900/60 text-blue-300 border border-blue-500/50 shadow-md shadow-blue-900/30' :
                done  ? 'bg-green-900/30 text-green-400 border border-green-800/30' :
                        'bg-slate-800/30 text-slate-600 border border-slate-700/30'
              }`}
            >
              <span className="text-xs flex-shrink-0">{done ? '✓' : s.icon}</span>
              <span className="text-[11px] leading-tight">{s.short}</span>
            </div>
          )
        })}
      </div>

      {/* Toggle: fact vs game */}
      <div className="flex items-center gap-1.5" style={{ zIndex: 1 }}>
        <button
          onClick={() => setShowGame(false)}
          className={`px-2.5 py-1 rounded-full text-[11px] font-medium transition-colors ${!showGame ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
        >
          💡 Fun Fact
        </button>
        <button
          onClick={() => setShowGame(true)}
          className={`px-2.5 py-1 rounded-full text-[11px] font-medium transition-colors ${showGame ? 'bg-green-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
        >
          🎮 Mini Game
        </button>
      </div>

      {/* Fun fact card or mini game */}
      <div className="w-full max-w-sm" style={{ zIndex: 1 }}>
        {showGame ? (
          <MiniGame />
        ) : fact ? (
          <FactCard fact={fact} countdown={countdown} />
        ) : (
          <div
            className="w-full max-w-md rounded-xl p-4"
            style={{ background: 'rgba(30,41,59,0.5)', border: '1px solid rgba(51,65,85,0.5)' }}
          >
            <div className="h-3 w-24 rounded bg-slate-700 mb-3 animate-pulse" />
            <div className="h-3 w-full rounded bg-slate-700 mb-2 animate-pulse" />
            <div className="h-3 w-4/5 rounded bg-slate-700 animate-pulse" />
          </div>
        )}
      </div>

      {/* Tip */}
      <p className="text-[10px] text-slate-600 text-center max-w-sm" style={{ zIndex: 1 }}>
        Keep this window open. Long videos may take 5–15 minutes.
      </p>
    </div>
  )
}
