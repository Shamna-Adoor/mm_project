import { useRef } from 'react'
import type { Segment } from '../types'
import { LABEL_COLORS, LABEL_NAMES } from '../types'

interface ControlsProps {
  isPlaying:       boolean
  contentOnlyMode: boolean
  currentSegment:  Segment | null
  hasSegments:     boolean
  disabled:        boolean
  volume:          number
  muted:           boolean
  playbackRate:    number
  currentTime:     number
  duration:        number
  onTogglePlay:         () => void
  onSeekRelative:       (delta: number) => void
  onSkipNonContent:     () => void
  onToggleContentOnly:  () => void
  onVolumeChange:       (v: number) => void
  onToggleMute:         () => void
  onPlaybackRateChange: (r: number) => void
  onFullscreen:         () => void
}

const RATES = [0.5, 0.75, 1, 1.25, 1.5, 2]

function fmt(s: number) {
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const ss = Math.floor(s % 60)
  if (h > 0) return `${h}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`
  return `${m}:${String(ss).padStart(2,'0')}`
}

export default function Controls({
  isPlaying, contentOnlyMode, currentSegment, hasSegments, disabled,
  volume, muted, playbackRate, currentTime, duration,
  onTogglePlay, onSeekRelative, onSkipNonContent, onToggleContentOnly,
  onVolumeChange, onToggleMute, onPlaybackRateChange, onFullscreen,
}: ControlsProps) {
  const rateRef = useRef<HTMLSelectElement>(null)

  const canSkip = hasSegments && !disabled && currentSegment?.skip_recommended === true
  const volIcon = muted || volume === 0 ? '🔇' : volume < 0.5 ? '🔉' : '🔊'

  return (
    <div
      className="flex items-center gap-1 px-3 py-2 flex-wrap"
      style={{ background: '#0f172a', borderTop: '1px solid #1e293b' }}
    >
      {/* ── Left: playback controls ── */}
      <div className="flex items-center gap-1">
        {/* Seek back */}
        <button
          onClick={() => onSeekRelative(-10)}
          disabled={disabled}
          title="Back 10s"
          className="w-8 h-8 flex items-center justify-center rounded hover:bg-slate-700 disabled:opacity-30 text-slate-300 text-sm transition-colors"
        >
          ⏪
        </button>

        {/* Play/Pause */}
        <button
          onClick={onTogglePlay}
          disabled={disabled}
          className="w-9 h-9 flex items-center justify-center rounded-full bg-blue-600 hover:bg-blue-500 disabled:opacity-30 text-white text-base transition-colors shadow"
        >
          {isPlaying ? '⏸' : '▶'}
        </button>

        {/* Seek forward */}
        <button
          onClick={() => onSeekRelative(10)}
          disabled={disabled}
          title="Forward 10s"
          className="w-8 h-8 flex items-center justify-center rounded hover:bg-slate-700 disabled:opacity-30 text-slate-300 text-sm transition-colors"
        >
          ⏩
        </button>
      </div>

      {/* Time */}
      <div className="text-xs font-mono text-slate-400 px-2 hidden sm:block">
        <span className="text-white">{fmt(currentTime)}</span>
        <span className="text-slate-600 mx-1">/</span>
        <span>{fmt(duration)}</span>
      </div>

      {/* ── Center: Content-Only mode (hero button) ── */}
      <div className="flex items-center gap-2 mx-auto">
        <button
          onClick={onToggleContentOnly}
          disabled={disabled || !hasSegments}
          className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-semibold transition-all shadow disabled:opacity-30 disabled:cursor-not-allowed ${
            contentOnlyMode
              ? 'bg-green-500 hover:bg-green-400 text-white ring-2 ring-green-400/50'
              : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
          }`}
        >
          <span className={`w-2 h-2 rounded-full ${contentOnlyMode ? 'bg-white animate-pulse' : 'bg-slate-500'}`} />
          {contentOnlyMode ? 'Content-Only: ON' : 'Content-Only'}
        </button>

        {/* Skip button (only enabled when in a skip segment) */}
        {canSkip && (
          <button
            onClick={onSkipNonContent}
            className="flex items-center gap-1 px-3 py-1.5 rounded-full text-sm font-medium bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-300 border border-yellow-500/40 transition-colors"
            title="Skip non-content (S)"
          >
            Skip <span className="font-mono text-xs opacity-70">[S]</span>
          </button>
        )}
      </div>

      {/* ── Right: volume + speed + fullscreen ── */}
      <div className="flex items-center gap-2 ml-auto">
        {/* Current segment badge */}
        {hasSegments && currentSegment && (
          <span
            className="text-xs px-2 py-0.5 rounded-full font-medium hidden md:block"
            style={{
              background: LABEL_COLORS[currentSegment.label].bar + '22',
              color:      LABEL_COLORS[currentSegment.label].bar,
              border:     `1px solid ${LABEL_COLORS[currentSegment.label].bar}44`,
            }}
          >
            {LABEL_NAMES[currentSegment.label]}
          </span>
        )}

        {/* Mute + Volume */}
        <div className="flex items-center gap-1">
          <button
            onClick={onToggleMute}
            disabled={disabled}
            className="text-base w-7 h-7 flex items-center justify-center hover:bg-slate-700 rounded disabled:opacity-30 transition-colors"
          >
            {volIcon}
          </button>
          <input
            type="range" min={0} max={1} step={0.02}
            value={muted ? 0 : volume}
            onChange={e => onVolumeChange(parseFloat(e.target.value))}
            disabled={disabled}
            className="w-16 h-1 accent-blue-500 cursor-pointer disabled:opacity-30"
          />
        </div>

        {/* Speed */}
        <select
          ref={rateRef}
          value={playbackRate}
          onChange={e => onPlaybackRateChange(parseFloat(e.target.value))}
          disabled={disabled}
          className="text-xs bg-slate-700 border border-slate-600 text-slate-200 rounded px-1.5 py-0.5 cursor-pointer disabled:opacity-30 hover:bg-slate-600 transition-colors"
        >
          {RATES.map(r => (
            <option key={r} value={r}>{r}×</option>
          ))}
        </select>

        {/* Fullscreen */}
        <button
          onClick={onFullscreen}
          disabled={disabled}
          title="Fullscreen (F)"
          className="w-7 h-7 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-700 rounded disabled:opacity-30 transition-colors text-sm"
        >
          ⛶
        </button>
      </div>
    </div>
  )
}
