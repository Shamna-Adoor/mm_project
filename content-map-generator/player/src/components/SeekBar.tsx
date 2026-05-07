import { useRef, useState, useEffect } from 'react'
import type { Segment, Chapter } from '../types'
import { LABEL_COLORS, LABEL_NAMES, formatTime } from '../types'

interface SeekBarProps {
  currentTime: number
  duration: number
  segments?: Segment[]
  chapters?: Chapter[]
  onSeek: (time: number) => void
}

export default function SeekBar({
  currentTime,
  duration,
  segments = [],
  chapters = [],
  onSeek,
}: SeekBarProps) {
  const trackRef    = useRef<HTMLDivElement>(null)
  const isDragging  = useRef(false)
  const durationRef = useRef(duration)
  const onSeekRef   = useRef(onSeek)
  useEffect(() => { durationRef.current = duration }, [duration])
  useEffect(() => { onSeekRef.current   = onSeek   }, [onSeek])

  const [dragging,  setDragging]  = useState(false)
  const [hoverX,    setHoverX]    = useState<number | null>(null)
  const [hoverSeg,  setHoverSeg]  = useState<Segment | null>(null)
  const [hoverChap, setHoverChap] = useState<Chapter | null>(null)

  const timeFromX = (clientX: number): number => {
    if (!trackRef.current) return 0
    const r = trackRef.current.getBoundingClientRect()
    return Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * durationRef.current
  }

  useEffect(() => {
    const onMove = (e: MouseEvent) => { if (isDragging.current) onSeekRef.current(timeFromX(e.clientX)) }
    const onUp   = () => { if (isDragging.current) { isDragging.current = false; setDragging(false) } }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup',   onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [])

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault(); isDragging.current = true; setDragging(true); onSeek(timeFromX(e.clientX))
  }
  const handleMouseMove = (e: React.MouseEvent) => {
    const t = timeFromX(e.clientX)
    setHoverX(e.clientX)
    setHoverSeg(segments.find(s => t >= s.start && t < s.end) ?? null)
    setHoverChap(chapters.slice().reverse().find(c => t >= c.start) ?? null)
  }

  const pct = (t: number) => duration > 0 ? `${(t / duration) * 100}%` : '0%'
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0

  const hoverPct =
    hoverX !== null && trackRef.current
      ? Math.max(1, Math.min(99, ((hoverX - trackRef.current.getBoundingClientRect().left) / trackRef.current.getBoundingClientRect().width) * 100))
      : null

  return (
    <div className="px-4 pt-2 pb-1 select-none" style={{ background: '#0f172a' }}>

      {/* Chapter tick marks */}
      {chapters.length > 0 && (
        <div className="relative mb-1" style={{ height: 12 }}>
          {chapters.map((ch, i) => (
            <button
              key={i}
              onClick={() => onSeek(ch.start)}
              className="absolute top-0 h-full flex flex-col items-center group"
              style={{ left: pct(ch.start), transform: 'translateX(-50%)' }}
              title={ch.title}
            >
              <div className="w-px bg-slate-500 group-hover:bg-white h-2" />
              <div className="w-1.5 h-1.5 rounded-full bg-slate-500 group-hover:bg-white -mt-px" />
            </button>
          ))}
        </div>
      )}

      {/* Segment color strip */}
      <div className="relative rounded overflow-hidden mb-1.5" style={{ height: 4 }}>
        {segments.map((seg, i) => (
          <div
            key={i}
            className="absolute top-0 h-full"
            style={{
              left: pct(seg.start),
              width: pct(seg.end - seg.start),
              backgroundColor: LABEL_COLORS[seg.label].bar,
              opacity: 0.7,
            }}
          />
        ))}
        {/* Played overlay */}
        <div
          className="absolute top-0 left-0 h-full pointer-events-none"
          style={{ width: `${progress}%`, background: 'rgba(255,255,255,0.25)' }}
        />
      </div>

      {/* Main seek track */}
      <div
        ref={trackRef}
        className="relative rounded-full cursor-pointer"
        style={{ height: 6, background: '#334155' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => { setHoverX(null); setHoverSeg(null); setHoverChap(null) }}
      >
        {/* Progress fill */}
        <div
          className="absolute left-0 top-0 h-full rounded-l-full pointer-events-none"
          style={{ width: `${progress}%`, background: '#3b82f6' }}
        />

        {/* Skip-segment tints over progress track */}
        {segments.filter(s => s.skip_recommended).map((seg, i) => (
          <div
            key={i}
            className="absolute top-0 h-full pointer-events-none"
            style={{
              left: pct(seg.start),
              width: pct(seg.end - seg.start),
              backgroundColor: LABEL_COLORS[seg.label].bar,
              opacity: 0.35,
              mixBlendMode: 'screen',
            }}
          />
        ))}

        {/* Thumb */}
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full bg-white shadow-lg pointer-events-none transition-all"
          style={{
            left: `${progress}%`,
            width:  dragging ? 18 : 14,
            height: dragging ? 18 : 14,
            boxShadow: '0 0 6px rgba(59,130,246,0.7)',
          }}
        />

        {/* Hover tooltip */}
        {hoverPct !== null && duration > 0 && (
          <div
            className="absolute -translate-x-1/2 pointer-events-none whitespace-nowrap rounded-md shadow-xl z-20"
            style={{
              left: `${hoverPct}%`,
              bottom: 'calc(100% + 10px)',
              background: '#1e293b',
              border: '1px solid #334155',
              padding: '4px 8px',
              fontSize: 11,
            }}
          >
            <span className="text-white font-mono">{formatTime(timeFromX(hoverX!))}</span>
            {hoverSeg && (
              <span style={{ color: LABEL_COLORS[hoverSeg.label].bar, marginLeft: 6 }}>
                {LABEL_NAMES[hoverSeg.label]}
              </span>
            )}
            {hoverChap && (
              <div className="text-slate-400 text-xs mt-0.5 truncate max-w-36">{hoverChap.title}</div>
            )}
          </div>
        )}
      </div>

      {/* Time labels */}
      <div className="flex justify-between text-xs mt-1" style={{ color: '#64748b' }}>
        <span className="font-mono">{formatTime(currentTime)}</span>
        <span className="font-mono">{formatTime(duration)}</span>
      </div>
    </div>
  )
}
