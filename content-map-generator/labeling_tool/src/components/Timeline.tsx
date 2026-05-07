import { useRef, useState } from 'react'
import type { Segment } from '../types'
import { LABEL_COLORS, LABEL_NAMES, formatTime } from '../types'

interface Tooltip {
  segment: Segment
  x: number
  y: number
}

interface TimelineProps {
  segments: Segment[]
  duration: number
  currentTime: number
  pendingStart: number | null
  pendingEnd: number | null
  onSeek: (time: number) => void
}

export default function Timeline({
  segments,
  duration,
  currentTime,
  pendingStart,
  pendingEnd,
  onSeek,
}: TimelineProps) {
  const barRef = useRef<HTMLDivElement>(null)
  const [tooltip, setTooltip] = useState<Tooltip | null>(null)

  if (duration <= 0) return null

  const pct = (t: number) => `${(t / duration) * 100}%`

  const handleBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    onSeek(((e.clientX - rect.left) / rect.width) * duration)
  }

  // Pending region
  const pendingRegion =
    pendingStart !== null && pendingEnd !== null
      ? { start: Math.min(pendingStart, pendingEnd), end: Math.max(pendingStart, pendingEnd) }
      : pendingStart !== null
      ? { start: pendingStart, end: currentTime }
      : null

  return (
    <div className="px-4 py-3 border-t border-gray-700 select-none" style={{ background: '#111827' }}>
      <div
        ref={barRef}
        className="relative h-8 rounded overflow-hidden cursor-pointer bg-gray-700"
        onClick={handleBarClick}
      >
        {segments.map((seg, i) => (
          <div
            key={i}
            className="absolute top-0 h-full opacity-70 hover:opacity-90 transition-opacity"
            style={{
              left: pct(seg.start),
              width: pct(seg.end - seg.start),
              backgroundColor: LABEL_COLORS[seg.label].bar,
            }}
            onMouseMove={(e) => {
              e.stopPropagation()
              setTooltip({ segment: seg, x: e.clientX, y: e.clientY })
            }}
            onMouseLeave={() => setTooltip(null)}
          />
        ))}

        {/* Pending region overlay */}
        {pendingRegion && pendingRegion.end > pendingRegion.start && (
          <div
            className="absolute top-0 h-full pointer-events-none"
            style={{
              left: pct(pendingRegion.start),
              width: pct(pendingRegion.end - pendingRegion.start),
              backgroundColor: 'rgba(255,255,255,0.25)',
              border: '1px dashed rgba(255,255,255,0.6)',
            }}
          />
        )}

        {/* Pending start marker */}
        {pendingStart !== null && (
          <div
            className="absolute top-0 h-full w-0.5 bg-white z-20 pointer-events-none"
            style={{ left: pct(pendingStart) }}
          />
        )}

        {/* Playhead */}
        <div
          className="absolute top-0 h-full w-0.5 bg-blue-400 z-10 pointer-events-none"
          style={{ left: pct(currentTime) }}
        />
      </div>

      <div className="flex justify-between text-xs text-gray-500 mt-1 px-0.5">
        <span>{formatTime(0)}</span>
        <span className="text-blue-400">{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      {tooltip && (
        <div
          className="fixed z-50 pointer-events-none bg-gray-900 border border-gray-600 rounded p-2 text-xs shadow-lg"
          style={{ left: tooltip.x + 12, top: tooltip.y - 60 }}
        >
          <div className="font-semibold" style={{ color: LABEL_COLORS[tooltip.segment.label].bar }}>
            {LABEL_NAMES[tooltip.segment.label]}
          </div>
          <div className="text-gray-300">
            {formatTime(tooltip.segment.start)} – {formatTime(tooltip.segment.end)}
          </div>
        </div>
      )}
    </div>
  )
}
