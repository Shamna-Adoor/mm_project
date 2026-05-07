import type { Segment } from '../types'
import { LABEL_COLORS, LABEL_NAMES, formatTime } from '../types'

interface SegmentListProps {
  segments: Segment[]
  currentTime: number
  onSeek: (time: number) => void
}

export default function SegmentList({ segments, currentTime, onSeek }: SegmentListProps) {
  return (
    <div className="flex-1 overflow-y-auto">
      {segments.map((seg, index) => {
        const isCurrent = currentTime >= seg.start && currentTime < seg.end
        const colors = LABEL_COLORS[seg.label]
        const duration = seg.end - seg.start

        return (
          <button
            key={index}
            type="button"
            onClick={() => onSeek(seg.start)}
            className={`w-full text-left px-4 py-3 border-b border-slate-800 transition-colors ${
              isCurrent ? 'bg-blue-900/50' : 'hover:bg-slate-800'
            }`}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-2 h-9 rounded-full flex-shrink-0"
                style={{ background: colors.bar }}
              />

              <div className="flex-1 min-w-0">
                <div className="flex justify-between gap-2">
                  <span className="font-semibold" style={{ color: colors.bar }}>
                    {LABEL_NAMES[seg.label]}
                  </span>

                  <span className="font-mono text-slate-400">
                    {formatTime(duration)}
                  </span>
                </div>

                <div className="font-mono text-slate-400 mt-1">
                  {formatTime(seg.start)} → {formatTime(seg.end)}
                </div>

                <div className="text-slate-500 truncate mt-1">
                  {seg.reason}
                </div>
              </div>
            </div>
          </button>
        )
      })}
    </div>
  )
}