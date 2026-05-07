import type { Segment } from '../types'
import { LABEL_COLORS, LABEL_NAMES, formatTime } from '../types'

interface SegmentListProps {
  segments: Segment[]
  onDelete: (index: number) => void
  onSeek: (time: number) => void
}

export default function SegmentList({ segments, onDelete, onSeek }: SegmentListProps) {
  const sorted = [...segments].map((s, origIndex) => ({ s, origIndex }))
    .sort((a, b) => a.s.start - b.s.start)

  return (
    <aside className="w-72 flex-shrink-0 bg-gray-800 border-l border-gray-700 flex flex-col">
      <div className="px-4 py-3 border-b border-gray-700 text-sm font-semibold text-gray-300">
        Labeled Segments ({segments.length})
      </div>
      {segments.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-xs text-gray-600 text-center px-4">
          No segments yet.<br />Mark start + end, then add a segment.
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          {sorted.map(({ s, origIndex }) => {
            const colors = LABEL_COLORS[s.label]
            return (
              <div
                key={origIndex}
                className="flex items-start px-4 py-2.5 border-b border-gray-700/50 hover:bg-gray-700/50 group"
              >
                <button
                  className="flex-1 text-left min-w-0"
                  onClick={() => onSeek(s.start)}
                >
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full flex-shrink-0 ${colors.dot}`} />
                    <span className={`text-xs font-medium ${colors.text}`}>
                      {LABEL_NAMES[s.label]}
                    </span>
                  </div>
                  <div className="text-xs text-gray-400 mt-0.5 pl-4">
                    {formatTime(s.start)} – {formatTime(s.end)}
                  </div>
                </button>
                <button
                  onClick={() => onDelete(origIndex)}
                  className="ml-2 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity text-sm flex-shrink-0"
                  title="Delete segment"
                >
                  ✕
                </button>
              </div>
            )
          })}
        </div>
      )}
    </aside>
  )
}
