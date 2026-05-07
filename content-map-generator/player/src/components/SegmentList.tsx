import { useState, useEffect, useRef } from 'react'
import type { Segment } from '../types'
import { LABEL_COLORS, LABEL_NAMES, formatTime } from '../types'

interface SegmentListProps {
  segments:    Segment[]
  currentTime: number
  onSeek:      (time: number) => void
}

type Filter = 'all' | 'content' | 'skip'

export default function SegmentList({ segments, currentTime, onSeek }: SegmentListProps) {
  const [filter, setFilter] = useState<Filter>('all')
  const activeIdx = segments.findIndex(s => currentTime >= s.start && currentTime < s.end)
  const listRef   = useRef<HTMLDivElement>(null)
  const itemRefs  = useRef<(HTMLButtonElement | null)[]>([])

  useEffect(() => {
    const el = itemRefs.current[activeIdx]
    if (el && listRef.current) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [activeIdx])

  const skipCount    = segments.filter(s => s.skip_recommended).length
  const contentCount = segments.length - skipCount

  const filtered = segments.filter(s =>
    filter === 'all'     ? true :
    filter === 'content' ? !s.skip_recommended :
    s.skip_recommended
  )

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Filter chips */}
      <div className="px-3 pt-2 pb-1.5 flex items-center gap-1.5 border-b border-slate-800 flex-shrink-0">
        {(['all', 'content', 'skip'] as Filter[]).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors ${
              filter === f
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
            }`}
          >
            {f === 'all' ? `All ${segments.length}` : f === 'content' ? `Content ${contentCount}` : `Skip ${skipCount}`}
          </button>
        ))}
      </div>

      {/* List */}
      <div ref={listRef} className="flex-1 overflow-y-auto">
        {filtered.map((seg, i) => {
          const globalIdx = segments.indexOf(seg)
          const isCurrent = globalIdx === activeIdx
          const colors    = LABEL_COLORS[seg.label]
          const dur       = seg.end - seg.start

          return (
            <button
              key={globalIdx}
              ref={el => { itemRefs.current[globalIdx] = el }}
              onClick={() => onSeek(seg.start)}
              className={`w-full text-left px-3 py-2 border-b border-slate-800/50 transition-colors ${
                isCurrent ? 'bg-slate-700/60' : 'hover:bg-slate-800/50'
              }`}
            >
              <div className="flex items-center gap-2">
                <div
                  className="w-1.5 h-5 rounded-full flex-shrink-0"
                  style={{ background: colors.bar, opacity: isCurrent ? 1 : 0.7 }}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-xs font-semibold" style={{ color: colors.bar }}>
                      {LABEL_NAMES[seg.label]}
                    </span>
                    <span className="text-xs font-mono text-slate-500 flex-shrink-0">
                      {formatTime(dur)}
                    </span>
                  </div>
                  <div className="text-xs text-slate-500 font-mono mt-0.5">
                    {formatTime(seg.start)} → {formatTime(seg.end)}
                  </div>
                  {seg.reason && (
                    <div className="text-xs text-slate-600 mt-0.5 truncate" title={seg.reason}>
                      {seg.reason}
                    </div>
                  )}
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
