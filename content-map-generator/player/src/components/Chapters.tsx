import { useEffect, useRef } from 'react'
import type { Chapter, Segment } from '../types'
import { LABEL_COLORS, formatTime } from '../types'

interface ChaptersProps {
  chapters:    Chapter[]
  segments:    Segment[]
  currentTime: number
  onSeek:      (t: number) => void
}

function chapterIsSkip(ch: Chapter, segments: Segment[]): boolean {
  const dur = ch.end - ch.start
  if (dur <= 0) return false
  const skipSecs = segments
    .filter(s => s.skip_recommended)
    .reduce((acc, s) => acc + Math.max(0, Math.min(s.end, ch.end) - Math.max(s.start, ch.start)), 0)
  return skipSecs / dur > 0.5
}

/** Dominant non-content label in a chapter (for color coding). */
function chapterLabel(ch: Chapter, segments: Segment[]): string | null {
  const counts: Record<string, number> = {}
  for (const s of segments) {
    if (!s.skip_recommended) continue
    const overlap = Math.max(0, Math.min(s.end, ch.end) - Math.max(s.start, ch.start))
    if (overlap > 0) counts[s.label] = (counts[s.label] ?? 0) + overlap
  }
  if (!Object.keys(counts).length) return null
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]
}

export default function Chapters({ chapters, segments, currentTime, onSeek }: ChaptersProps) {
  const activeIdx = chapters.reduce<number>((best, ch, i) => currentTime >= ch.start ? i : best, 0)
  const listRef   = useRef<HTMLDivElement>(null)
  const itemRefs  = useRef<(HTMLButtonElement | null)[]>([])

  useEffect(() => {
    const el = itemRefs.current[activeIdx]
    if (el && listRef.current) {
      el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [activeIdx])

  if (!chapters.length) return null

  return (
    <div ref={listRef} className="flex-1 overflow-y-auto">
      {chapters.map((ch, i) => {
        const isActive = i === activeIdx
        const skip     = chapterIsSkip(ch, segments)
        const domLabel = chapterLabel(ch, segments)
        const accentColor = domLabel && skip
          ? LABEL_COLORS[domLabel as keyof typeof LABEL_COLORS]?.bar ?? '#6b7280'
          : '#22c55e'

        return (
          <button
            key={i}
            ref={el => { itemRefs.current[i] = el }}
            onClick={() => onSeek(ch.start)}
            className={`w-full text-left px-3 py-2.5 flex items-start gap-2.5 transition-colors border-b ${
              isActive
                ? 'bg-blue-900/40 border-blue-800/50'
                : 'hover:bg-slate-800/70 border-slate-800/50'
            }`}
          >
            {/* Color dot */}
            <div
              className="w-2 h-2 rounded-full flex-shrink-0 mt-1"
              style={{ background: isActive ? '#3b82f6' : accentColor }}
            />
            <div className="flex-1 min-w-0">
              <div className={`text-xs font-mono mb-0.5 ${isActive ? 'text-blue-400' : 'text-slate-500'}`}>
                {formatTime(ch.start)}
              </div>
              <div className={`text-xs leading-snug font-medium truncate ${isActive ? 'text-white' : 'text-slate-300'}`}>
                {ch.title}
              </div>
              {skip && domLabel && (
                <div
                  className="text-xs mt-0.5 font-medium"
                  style={{ color: accentColor }}
                >
                  {domLabel === 'sponsor' ? '· Ad' : domLabel === 'intro' ? '· Intro' : domLabel === 'outro' ? '· Outro' : '· Skip'}
                </div>
              )}
            </div>
          </button>
        )
      })}
    </div>
  )
}
