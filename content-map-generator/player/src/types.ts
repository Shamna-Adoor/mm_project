export type SegmentLabel = 'intro' | 'main_content' | 'sponsor' | 'outro' | 'dead_air'

export interface Segment {
  start: number
  end: number
  label: SegmentLabel
  confidence: number
  skip_recommended: boolean
  reason: string
  signals_used: string[]
}

export interface Chapter {
  start: number
  end:   number
  title: string
}

export interface SegmentMap {
  video_id: string
  duration_seconds: number
  generated_at: string
  segments: Segment[]
  chapters?: Chapter[]
}

export const LABEL_COLORS: Record<SegmentLabel, { bar: string; badge: string; text: string; bg: string }> = {
  main_content: { bar: '#22c55e', badge: 'bg-green-600',   text: 'text-green-400',  bg: 'bg-green-900/30' },
  sponsor:      { bar: '#ef4444', badge: 'bg-red-600',     text: 'text-red-400',    bg: 'bg-red-900/30'   },
  intro:        { bar: '#eab308', badge: 'bg-yellow-500',  text: 'text-yellow-400', bg: 'bg-yellow-900/30'},
  outro:        { bar: '#f59e0b', badge: 'bg-amber-500',   text: 'text-amber-400',  bg: 'bg-amber-900/30' },
  dead_air:     { bar: '#6b7280', badge: 'bg-gray-600',    text: 'text-gray-400',   bg: 'bg-gray-800/50'  },
}

export const LABEL_NAMES: Record<SegmentLabel, string> = {
  main_content: 'Main Content',
  sponsor:      'Sponsor',
  intro:        'Intro',
  outro:        'Outro',
  dead_air:     'Dead Air',
}

export const SKIP_LABELS = new Set<SegmentLabel>(['intro', 'sponsor', 'outro', 'dead_air'])

export function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${m}:${String(s).padStart(2, '0')}`
}

export function segmentStats(segments: Segment[], duration: number) {
  const skipSecs    = segments.filter(s => s.skip_recommended).reduce((a, s) => a + (s.end - s.start), 0)
  const contentSecs = Math.max(0, duration - skipSecs)
  const savedPct    = duration > 0 ? Math.round((skipSecs / duration) * 100) : 0
  return { skipSecs, contentSecs, savedPct }
}
