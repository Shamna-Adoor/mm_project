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

export interface SegmentMap {
  video_id: string
  duration_seconds: number
  generated_at: string
  segments: Segment[]
}

export const LABEL_COLORS: Record<SegmentLabel, { bar: string; dot: string; text: string }> = {
  main_content: { bar: '#22c55e', dot: 'bg-green-500',  text: 'text-green-400' },
  sponsor:      { bar: '#ef4444', dot: 'bg-red-500',    text: 'text-red-400'   },
  intro:        { bar: '#facc15', dot: 'bg-yellow-400', text: 'text-yellow-400'},
  outro:        { bar: '#eab308', dot: 'bg-yellow-500', text: 'text-yellow-500'},
  dead_air:     { bar: '#6b7280', dot: 'bg-gray-500',   text: 'text-gray-400'  },
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
