import type { SegmentLabel } from '../types'
import { LABEL_NAMES, SKIP_LABELS, formatTime } from '../types'

const ALL_LABELS: SegmentLabel[] = ['intro', 'main_content', 'sponsor', 'outro', 'dead_air']

interface LabelingControlsProps {
  isPlaying: boolean
  currentTime: number
  pendingStart: number | null
  pendingEnd: number | null
  selectedLabel: SegmentLabel
  canAdd: boolean
  disabled: boolean
  onTogglePlay: () => void
  onMarkStart: () => void
  onMarkEnd: () => void
  onLabelChange: (label: SegmentLabel) => void
  onAddSegment: () => void
  onClearPending: () => void
}

export default function LabelingControls({
  isPlaying,
  currentTime,
  pendingStart,
  pendingEnd,
  selectedLabel,
  canAdd,
  disabled,
  onTogglePlay,
  onMarkStart,
  onMarkEnd,
  onLabelChange,
  onAddSegment,
  onClearPending,
}: LabelingControlsProps) {
  return (
    <div className="flex flex-col gap-0 border-t border-gray-700 bg-gray-800">
      {/* Playback row */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-700/60">
        <button
          onClick={onTogglePlay}
          disabled={disabled}
          className="px-4 py-1.5 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-sm font-medium min-w-[80px]"
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        <span className="text-sm text-blue-300 tabular-nums">{formatTime(currentTime)}</span>
        <span className="text-xs text-gray-500 ml-auto">Space to play/pause</span>
      </div>

      {/* Labeling row */}
      <div className="flex items-center gap-2 px-4 py-2 flex-wrap">
        {/* Mark Start */}
        <button
          onClick={onMarkStart}
          disabled={disabled}
          className="px-3 py-1.5 rounded bg-yellow-600 hover:bg-yellow-500 disabled:opacity-40 text-sm"
          title="Mark start (I)"
        >
          Mark Start
          {pendingStart !== null && (
            <span className="ml-1.5 text-yellow-200 text-xs">{formatTime(pendingStart)}</span>
          )}
        </button>

        {/* Mark End */}
        <button
          onClick={onMarkEnd}
          disabled={disabled || pendingStart === null}
          className="px-3 py-1.5 rounded bg-yellow-600 hover:bg-yellow-500 disabled:opacity-40 text-sm"
          title="Mark end (O)"
        >
          Mark End
          {pendingEnd !== null && (
            <span className="ml-1.5 text-yellow-200 text-xs">{formatTime(pendingEnd)}</span>
          )}
        </button>

        {/* Label dropdown */}
        <select
          value={selectedLabel}
          onChange={(e) => onLabelChange(e.target.value as SegmentLabel)}
          className="px-2 py-1.5 rounded bg-gray-700 border border-gray-600 text-sm text-white"
        >
          {ALL_LABELS.map(l => (
            <option key={l} value={l}>{LABEL_NAMES[l]}</option>
          ))}
        </select>

        {/* Add Segment */}
        <button
          onClick={onAddSegment}
          disabled={!canAdd}
          className="px-3 py-1.5 rounded bg-green-700 hover:bg-green-600 disabled:opacity-40 disabled:cursor-not-allowed text-sm font-medium"
        >
          + Add Segment
        </button>

        {/* Clear pending */}
        {(pendingStart !== null || pendingEnd !== null) && (
          <button
            onClick={onClearPending}
            className="px-2 py-1.5 rounded text-gray-500 hover:text-gray-300 text-sm"
          >
            Clear
          </button>
        )}

        <span className="text-xs text-gray-600 ml-auto hidden sm:block">I · O to mark</span>
      </div>
    </div>
  )
}
