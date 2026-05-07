import { useState, useRef, useEffect, useCallback } from 'react'
import type { Segment, SegmentLabel } from './types'
import { SKIP_LABELS } from './types'
import Timeline from './components/Timeline'
import SegmentList from './components/SegmentList'
import LabelingControls from './components/LabelingControls'

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)

  const [videoSrc, setVideoSrc] = useState<string | null>(null)
  const [videoFilename, setVideoFilename] = useState('')
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const [segments, setSegments] = useState<Segment[]>([])
  const [pendingStart, setPendingStart] = useState<number | null>(null)
  const [pendingEnd, setPendingEnd] = useState<number | null>(null)
  const [selectedLabel, setSelectedLabel] = useState<SegmentLabel>('main_content')

  const canAdd = pendingStart !== null && pendingEnd !== null && pendingEnd !== pendingStart

  const videoId = videoFilename
    ? videoFilename.replace(/\.[^.]+$/, '').replace(/\s+/g, '_').toLowerCase()
    : 'unlabeled'

  // ── File loader ───────────────────────────────────────────────────────────

  const handleVideoLoad = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (videoSrc) URL.revokeObjectURL(videoSrc)
    setVideoSrc(URL.createObjectURL(file))
    setVideoFilename(file.name)
    setSegments([])
    setPendingStart(null)
    setPendingEnd(null)
    setCurrentTime(0)
    e.target.value = ''
  }

  // ── Video events ──────────────────────────────────────────────────────────

  const handleTimeUpdate = () => {
    if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
  }

  const handleLoadedMetadata = () => {
    if (videoRef.current) setDuration(videoRef.current.duration)
  }

  // ── Playback ──────────────────────────────────────────────────────────────

  const togglePlay = useCallback(() => {
    if (!videoRef.current) return
    videoRef.current.paused ? videoRef.current.play() : videoRef.current.pause()
  }, [])

  const handleSeek = useCallback((time: number) => {
    if (videoRef.current) videoRef.current.currentTime = time
  }, [])

  // ── Labeling ──────────────────────────────────────────────────────────────

  const markStart = useCallback(() => {
    if (!videoRef.current) return
    setPendingStart(videoRef.current.currentTime)
    setPendingEnd(null)
  }, [])

  const markEnd = useCallback(() => {
    if (!videoRef.current) return
    setPendingEnd(videoRef.current.currentTime)
  }, [])

  const addSegment = useCallback(() => {
    if (pendingStart === null || pendingEnd === null) return
    const start = Math.min(pendingStart, pendingEnd)
    const end = Math.max(pendingStart, pendingEnd)
    if (end - start < 0.1) return
    setSegments(prev => [
      ...prev,
      {
        start,
        end,
        label: selectedLabel,
        confidence: 1.0,
        skip_recommended: SKIP_LABELS.has(selectedLabel),
        reason: 'Manual label',
        signals_used: ['manual'],
      },
    ])
    setPendingStart(null)
    setPendingEnd(null)
  }, [pendingStart, pendingEnd, selectedLabel])

  const deleteSegment = useCallback((index: number) => {
    setSegments(prev => prev.filter((_, i) => i !== index))
  }, [])

  const clearPending = useCallback(() => {
    setPendingStart(null)
    setPendingEnd(null)
  }, [])

  // ── Export ────────────────────────────────────────────────────────────────

  const exportJson = () => {
    const sorted = [...segments].sort((a, b) => a.start - b.start)
    const data = {
      video_id: videoId,
      duration_seconds: duration,
      generated_at: new Date().toISOString(),
      segments: sorted,
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${videoId}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ── Keyboard shortcuts ────────────────────────────────────────────────────

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return
      switch (e.code) {
        case 'Space':  e.preventDefault(); togglePlay(); break
        case 'KeyI':   e.preventDefault(); markStart(); break
        case 'KeyO':   e.preventDefault(); markEnd(); break
        case 'Enter':  e.preventDefault(); if (canAdd) addSegment(); break
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [togglePlay, markStart, markEnd, addSegment, canAdd])

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col" style={{ fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <header className="flex items-center gap-3 px-4 py-3 bg-gray-800 border-b border-gray-700 flex-wrap">
        <h1 className="text-base font-bold mr-2">Content Map Labeler</h1>

        <label className="cursor-pointer px-3 py-1.5 bg-blue-700 hover:bg-blue-600 rounded text-sm">
          Load Video
          <input type="file" accept=".mp4,.m4v,.mov,.webm,.mkv,.avi,.ogg,video/*" className="hidden" onChange={handleVideoLoad} />
        </label>

        {videoFilename && (
          <span className="text-sm text-gray-400 truncate max-w-xs">{videoFilename}</span>
        )}

        {segments.length > 0 && (
          <button
            onClick={exportJson}
            className="ml-auto px-3 py-1.5 bg-green-700 hover:bg-green-600 rounded text-sm font-medium"
          >
            Export JSON ({segments.length} segments)
          </button>
        )}
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        <div className="flex flex-col flex-1 min-w-0">
          {/* Video area */}
          <div className="bg-black flex items-center justify-center" style={{ aspectRatio: '16/9', maxHeight: '55vh' }}>
            {videoSrc ? (
              <video
                ref={videoRef}
                src={videoSrc}
                className="max-h-full max-w-full"
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
              />
            ) : (
              <div className="text-gray-500 text-sm text-center">
                <div className="text-4xl mb-2">✂</div>
                Load a video file to start labeling
              </div>
            )}
          </div>

          <LabelingControls
            isPlaying={isPlaying}
            currentTime={currentTime}
            pendingStart={pendingStart}
            pendingEnd={pendingEnd}
            selectedLabel={selectedLabel}
            canAdd={canAdd}
            disabled={!videoSrc}
            onTogglePlay={togglePlay}
            onMarkStart={markStart}
            onMarkEnd={markEnd}
            onLabelChange={setSelectedLabel}
            onAddSegment={addSegment}
            onClearPending={clearPending}
          />

          {duration > 0 && (
            <Timeline
              segments={segments}
              duration={duration}
              currentTime={currentTime}
              pendingStart={pendingStart}
              pendingEnd={pendingEnd}
              onSeek={handleSeek}
            />
          )}
        </div>

        {/* Sidebar */}
        <SegmentList
          segments={segments}
          onDelete={deleteSegment}
          onSeek={handleSeek}
        />
      </div>
    </div>
  )
}
