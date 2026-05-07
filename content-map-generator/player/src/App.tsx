import { useState, useRef, useEffect, useCallback } from 'react'
import type { Segment, SegmentMap } from './types'
import { LABEL_COLORS, LABEL_NAMES, formatTime, segmentStats } from './types'
import SeekBar       from './components/SeekBar'
import Controls      from './components/Controls'
import Chapters      from './components/Chapters'
import SegmentList   from './components/SegmentList'
import LoadingScreen from './components/LoadingScreen'
import ChatPanel     from './components/ChatPanel'

const API = 'http://localhost:8000'

interface AnalyzeProgress {
  percent:      number
  message:      string
  stage_num:    number
  total_stages: number
  status:       string
}

type SidebarTab = 'chapters' | 'segments' | 'chat'

export default function App() {
  const videoRef      = useRef<HTMLVideoElement>(null)
  const videoWrapRef  = useRef<HTMLDivElement>(null)
  const pollRef       = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Core state ────────────────────────────────────────────────────────────
  const [videoSrc,        setVideoSrc]        = useState<string | null>(null)
  const [videoName,       setVideoName]       = useState<string>('')
  const [segmentMap,      setSegmentMap]      = useState<SegmentMap | null>(null)
  const [currentTime,     setCurrentTime]     = useState(0)
  const [duration,        setDuration]        = useState(0)
  const [isPlaying,       setIsPlaying]       = useState(false)
  const [contentOnlyMode, setContentOnlyMode] = useState(false)

  // ── Analysis state ────────────────────────────────────────────────────────
  const [analyzing,    setAnalyzing]    = useState(false)
  const [analyzeError, setAnalyzeError] = useState<string | null>(null)
  const [progress,     setProgress]     = useState<AnalyzeProgress | null>(null)

  // ── Player controls ───────────────────────────────────────────────────────
  const [volume,       setVolume]       = useState(1)
  const [muted,        setMuted]        = useState(false)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [sidebarTab,   setSidebarTab]   = useState<SidebarTab>('chapters')
  const [ytUrl,        setYtUrl]        = useState('')
  const [ytInputOpen,  setYtInputOpen]  = useState(false)
  const [jobId,        setJobId]        = useState<string | null>(null)

  // ── Refs for callbacks ────────────────────────────────────────────────────
  const contentOnlyRef  = useRef(false)
  const segmentMapRef   = useRef<SegmentMap | null>(null)
  useEffect(() => { contentOnlyRef.current = contentOnlyMode }, [contentOnlyMode])
  useEffect(() => { segmentMapRef.current  = segmentMap      }, [segmentMap])

  const currentSegment: Segment | null =
    segmentMap?.segments.find(s => currentTime >= s.start && currentTime < s.end) ?? null

  // ── File upload + analysis ────────────────────────────────────────────────
  const handleVideoLoad = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (videoSrc) URL.revokeObjectURL(videoSrc)

    setVideoSrc(URL.createObjectURL(file))
    setVideoName(file.name)
    setCurrentTime(0)
    setIsPlaying(false)
    setSegmentMap(null)
    setAnalyzeError(null)
    setProgress(null)
    if (pollRef.current) clearInterval(pollRef.current)
    setAnalyzing(true)

    let jobId: string
    try {
      const form = new FormData()
      form.append('video', file)
      let res: Response
      try {
        res = await fetch(`${API}/api/analyze`, { method: 'POST', body: form })
      } catch {
        throw new Error('Cannot reach analysis server. Run "make player" first.')
      }
      if (!res.ok) throw new Error(await res.text())
      jobId = (await res.json()).job_id
      setJobId(jobId)
    } catch (err) {
      setAnalyzeError(err instanceof Error ? err.message : String(err))
      setAnalyzing(false)
      e.target.value = ''
      return
    }

    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/status/${jobId}`)
        if (!r.ok) return
        const p: AnalyzeProgress = await r.json()
        setProgress(p)
        if (p.status === 'done') {
          clearInterval(pollRef.current!)
          const result = await (await fetch(`${API}/api/result/${jobId}`)).json()
          setSegmentMap({
            video_id:         result.video_id         ?? jobId,
            duration_seconds: result.duration_seconds ?? 0,
            generated_at:     result.generated_at     ?? new Date().toISOString(),
            segments:         result.segments,
            chapters:         result.chapters ?? [],
          })
          setAnalyzing(false)
          setProgress(null)
        } else if (p.status === 'error') {
          clearInterval(pollRef.current!)
          setAnalyzeError(p.message)
          setAnalyzing(false)
          setProgress(null)
        }
      } catch { /* transient */ }
    }, 2000)

    e.target.value = ''
  }

  // ── YouTube URL submit ────────────────────────────────────────────────────
  const handleYouTubeSubmit = async (url: string) => {
    const trimmed = url.trim()
    if (!trimmed) return
    setYtInputOpen(false)
    setYtUrl('')
    setSegmentMap(null)
    setAnalyzeError(null)
    setProgress(null)
    setVideoName(trimmed)
    if (pollRef.current) clearInterval(pollRef.current)
    setAnalyzing(true)

    let jid: string
    try {
      const form = new FormData()
      form.append('youtube_url', trimmed)
      let res: Response
      try {
        res = await fetch(`${API}/api/youtube`, { method: 'POST', body: form })
      } catch {
        throw new Error('Cannot reach analysis server.')
      }
      if (!res.ok) throw new Error(await res.text())
      jid = (await res.json()).job_id
      setJobId(jid)
    } catch (err) {
      setAnalyzeError(err instanceof Error ? err.message : String(err))
      setAnalyzing(false)
      return
    }

    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/status/${jid}`)
        if (!r.ok) return
        const p = await r.json()
        setProgress(p)
        if (p.status === 'done') {
          clearInterval(pollRef.current!)
          const result = await (await fetch(`${API}/api/result/${jid}`)).json()
          // For YouTube videos, set video src to server endpoint
          setVideoSrc(`${API}/api/video/${jid}`)
          setSegmentMap({
            video_id:         result.video_id         ?? jid,
            duration_seconds: result.duration_seconds ?? 0,
            generated_at:     result.generated_at     ?? new Date().toISOString(),
            segments:         result.segments,
            chapters:         result.chapters ?? [],
          })
          setAnalyzing(false)
          setProgress(null)
        } else if (p.status === 'error') {
          clearInterval(pollRef.current!)
          setAnalyzeError(p.message)
          setAnalyzing(false)
          setProgress(null)
        }
      } catch { /* transient */ }
    }, 2000)
  }

  // ── Playback helpers ──────────────────────────────────────────────────────
  const jumpToNextContent = useCallback((fromTime: number) => {
    const segs = segmentMapRef.current?.segments
    if (!segs || !videoRef.current) return
    const cur = segs.find(s => fromTime >= s.start && fromTime < s.end)
    if (!cur?.skip_recommended) return
    const next = segs.find(s => s.start >= cur.end && !s.skip_recommended)
    videoRef.current.currentTime = next ? next.start : (segs[segs.length - 1]?.end ?? fromTime)
  }, [])

  const handleTimeUpdate = useCallback(() => {
    if (!videoRef.current) return
    const t = videoRef.current.currentTime
    setCurrentTime(t)
    if (contentOnlyRef.current) jumpToNextContent(t)
  }, [jumpToNextContent])

  const handleLoadedMetadata = () => {
    if (videoRef.current) setDuration(videoRef.current.duration)
  }

  const togglePlay = useCallback(() => {
    if (!videoRef.current) return
    videoRef.current.paused ? videoRef.current.play() : videoRef.current.pause()
  }, [])

  const skipCurrentNonContent = useCallback(() => {
    if (videoRef.current) jumpToNextContent(videoRef.current.currentTime)
  }, [jumpToNextContent])

  const skipToNextSegment = useCallback(() => {
    if (!videoRef.current || !segmentMapRef.current) return
    const next = segmentMapRef.current.segments.find(s => s.start > videoRef.current!.currentTime + 0.1)
    if (next) videoRef.current.currentTime = next.start
  }, [])

  const handleSeek = useCallback((t: number) => {
    if (videoRef.current) videoRef.current.currentTime = t
  }, [])

  const handleVolumeChange = useCallback((v: number) => {
    setVolume(v)
    if (videoRef.current) videoRef.current.volume = v
    if (v > 0) setMuted(false)
  }, [])

  const handleToggleMute = useCallback(() => {
    setMuted(m => {
      if (videoRef.current) videoRef.current.muted = !m
      return !m
    })
  }, [])

  const handlePlaybackRate = useCallback((r: number) => {
    setPlaybackRate(r)
    if (videoRef.current) videoRef.current.playbackRate = r
  }, [])

  const handleFullscreen = useCallback(() => {
    const el = videoWrapRef.current
    if (!el) return
    if (document.fullscreenElement) document.exitFullscreen()
    else el.requestFullscreen?.()
  }, [])

  const seekRelative = useCallback((delta: number) => {
    if (!videoRef.current) return
    videoRef.current.currentTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + delta))
  }, [duration])

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return
      switch (e.code) {
        case 'Space':      e.preventDefault(); togglePlay(); break
        case 'ArrowRight': e.preventDefault(); skipToNextSegment(); break
        case 'ArrowLeft':  e.preventDefault(); seekRelative(-10); break
        case 'KeyS':       e.preventDefault(); skipCurrentNonContent(); break
        case 'KeyF':       e.preventDefault(); handleFullscreen(); break
        case 'KeyM':       e.preventDefault(); handleToggleMute(); break
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [togglePlay, skipToNextSegment, seekRelative, skipCurrentNonContent, handleFullscreen, handleToggleMute])

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  // ── Derived values ────────────────────────────────────────────────────────
  const effectiveDuration = duration || segmentMap?.duration_seconds || 0
  const stats = segmentMap ? segmentStats(segmentMap.segments, effectiveDuration) : null
  const isSkipSegment = currentSegment?.skip_recommended ?? false

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div
      className="flex flex-col h-screen overflow-hidden"
      style={{ background: '#0f172a', color: '#f1f5f9', fontFamily: 'system-ui, sans-serif' }}
    >
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header
        className="flex items-center gap-3 px-4 py-2.5 flex-shrink-0 flex-wrap"
        style={{ background: '#0f172a', borderBottom: '1px solid #1e293b' }}
      >
        <div className="flex items-center gap-2 mr-2">
          <div className="w-6 h-6 rounded bg-blue-600 flex items-center justify-center text-white text-xs font-bold">▶</div>
          <span className="font-bold text-sm text-white">Content Map</span>
        </div>

        {/* Local file */}
        <label className={`cursor-pointer px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
          analyzing ? 'bg-slate-700 opacity-60 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-500 text-white'
        }`}>
          {analyzing ? 'Analyzing…' : '+ Local File'}
          <input
            type="file"
            accept=".mp4,.m4v,.mov,.webm,.mkv,.avi,video/*"
            className="hidden"
            disabled={analyzing}
            onChange={handleVideoLoad}
          />
        </label>

        {/* YouTube URL */}
        {ytInputOpen ? (
          <div className="flex items-center gap-1">
            <input
              autoFocus
              value={ytUrl}
              onChange={e => setYtUrl(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') handleYouTubeSubmit(ytUrl)
                if (e.key === 'Escape') setYtInputOpen(false)
              }}
              placeholder="https://youtube.com/watch?v=..."
              className="text-xs rounded-lg px-2.5 py-1.5 w-64 outline-none"
              style={{ background: '#1e293b', border: '1px solid #3b82f6', color: '#f1f5f9' }}
            />
            <button
              onClick={() => handleYouTubeSubmit(ytUrl)}
              disabled={!ytUrl.trim()}
              className="px-2.5 py-1.5 rounded-lg bg-red-600 hover:bg-red-500 disabled:opacity-40 text-white text-xs font-medium transition-colors"
            >
              ▶
            </button>
            <button onClick={() => setYtInputOpen(false)} className="text-slate-500 hover:text-white text-xs px-1">✕</button>
          </div>
        ) : (
          <button
            onClick={() => setYtInputOpen(true)}
            disabled={analyzing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium bg-red-700 hover:bg-red-600 disabled:opacity-50 text-white transition-colors"
          >
            <span className="text-base leading-none">▶</span> YouTube
          </button>
        )}

        {videoName && !analyzing && (
          <span className="text-xs text-slate-500 truncate max-w-48" title={videoName}>{videoName}</span>
        )}

        {analyzeError && (
          <div className="flex items-center gap-2 text-xs text-red-400 bg-red-900/20 border border-red-800/40 px-3 py-1.5 rounded-lg">
            <span>⚠</span>
            <span className="truncate max-w-80">{analyzeError}</span>
          </div>
        )}

        {segmentMap && !analyzing && (
          <div className="ml-auto flex items-center gap-3">
            <span className="text-xs text-green-400 bg-green-900/20 border border-green-800/30 px-2.5 py-1 rounded-full">
              ✓ {segmentMap.segments.length} segments
            </span>
            <span className="text-xs text-blue-400 bg-blue-900/20 border border-blue-800/30 px-2.5 py-1 rounded-full">
              AI-powered
            </span>
            <a
              href={`${API}/api/transcript/${jobId ?? segmentMap.video_id}`}
              download
              className="text-xs px-3 py-1 rounded-lg font-medium bg-slate-700 hover:bg-slate-600 text-slate-200 transition-colors"
            >
              ↓ Transcript
            </a>
          </div>
        )}
      </header>

      {/* ── Body ───────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Left: video + controls ─────────────────────────────────────── */}
        <div className="flex flex-col flex-1 min-w-0 overflow-hidden">

          {/* Video area */}
          <div
            ref={videoWrapRef}
            className="relative bg-black flex items-center justify-center flex-1 min-h-0"
          >
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
              <div className="text-center select-none">
                <div className="text-6xl mb-4 opacity-20">▶</div>
                <div className="text-sm text-slate-500">Load a video to begin</div>
                <div className="text-xs text-slate-600 mt-1">MP4, MOV, MKV, WebM supported</div>
              </div>
            )}

            {/* Analysis progress overlay */}
            {analyzing && (
              <LoadingScreen
                message={progress?.message ?? 'Starting…'}
                stageNum={progress?.stage_num ?? 0}
                totalStages={progress?.total_stages ?? 9}
                percent={progress?.percent ?? 0}
                videoName={videoName}
              />
            )}

            {/* Non-content banner (when in a skip segment and content-only is OFF) */}
            {!analyzing && isSkipSegment && !contentOnlyMode && currentSegment && (
              <div
                className="absolute top-3 left-3 right-3 flex items-center justify-between px-4 py-2 rounded-xl text-sm shadow-2xl"
                style={{
                  background: LABEL_COLORS[currentSegment.label].bar + '22',
                  border: `1px solid ${LABEL_COLORS[currentSegment.label].bar}55`,
                  backdropFilter: 'blur(8px)',
                }}
              >
                <div className="flex items-center gap-2">
                  <div
                    className="w-2 h-2 rounded-full animate-pulse"
                    style={{ background: LABEL_COLORS[currentSegment.label].bar }}
                  />
                  <span className="font-semibold" style={{ color: LABEL_COLORS[currentSegment.label].bar }}>
                    {LABEL_NAMES[currentSegment.label]}
                  </span>
                  <span className="text-slate-400 text-xs">
                    · ends at {formatTime(currentSegment.end)}
                  </span>
                </div>
                <button
                  onClick={skipCurrentNonContent}
                  className="text-xs px-3 py-1 rounded-lg font-semibold transition-colors hover:opacity-90"
                  style={{
                    background: LABEL_COLORS[currentSegment.label].bar,
                    color: '#0f172a',
                  }}
                >
                  Skip →
                </button>
              </div>
            )}
          </div>

          {/* Seek bar */}
          {effectiveDuration > 0 && (
            <SeekBar
              currentTime={currentTime}
              duration={effectiveDuration}
              segments={segmentMap?.segments}
              chapters={segmentMap?.chapters}
              onSeek={handleSeek}
            />
          )}

          {/* Controls */}
          <Controls
            isPlaying={isPlaying}
            contentOnlyMode={contentOnlyMode}
            currentSegment={currentSegment}
            hasSegments={!!segmentMap}
            disabled={!videoSrc}
            volume={volume}
            muted={muted}
            playbackRate={playbackRate}
            currentTime={currentTime}
            duration={effectiveDuration}
            onTogglePlay={togglePlay}
            onSeekRelative={seekRelative}
            onSkipNonContent={skipCurrentNonContent}
            onToggleContentOnly={() => setContentOnlyMode(m => {
              const next = !m
              if (next && videoRef.current) jumpToNextContent(videoRef.current.currentTime)
              return next
            })}
            onVolumeChange={handleVolumeChange}
            onToggleMute={handleToggleMute}
            onPlaybackRateChange={handlePlaybackRate}
            onFullscreen={handleFullscreen}
          />

          {/* Stats bar */}
          {stats && (
            <div
              className="flex items-center gap-4 px-4 py-2 text-xs flex-wrap"
              style={{ background: '#0a0f1e', borderTop: '1px solid #1e293b' }}
            >
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                <span className="text-slate-400">Content</span>
                <span className="text-white font-mono font-medium">{formatTime(stats.contentSecs)}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-red-500" />
                <span className="text-slate-400">Skip</span>
                <span className="text-white font-mono font-medium">{formatTime(stats.skipSecs)}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-slate-500">Time saved</span>
                <span
                  className="font-bold"
                  style={{ color: stats.savedPct > 0 ? '#22c55e' : '#64748b' }}
                >
                  {stats.savedPct}%
                </span>
              </div>
              <div className="ml-auto text-slate-600 hidden sm:block">
                Space · S · ← → · M · F
              </div>
            </div>
          )}
        </div>

        {/* ── Right sidebar ──────────────────────────────────────────────── */}
        {segmentMap && (
          <aside
            className="w-64 flex-shrink-0 flex flex-col border-l"
            style={{ background: '#0f172a', borderColor: '#1e293b' }}
          >
            {/* Tabs */}
            <div className="flex border-b flex-shrink-0" style={{ borderColor: '#1e293b' }}>
              {[
                { id: 'chapters', label: segmentMap.chapters?.length ? `Chaps (${segmentMap.chapters.length})` : 'Chaps' },
                { id: 'segments', label: `Segs (${segmentMap.segments.length})` },
                { id: 'chat',     label: '💬 Chat' },
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setSidebarTab(tab.id as SidebarTab)}
                  className={`flex-1 py-2 text-xs font-semibold tracking-wide transition-colors ${
                    sidebarTab === tab.id
                      ? 'text-blue-400 border-b-2 border-blue-500'
                      : 'text-slate-500 hover:text-slate-300'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab content */}
            {sidebarTab === 'chapters' ? (
              segmentMap.chapters && segmentMap.chapters.length > 0 ? (
                <Chapters
                  chapters={segmentMap.chapters}
                  segments={segmentMap.segments}
                  currentTime={currentTime}
                  onSeek={handleSeek}
                />
              ) : (
                <div className="flex-1 flex items-center justify-center text-xs text-slate-600 p-4 text-center">
                  No chapters generated.<br />Ollama may not be running.
                </div>
              )
            ) : sidebarTab === 'segments' ? (
              <SegmentList
                segments={segmentMap.segments}
                currentTime={currentTime}
                onSeek={handleSeek}
              />
            ) : (
              <ChatPanel
                jobId={jobId ?? segmentMap.video_id}
                onSeek={handleSeek}
                disabled={!segmentMap}
              />
            )}
          </aside>
        )}
      </div>
    </div>
  )
}
