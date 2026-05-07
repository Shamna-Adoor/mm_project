import { useState, useRef, useEffect } from 'react'
import { formatTime } from '../types'

interface Message {
  id:         number
  role:       'user' | 'assistant' | 'system'
  text:       string
  seek_to?:   number
  timestamps?: { time: number; label: string }[]
  loading?:   boolean
}

interface ChatPanelProps {
  jobId:    string
  onSeek:   (t: number) => void
  disabled: boolean
}

const SUGGESTIONS = [
  'What is this video about?',
  'Go to the intro',
  'Find the sponsor segment',
  'Summarize the main topics',
  'Jump to the most interesting part',
]

let msgId = 0

export default function ChatPanel({ jobId, onSeek, disabled }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: ++msgId, role: 'assistant',
      text: "Hi! I've read the full transcript. Ask me anything about this video, or tell me where to jump — e.g. \"go to the sponsor\" or \"what's discussed at 5 minutes?\"",
    },
  ])
  const [input,   setInput]   = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef  = useRef<HTMLInputElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (text: string = input.trim()) => {
    if (!text || loading || disabled) return
    setInput('')

    const userMsg: Message = { id: ++msgId, role: 'user', text }
    const placeholder: Message = { id: ++msgId, role: 'assistant', text: '', loading: true }

    setMessages(prev => [...prev, userMsg, placeholder])
    setLoading(true)

    // Build history from current messages
    const history = messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => ({ role: m.role, content: m.text }))
    history.push({ role: 'user', content: text })

    try {
      const res = await fetch(`http://localhost:8000/api/chat/${jobId}`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: text, history }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()

      const reply: Message = {
        id:         placeholder.id,
        role:       'assistant',
        text:       data.reply ?? '…',
        seek_to:    data.seek_to ?? undefined,
        timestamps: data.timestamps ?? [],
      }

      setMessages(prev => prev.map(m => m.id === placeholder.id ? reply : m))

      // Auto-seek if LLM provided a timestamp
      if (typeof data.seek_to === 'number') {
        onSeek(data.seek_to)
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === placeholder.id
          ? { ...m, loading: false, text: 'Error: ' + String(err) }
          : m
      ))
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  return (
    <div className="flex flex-col h-full" style={{ background: '#0a0f1e' }}>
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
        {messages.map(msg => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role !== 'user' && (
              <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center text-xs font-bold mr-2 flex-shrink-0 mt-0.5">
                AI
              </div>
            )}
            <div
              className={`max-w-[88%] rounded-2xl px-3 py-2 text-xs leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white rounded-br-sm'
                  : 'text-slate-200 rounded-bl-sm'
              }`}
              style={msg.role !== 'user' ? { background: '#1e293b', border: '1px solid #334155' } : {}}
            >
              {msg.loading ? (
                <div className="flex items-center gap-1 py-1">
                  {[0, 1, 2].map(i => (
                    <div key={i} className="w-1.5 h-1.5 rounded-full bg-blue-400"
                      style={{ animation: 'bounce 1s ease-in-out infinite', animationDelay: `${i * 0.15}s` }} />
                  ))}
                </div>
              ) : (
                <>
                  <p className="whitespace-pre-wrap">{msg.text}</p>

                  {/* Timestamps in AI reply */}
                  {msg.timestamps && msg.timestamps.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {msg.timestamps.map((ts, i) => (
                        <button
                          key={i}
                          onClick={() => onSeek(ts.time)}
                          className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-mono bg-blue-900/60 hover:bg-blue-800/80 text-blue-300 border border-blue-700/40 transition-colors"
                        >
                          ▶ {formatTime(ts.time)}{ts.label ? ` · ${ts.label}` : ''}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Auto-seek notification */}
                  {msg.seek_to !== undefined && (
                    <div
                      className="flex items-center gap-1.5 mt-2 px-2 py-1 rounded-lg text-xs"
                      style={{ background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.3)' }}
                    >
                      <span className="text-blue-400">⏩</span>
                      <span className="text-blue-300">Jumped to {formatTime(msg.seek_to)}</span>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Suggestions (shown only when no history) */}
      {messages.length === 1 && (
        <div className="px-3 pb-2 flex flex-col gap-1">
          {SUGGESTIONS.map(s => (
            <button
              key={s}
              onClick={() => send(s)}
              disabled={disabled || loading}
              className="text-left text-xs px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition-colors disabled:opacity-40"
              style={{ background: '#1e293b', border: '1px solid #334155' }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div
        className="flex items-center gap-2 px-3 py-2 flex-shrink-0"
        style={{ borderTop: '1px solid #1e293b' }}
      >
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={disabled || loading}
          placeholder={disabled ? 'Analyze a video first…' : 'Ask anything about this video…'}
          className="flex-1 text-xs rounded-lg px-3 py-2 outline-none disabled:opacity-40"
          style={{ background: '#1e293b', border: '1px solid #334155', color: '#f1f5f9' }}
        />
        <button
          onClick={() => send()}
          disabled={!input.trim() || disabled || loading}
          className="w-7 h-7 flex items-center justify-center rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-30 text-white text-xs transition-colors flex-shrink-0"
        >
          ↑
        </button>
      </div>
    </div>
  )
}
