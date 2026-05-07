import { useEffect, useRef, useState, useCallback } from 'react'

const COLS = 18
const ROWS = 14
const CELL = 16
const W    = COLS * CELL
const H    = ROWS * CELL
const TICK = 120   // ms per frame

type Dir = 'U' | 'D' | 'L' | 'R'
type Pt  = { x: number; y: number }

const rand = (max: number) => Math.floor(Math.random() * max)
const eq   = (a: Pt, b: Pt) => a.x === b.x && a.y === b.y

function newFood(snake: Pt[]): Pt {
  let p: Pt
  do { p = { x: rand(COLS), y: rand(ROWS) } } while (snake.some(s => eq(s, p)))
  return p
}

const FOOD_LABELS = ['AD', 'SKIP', 'INTRO', 'AD', 'OUTRO', 'SKIP', 'AD']

export default function MiniGame() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stateRef  = useRef({
    snake: [{ x: 9, y: 7 }, { x: 8, y: 7 }, { x: 7, y: 7 }] as Pt[],
    dir:   'R' as Dir,
    next:  'R' as Dir,
    food:  { x: 14, y: 4 } as Pt,
    score: 0,
    dead:  false,
    foodLabel: 'AD',
  })
  const [score, setScore] = useState(0)
  const [dead,  setDead]  = useState(false)
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const reset = useCallback(() => {
    stateRef.current = {
      snake: [{ x: 9, y: 7 }, { x: 8, y: 7 }, { x: 7, y: 7 }],
      dir: 'R', next: 'R',
      food: { x: 14, y: 4 },
      score: 0, dead: false,
      foodLabel: 'AD',
    }
    setScore(0)
    setDead(false)
  }, [])

  // Draw
  const draw = useCallback(() => {
    const cv = canvasRef.current
    if (!cv) return
    const ctx = cv.getContext('2d')!
    const s   = stateRef.current

    // Background grid
    ctx.fillStyle = '#0a0f1e'
    ctx.fillRect(0, 0, W, H)
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        ctx.fillStyle = (r + c) % 2 === 0 ? '#0d1424' : '#0a0f1e'
        ctx.fillRect(c * CELL, r * CELL, CELL, CELL)
      }
    }

    // Food (pulsing red block)
    const f = s.food
    ctx.fillStyle = '#ef4444'
    ctx.shadowBlur = 10
    ctx.shadowColor = '#ef4444'
    ctx.fillRect(f.x * CELL + 1, f.y * CELL + 1, CELL - 2, CELL - 2)
    ctx.shadowBlur = 0
    ctx.fillStyle = '#fff'
    ctx.font = `bold ${CELL * 0.45}px monospace`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(s.foodLabel, f.x * CELL + CELL / 2, f.y * CELL + CELL / 2)

    // Snake
    s.snake.forEach((seg, i) => {
      const ratio = 1 - i / s.snake.length
      const r = Math.round(34  + ratio * (59  - 34))
      const g = Math.round(197 + ratio * (130 - 197))
      const b = Math.round(94  + ratio * (246 - 94))
      ctx.fillStyle = `rgb(${r},${g},${b})`
      if (i === 0) {
        ctx.shadowBlur = 8
        ctx.shadowColor = '#22c55e'
      }
      const pad = i === 0 ? 1 : 2
      ctx.fillRect(seg.x * CELL + pad, seg.y * CELL + pad, CELL - pad * 2, CELL - pad * 2)
      ctx.shadowBlur = 0
    })

    // Dead overlay
    if (s.dead) {
      ctx.fillStyle = 'rgba(0,0,0,0.6)'
      ctx.fillRect(0, 0, W, H)
      ctx.fillStyle = '#ef4444'
      ctx.font = 'bold 16px system-ui'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('GAME OVER', W / 2, H / 2 - 10)
      ctx.fillStyle = '#94a3b8'
      ctx.font = '11px system-ui'
      ctx.fillText('Press Space or Enter to restart', W / 2, H / 2 + 10)
    }
  }, [])

  // Tick
  const tick = useCallback(() => {
    const s = stateRef.current
    if (s.dead) return

    s.dir = s.next
    const head = { ...s.snake[0] }
    if (s.dir === 'U') head.y--
    if (s.dir === 'D') head.y++
    if (s.dir === 'L') head.x--
    if (s.dir === 'R') head.x++

    // Wall or self collision
    if (head.x < 0 || head.x >= COLS || head.y < 0 || head.y >= ROWS || s.snake.some(p => eq(p, head))) {
      s.dead = true
      setDead(true)
      draw()
      return
    }

    s.snake.unshift(head)

    if (eq(head, s.food)) {
      s.score++
      setScore(s.score)
      s.food = newFood(s.snake)
      s.foodLabel = FOOD_LABELS[s.score % FOOD_LABELS.length]
    } else {
      s.snake.pop()
    }

    draw()
  }, [draw])

  // Keys
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const s = stateRef.current
      const d = s.dir
      if      (e.key === 'ArrowUp'    && d !== 'D') stateRef.current.next = 'U'
      else if (e.key === 'ArrowDown'  && d !== 'U') stateRef.current.next = 'D'
      else if (e.key === 'ArrowLeft'  && d !== 'R') stateRef.current.next = 'L'
      else if (e.key === 'ArrowRight' && d !== 'L') stateRef.current.next = 'R'
      else if ((e.key === ' ' || e.key === 'Enter') && s.dead) reset()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [reset])

  // Game loop
  useEffect(() => {
    tickRef.current = setInterval(tick, TICK)
    return () => { if (tickRef.current) clearInterval(tickRef.current) }
  }, [tick])

  // Restart on dead state change
  useEffect(() => {
    if (!dead) {
      if (tickRef.current) clearInterval(tickRef.current)
      tickRef.current = setInterval(tick, TICK)
      draw()
    }
  }, [dead, tick, draw])

  // Initial draw
  useEffect(() => { draw() }, [draw])

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex items-center justify-between w-full px-1">
        <span className="text-xs text-slate-500 uppercase tracking-wider">Mini Game</span>
        <span className="text-xs font-mono text-green-400">Score: {score}</span>
      </div>

      <div className="relative rounded-lg overflow-hidden" style={{ border: '1px solid #1e3a5f', boxShadow: '0 0 20px rgba(59,130,246,0.1)' }}>
        <canvas ref={canvasRef} width={W} height={H} />
      </div>

      <div className="text-xs text-slate-600 text-center">
        Arrow keys to move · Eat the <span className="text-red-400 font-mono">AD</span>/<span className="text-red-400 font-mono">SKIP</span> blocks
      </div>
    </div>
  )
}
