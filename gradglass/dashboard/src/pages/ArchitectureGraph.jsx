import { useState, useCallback, useEffect, useMemo } from 'react'
import { useParams } from 'react-router-dom'
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Handle,
  Position,
  MarkerType,
} from 'reactflow'
import dagre from '@dagrejs/dagre'
import 'reactflow/dist/style.css'
import { fetchArchitecture, fetchGradients } from '../api'
import { LoadingSpinner, ErrorMessage, EmptyState } from '../components/ui'
import { Network, Search, Edit3, Save, ChevronDown, ChevronRight } from 'lucide-react'

// ─────────────────────────────────────────────────────────────────
// Category colours
// ─────────────────────────────────────────────────────────────────
const CAT = {
  model: {
    node:    'bg-violet-500/15 border-violet-500/60 text-violet-200',
    group:   'rgba(139,92,246,0.06)',
    dash:    '#7c3aed',
    badge:   'bg-violet-500/20 text-violet-300',
    minimap: '#7c3aed',
    glow:    '0 0 18px rgba(139,92,246,0.35)',
  },
  data: {
    node:    'bg-sky-500/15 border-sky-500/60 text-sky-200',
    group:   'rgba(14,165,233,0.06)',
    dash:    '#0284c7',
    badge:   'bg-sky-500/20 text-sky-300',
    minimap: '#0284c7',
    glow:    '0 0 18px rgba(14,165,233,0.35)',
  },
  training: {
    node:    'bg-orange-500/15 border-orange-500/60 text-orange-200',
    group:   'rgba(249,115,22,0.06)',
    dash:    '#ea580c',
    badge:   'bg-orange-500/20 text-orange-300',
    minimap: '#ea580c',
    glow:    '0 0 18px rgba(249,115,22,0.35)',
  },
  eval: {
    node:    'bg-emerald-500/15 border-emerald-500/60 text-emerald-200',
    group:   'rgba(16,185,129,0.06)',
    dash:    '#059669',
    badge:   'bg-emerald-500/20 text-emerald-300',
    minimap: '#059669',
    glow:    '0 0 18px rgba(16,185,129,0.35)',
  },
}
const getCAT = (cat) => CAT[cat] || CAT.model

// ─────────────────────────────────────────────────────────────────
// ReactFlow node types
// ─────────────────────────────────────────────────────────────────

/** Central root node */
function CoreNode({ data }) {
  return (
    <div
      className="rounded-2xl px-6 py-4 font-bold text-white text-center select-none"
      style={{
        background: 'linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%)',
        boxShadow: '0 0 32px rgba(99,102,241,0.55), 0 0 0 3px rgba(139,92,246,0.35)',
        minWidth: 170,
      }}
    >
      <Handle type="source" position={Position.Right} style={{ background: '#818cf8', border: 'none' }} />
      <div className="text-[11px] uppercase tracking-widest text-indigo-200 mb-1">Model Core</div>
      <div className="text-base font-mono leading-tight">{data.label}</div>
      {data.paramCount > 0 && (
        <div className="text-[10px] text-indigo-300 mt-1">{(data.paramCount / 1e6).toFixed(2)}M params</div>
      )}
    </div>
  )
}

/** Dashed group container for a top-level module */
function GroupNode({ data }) {
  const c = getCAT(data.category)
  return (
    <div
      className="rounded-xl select-none"
      style={{
        width: data.width,
        height: data.height,
        background: c.group,
        border: `2px dashed ${c.dash}`,
        boxShadow: data.collapsed ? 'none' : c.glow,
        transition: 'box-shadow 0.2s, height 0.25s',
        position: 'relative',
      }}
      onClick={data.onToggle}
    >
      <Handle type="target" position={Position.Left} style={{ background: c.dash, border: 'none', top: 24 }} />
      <Handle type="source" position={Position.Right} style={{ background: c.dash, border: 'none', top: 24 }} />
      {/* header */}
      <div
        className="flex items-center gap-2 px-3 py-2 cursor-pointer"
        style={{ borderBottom: `1px solid ${c.dash}44` }}
      >
        <span style={{ color: c.dash }}>
          {data.collapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
        </span>
        <span className="font-mono font-bold text-[11px] text-white">{data.label}</span>
        <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${c.badge}`}>{data.category}</span>
        <span className="text-[9px] text-slate-500 ml-auto">{data.childCount} sub-modules</span>
      </div>
      {data.collapsed && (
        <div className="px-3 py-2 text-[10px] text-slate-500 italic">click to expand</div>
      )}
    </div>
  )
}

/** Leaf layer node inside a group */
function LeafNode({ data, isConnectable }) {
  const c = getCAT(data.category)
  const isFrozen = !data.trainable
  return (
    <div
      className={`rounded-lg border px-3 py-2 min-w-[130px] select-none transition-all hover:brightness-110 ${c.node} ${isFrozen ? 'opacity-50' : ''}`}
      style={{ boxShadow: '0 2px 8px rgba(0,0,0,0.4)' }}
    >
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} style={{ background: '#64748b', border: 'none' }} />
      <div className="font-mono text-[10px] font-bold text-white/90 leading-tight truncate max-w-[160px]"
        title={data.id}>
        {data.shortLabel}
      </div>
      <div className={`text-[9px] px-1.5 py-0.5 rounded mt-1 inline-block ${c.badge}`}>{data.layerType}</div>
      {data.paramCount > 0 && (
        <div className="text-[9px] text-slate-400 mt-1">
          {data.paramCount >= 1e6
            ? `${(data.paramCount / 1e6).toFixed(1)}M`
            : data.paramCount >= 1e3
            ? `${(data.paramCount / 1e3).toFixed(0)}K`
            : data.paramCount} params
        </div>
      )}
      {isFrozen && <div className="text-[9px] text-slate-500 italic mt-0.5">frozen</div>}
      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} style={{ background: '#64748b', border: 'none' }} />
    </div>
  )
}

const nodeTypes = { core: CoreNode, group: GroupNode, leaf: LeafNode }

// ─────────────────────────────────────────────────────────────────
// Dagre layout helpers
// ─────────────────────────────────────────────────────────────────
const NODE_W = 170
const NODE_H = 56
const GROUP_PAD = 48       // top padding inside group (header height)
const GROUP_SIDE_PAD = 20

/**
 * Layout *within* a group using dagre (top-to-bottom subtree).
 * Returns { nodes: [{id,x,y}], width, height }
 */
function layoutGroup(children, layerMap) {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'TB', nodesep: 28, ranksep: 36, marginx: GROUP_SIDE_PAD, marginy: 10 })

  for (const cid of children) {
    g.setNode(cid, { width: NODE_W, height: NODE_H })
  }
  // edges within the group: child → its children (if they exist in this group set)
  const childSet = new Set(children)
  for (const cid of children) {
    const layer = layerMap[cid]
    if (layer) {
      for (const grandchild of (layer.children || [])) {
        if (childSet.has(grandchild)) {
          g.setEdge(cid, grandchild)
        }
      }
    }
  }
  dagre.layout(g)

  let maxX = 0, maxY = 0
  const positions = {}
  for (const cid of children) {
    const n = g.node(cid)
    if (n) {
      positions[cid] = { x: n.x - NODE_W / 2, y: n.y - NODE_H / 2 }
      maxX = Math.max(maxX, n.x + NODE_W / 2)
      maxY = Math.max(maxY, n.y + NODE_H / 2)
    }
  }
  return {
    positions,
    width:  Math.max(NODE_W + GROUP_SIDE_PAD * 2, maxX + GROUP_SIDE_PAD),
    height: Math.max(NODE_H + GROUP_PAD + 16, maxY + GROUP_PAD + 24),
  }
}

/**
 * Layout top-level groups using dagre (left-to-right from core).
 * Returns positions for each group node.
 */
function layoutTopLevel(topLevel, groupSizes, coreId) {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', nodesep: 60, ranksep: 120 })

  // core node
  g.setNode(coreId, { width: 190, height: 80 })
  // group nodes
  for (const id of topLevel) {
    const sz = groupSizes[id] || { width: 200, height: 120 }
    g.setNode(id, { width: sz.width, height: sz.height })
    g.setEdge(coreId, id)
  }
  // sequential data-flow edges between top-level siblings
  for (let i = 0; i < topLevel.length - 1; i++) {
    g.setEdge(topLevel[i], topLevel[i + 1])
  }

  dagre.layout(g)

  const positions = {}
  for (const id of [coreId, ...topLevel]) {
    const n = g.node(id)
    if (n) {
      const sz = id === coreId
        ? { width: 190, height: 80 }
        : groupSizes[id] || { width: 200, height: 120 }
      positions[id] = { x: n.x - sz.width / 2, y: n.y - sz.height / 2 }
    }
  }
  return positions
}

/** Collect ALL descendants of `id` that are present in the layerMap */
function allDescendants(id, layerMap) {
  const result = []
  const queue = [...(layerMap[id]?.children || [])]
  while (queue.length) {
    const cur = queue.shift()
    result.push(cur)
    for (const child of (layerMap[cur]?.children || [])) {
      queue.push(child)
    }
  }
  return result
}

/** Short display label from dotted id */
function shortName(id) {
  const parts = id.split('.')
  return parts[parts.length - 1]
}

function normalizeGradientLayerId(layerId) {
  if (!layerId) return layerId
  const normalized = layerId.replace(/\.(weight|bias|running_mean|running_var|num_batches_tracked|gamma|beta)$/i, '')
  return normalized || layerId
}

function buildFallbackArchitecture(gradients) {
  const summaries = gradients?.summaries || []
  if (!summaries.length) return null

  const latestLayers = summaries[summaries.length - 1]?.layers || {}
  const orderedIds = []
  const seen = new Set()

  for (const rawId of Object.keys(latestLayers)) {
    const normalizedId = normalizeGradientLayerId(rawId)
    if (!normalizedId || seen.has(normalizedId)) continue
    seen.add(normalizedId)
    orderedIds.push(normalizedId)
  }

  if (!orderedIds.length) return null

  return {
    root_type: 'Observed Training Graph',
    top_level: orderedIds,
    layers: orderedIds.map((id) => ({
      id,
      type: 'ObservedModule',
      params: {},
      param_count: 0,
      trainable: true,
      input_shape: null,
      output_shape: null,
      depth: 1,
      parent: '__root__',
      children: [],
      is_container: false,
      category: 'model',
    })),
    edges: orderedIds.slice(0, -1).map((id, index) => [id, orderedIds[index + 1]]),
  }
}

function isNotFoundError(message) {
  return typeof message === 'string' && (message.includes('404') || message.includes('No architecture data found'))
}

// ─────────────────────────────────────────────────────────────────
// Build ReactFlow nodes + edges from backend data
// ─────────────────────────────────────────────────────────────────
function buildGraph(architecture, collapsed, onToggle) {
  const layers = architecture.layers || []
  const backendEdges = architecture.edges || []
  const rootType = architecture.root_type || 'Model'
  const topLevel = architecture.top_level ||
    layers.filter(l => l.depth === 1).map(l => l.id)

  // id → layer map
  const layerMap = {}
  for (const l of layers) layerMap[l.id] = l

  // total param count for root
  const totalParams = layers.reduce((s, l) => s + (l.param_count || 0), 0)

  // ── layout each group's direct children ─────────────────────────
  const groupSizes = {}
  const groupChildPositions = {}   // groupId → { childId → {x, y} }

  for (const gid of topLevel) {
    const group = layerMap[gid]
    const directChildren = group?.children || []

    // When collapsed, size is fixed small
    if (collapsed[gid]) {
      groupSizes[gid] = { width: NODE_W + GROUP_SIDE_PAD * 2, height: 68 }
      groupChildPositions[gid] = {}
      continue
    }

    if (directChildren.length === 0) {
      // leaf top-level node — treat as single-child group
      groupSizes[gid] = { width: NODE_W + GROUP_SIDE_PAD * 2, height: NODE_H + GROUP_PAD + 24 }
      groupChildPositions[gid] = {}
      continue
    }

    // Collect all descendants to display inside this group
    const allDesc = allDescendants(gid, layerMap)
    const { positions, width, height } = layoutGroup(allDesc.length ? allDesc : directChildren, layerMap)
    groupSizes[gid] = { width, height }
    groupChildPositions[gid] = positions
  }

  // ── layout top-level groups + core ──────────────────────────────
  const CORE_ID = '__core__'
  const topPositions = layoutTopLevel(topLevel, groupSizes, CORE_ID)

  // ── emit ReactFlow nodes ─────────────────────────────────────────
  const rfNodes = []
  const rfEdges = []

  // Core node
  rfNodes.push({
    id: CORE_ID,
    type: 'core',
    position: topPositions[CORE_ID] || { x: 0, y: 0 },
    data: { label: rootType, paramCount: totalParams },
    zIndex: 10,
  })

  // Group nodes + their children
  for (const gid of topLevel) {
    const group = layerMap[gid]
    const sz = groupSizes[gid]
    const isCollapsed = !!collapsed[gid]
    const category = group?.category || 'model'
    const childCount = allDescendants(gid, layerMap).length || (group?.children?.length || 0)

    rfNodes.push({
      id: gid,
      type: 'group',
      position: topPositions[gid] || { x: 0, y: 0 },
      data: {
        label: shortName(gid),
        category,
        width: sz.width,
        height: sz.height,
        collapsed: isCollapsed,
        childCount,
        onToggle: () => onToggle(gid),
      },
      style: { width: sz.width, height: sz.height },
      zIndex: 1,
    })

    if (!isCollapsed) {
      const positions = groupChildPositions[gid] || {}
      const allDesc = allDescendants(gid, layerMap)
      const toPlace = allDesc.length ? allDesc : (group?.children || [])

      for (const cid of toPlace) {
        const layer = layerMap[cid]
        if (!layer) continue
        const pos = positions[cid] || { x: GROUP_SIDE_PAD, y: GROUP_PAD }

        rfNodes.push({
          id: cid,
          type: 'leaf',
          position: { x: pos.x, y: pos.y + GROUP_PAD },
          parentNode: gid,
          extent: 'parent',
          data: {
            id: cid,
            shortLabel: shortName(cid),
            layerType: layer.type,
            category: layer.category || 'model',
            paramCount: layer.param_count || 0,
            trainable: layer.trainable !== false,
          },
          zIndex: 5,
        })
      }
    }
  }

  // ── emit ReactFlow edges ─────────────────────────────────────────
  const edgeColor = '#4f46e5'

  // core → each top-level group
  for (const gid of topLevel) {
    rfEdges.push({
      id: `e-core-${gid}`,
      source: CORE_ID,
      target: gid,
      type: 'smoothstep',
      animated: false,
      style: { stroke: edgeColor, strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: edgeColor },
      label: 'features',
      labelStyle: { fill: '#94a3b8', fontSize: 9 },
      labelBgStyle: { fill: '#0f172a', fillOpacity: 0.8 },
    })
  }

  // sequential data-flow between top-level groups
  for (let i = 0; i < topLevel.length - 1; i++) {
    const src = topLevel[i]
    const dst = topLevel[i + 1]
    const c = getCAT(layerMap[dst]?.category || 'model')
    rfEdges.push({
      id: `e-seq-${src}-${dst}`,
      source: src,
      target: dst,
      type: 'smoothstep',
      animated: true,
      style: { stroke: c.dash, strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: c.dash },
      label: inferEdgeLabel(layerMap[src], layerMap[dst]),
      labelStyle: { fill: '#94a3b8', fontSize: 9 },
      labelBgStyle: { fill: '#0f172a', fillOpacity: 0.8 },
    })
  }

  // internal child → child edges within groups
  for (const gid of topLevel) {
    if (collapsed[gid]) continue
    const allDesc = new Set(allDescendants(gid, layerMap))
    for (const [src, dst] of backendEdges) {
      if (allDesc.has(src) && allDesc.has(dst)) {
        const c = getCAT(layerMap[dst]?.category || 'model')
        rfEdges.push({
          id: `e-${src}-${dst}`,
          source: src,
          target: dst,
          type: 'smoothstep',
          style: { stroke: c.dash + 'aa', strokeWidth: 1.5 },
          markerEnd: { type: MarkerType.ArrowClosed, color: c.dash },
        })
      }
    }
  }

  return { rfNodes, rfEdges }
}

/** Infer a human edge label from src/dst layer types */
function inferEdgeLabel(src, dst) {
  const srcT = (src?.type || '').toLowerCase()
  const dstT = (dst?.type || '').toLowerCase()
  if (dstT.includes('encoder') || srcT.includes('backbone')) return 'features'
  if (dstT.includes('decoder')) return 'encoded'
  if (dstT.includes('pool')) return 'sequence'
  if (dstT.includes('linear') || dstT.includes('embedding')) return 'pooled'
  if (dstT.includes('loss') || dstT.includes('criterion')) return 'logits'
  if (srcT.includes('encoder') && dstT.includes('decoder')) return 'z'
  return 'data'
}

// ─────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────
export default function ArchitectureGraph({ hideHeader, onNodeSelect, heightClass = "h-[calc(100vh-100px)]" }) {
  const { runId } = useParams()
  const [architecture, setArchitecture] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [isFallbackArchitecture, setIsFallbackArchitecture] = useState(false)
  const [collapsed, setCollapsed] = useState({})
  const [selectedNode, setSelectedNode] = useState(null)

  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])

  const layerMap = useMemo(() => {
    const m = {}
    for (const l of (architecture?.layers || [])) m[l.id] = l
    return m
  }, [architecture])

  const onToggle = useCallback((gid) => {
    setCollapsed(prev => ({ ...prev, [gid]: !prev[gid] }))
  }, [])

  // Rebuild graph whenever architecture or collapsed state changes
  useEffect(() => {
    if (!architecture?.layers?.length) return
    const { rfNodes, rfEdges } = buildGraph(architecture, collapsed, onToggle)
    setNodes(rfNodes)
    setEdges(rfEdges)
  }, [architecture, collapsed, onToggle, setNodes, setEdges])

  useEffect(() => {
    let cancelled = false

    setLoading(true)
    setError(null)
    setArchitecture(null)
    setIsFallbackArchitecture(false)

    Promise.allSettled([fetchArchitecture(runId), fetchGradients(runId)])
      .then(([architectureResult, gradientsResult]) => {
        if (cancelled) return

        const architectureData = architectureResult.status === 'fulfilled' ? architectureResult.value : null
        if (architectureData?.layers?.length) {
          setArchitecture(architectureData)
          return
        }

        const gradientsData = gradientsResult.status === 'fulfilled' ? gradientsResult.value : null
        const fallbackArchitecture = buildFallbackArchitecture(gradientsData)
        if (fallbackArchitecture?.layers?.length) {
          setArchitecture(fallbackArchitecture)
          setIsFallbackArchitecture(true)
          return
        }

        const architectureError = architectureResult.status === 'rejected' ? architectureResult.reason?.message : null
        const gradientsError = gradientsResult.status === 'rejected' ? gradientsResult.reason?.message : null
        const shouldShowError = [architectureError, gradientsError].some((message) => message && !isNotFoundError(message))

        if (shouldShowError) {
          setError(architectureError || gradientsError || 'Failed to load architecture data')
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [runId])

  const onConnect = useCallback(
    (params) => setEdges(eds => addEdge({ ...params, animated: true, style: { stroke: '#4f46e5', strokeWidth: 2 } }, eds)),
    [setEdges],
  )

  const onNodeClick = useCallback((_, node) => {
    if (node.type === 'leaf') {
      const selected = layerMap[node.id] || null;
      setSelectedNode(selected);
      if (onNodeSelect) onNodeSelect(selected);
    } else {
      setSelectedNode(null);
      if (onNodeSelect) onNodeSelect(null);
    }
  }, [layerMap, onNodeSelect])

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
    if (onNodeSelect) onNodeSelect(null);
  }, [onNodeSelect])

  if (loading) return <LoadingSpinner />
  if (error) return <ErrorMessage message={error} />
  if (!architecture?.layers?.length) {
    return (
      <EmptyState
        icon={Network}
        title="No architecture data"
        description="Architecture is captured when you call run.watch(model). If only gradient summaries are available, GradGlass will try to build a fallback DAG automatically."
      />
    )
  }

  return (
    <div className={`flex flex-col ${heightClass}`}>
      {/* Header */}
      {!hideHeader && (
        <div className="flex items-center justify-between mb-4 shrink-0">
        <div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
            Architecture Graph
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Branching tree of modules. Click a dashed group to collapse / expand it.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setCollapsed({})}
            className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-md text-sm transition"
          >
            Expand All
          </button>
          <button
            onClick={() => {
              const all = {}
              for (const l of (architecture?.top_level || [])) all[l] = true
              setCollapsed(all)
            }}
            className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-md text-sm transition"
          >
            Collapse All
          </button>
          <button className="flex items-center gap-2 px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-md text-sm font-medium transition shadow-lg shadow-indigo-500/20">
            <Save className="w-4 h-4" /> Save Layout
          </button>
        </div>
      </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-4 mb-3 shrink-0">
        {Object.entries(CAT).map(([cat, c]) => (
          <div key={cat} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full" style={{ background: c.dash }} />
            <span className="text-[10px] text-slate-400 capitalize">{cat}</span>
          </div>
        ))}
        <div className="flex items-center gap-1.5 ml-2">
          <div className="w-8 h-px border-t-2 border-dashed border-slate-500" />
          <span className="text-[10px] text-slate-400">group boundary</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-8 h-px border-t-2 border-indigo-500" style={{ borderStyle: 'dashed' }} />
          <span className="text-[10px] text-slate-400">animated = data flow</span>
        </div>
        {isFallbackArchitecture && (
          <div className="ml-auto rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-[10px] text-amber-300">
            Fallback DAG from gradient summaries
          </div>
        )}
      </div>

      <div className="flex-1 flex gap-6 min-h-0">
        {/* Canvas */}
        <div className="flex-1 card p-0 overflow-hidden relative border-slate-700/50">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            minZoom={0.08}
            maxZoom={2}
            className="bg-slate-950/50"
          >
            <Background color="#1e293b" gap={20} size={1} />
            <Controls className="!bg-slate-900 border-slate-800 !text-slate-300" />
            <MiniMap
              nodeColor={(n) => {
                if (n.type === 'core') return '#4f46e5'
                const cat = n.data?.category || 'model'
                return getCAT(cat).dash
              }}
              maskColor="rgba(15,23,42,0.8)"
              className="!bg-slate-900"
            />
          </ReactFlow>
        </div>

        {/* Inspector */}
        <div className="w-80 shrink-0 flex flex-col gap-4 overflow-y-auto custom-scrollbar">
          {selectedNode ? (
            <div className="card bg-slate-900/80 backdrop-blur-md border-indigo-500/30 shadow-[0_0_30px_rgba(99,102,241,0.1)]">
              <div className="flex items-center gap-3 mb-4 pb-4 border-b border-slate-800">
                <div className="w-9 h-9 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-base">⊞</div>
                <div>
                  <h3 className="font-mono text-sm text-white font-bold leading-tight">{selectedNode.id}</h3>
                  <p className="text-[11px] text-indigo-400 font-medium">{selectedNode.type}</p>
                </div>
              </div>
              <div className="space-y-3 text-sm">
                <div className="flex gap-2 flex-wrap">
                  <span className={`px-2 py-0.5 rounded text-[10px] border ${selectedNode.trainable !== false ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-slate-800 border-slate-700 text-slate-400'}`}>
                    {selectedNode.trainable !== false ? 'Trainable' : 'Frozen'}
                  </span>
                  <span className={`px-2 py-0.5 rounded text-[10px] ${getCAT(selectedNode.category).badge}`}>
                    {selectedNode.category}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <span className="text-slate-500 text-[10px] block mb-0.5">Output Shape</span>
                    <span className="font-mono text-slate-300 text-xs">
                      {selectedNode.output_shape
                        ? `[${Array.isArray(selectedNode.output_shape) ? selectedNode.output_shape.join(',') : selectedNode.output_shape}]`
                        : '—'}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 text-[10px] block mb-0.5">Params</span>
                    <span className="font-mono text-slate-300 text-xs">
                      {selectedNode.param_count?.toLocaleString() || '0'}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 text-[10px] block mb-0.5">Depth</span>
                    <span className="font-mono text-slate-300 text-xs">{selectedNode.depth ?? '—'}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 text-[10px] block mb-0.5">Children</span>
                    <span className="font-mono text-slate-300 text-xs">{selectedNode.children?.length ?? 0}</span>
                  </div>
                </div>
                {selectedNode.params && Object.keys(selectedNode.params).length > 0 && (
                  <div className="pt-2 border-t border-slate-800">
                    <span className="text-slate-500 text-[10px] uppercase tracking-wider block mb-2">Weight Shapes</span>
                    <div className="space-y-1">
                      {Object.entries(selectedNode.params).map(([key, val]) => (
                        <div key={key} className="flex justify-between text-[10px]">
                          <span className="text-slate-400">{key}</span>
                          <span className="font-mono text-slate-300">[{Array.isArray(val) ? val.join(',') : String(val)}]</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              <div className="mt-4 space-y-2">
                <button className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs transition">
                  {selectedNode.trainable !== false ? '❄️ Freeze Layer' : '🔥 Unfreeze Layer'}
                </button>
                <button className="w-full flex items-center justify-center gap-2 px-3 py-2 border border-slate-700 hover:bg-slate-800 text-slate-300 rounded-lg text-xs transition">
                  <Edit3 className="w-3.5 h-3.5" /> Edit Python Source
                </button>
              </div>
            </div>
          ) : (
            <div className="card h-full flex flex-col items-center justify-center text-center py-16 text-slate-500 border-dashed">
              <Network className="w-10 h-10 text-slate-700 mb-3" />
              <p className="text-sm">Click a leaf node to inspect its properties.</p>
              <p className="text-xs mt-2 text-slate-600">Click a dashed group border to collapse/expand.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
