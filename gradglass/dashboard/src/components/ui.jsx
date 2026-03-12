export function SeverityBadge({ severity }) {
  const classes = {
    LOW: 'severity-low',
    MEDIUM: 'severity-medium',
    HIGH: 'severity-high',
    CRITICAL: 'severity-critical',
  }
  return (
    <span className={`badge ${classes[severity] || 'text-slate-400'}`}>
      {severity}
    </span>
  )
}

export function StatusBadge({ status }) {
  const classes = {
    running: 'text-blue-400 bg-blue-400/10 border-blue-400/30',
    complete: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
    failed: 'text-red-400 bg-red-400/10 border-red-400/30',
  }
  return (
    <span className={`badge ${classes[status] || 'text-slate-400 bg-slate-400/10 border-slate-400/30'}`}>
      {status === 'running' && <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse mr-1.5" />}
      {status}
    </span>
  )
}

export function MetricValue({ label, value, unit = '', className = '' }) {
  return (
    <div className={`flex flex-col ${className}`}>
      <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
      <span className="text-lg font-semibold font-mono">
        {value}{unit && <span className="text-xs text-slate-500 ml-1">{unit}</span>}
      </span>
    </div>
  )
}

export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="w-8 h-8 border-2 border-glass-500 border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

export function ErrorMessage({ message }) {
  return (
    <div className="card border-red-500/30 text-red-400 text-center py-8">
      <p className="text-lg font-medium">Error</p>
      <p className="text-sm mt-1 text-red-400/70">{message}</p>
    </div>
  )
}

export function EmptyState({ icon: Icon, title, description }) {
  return (
    <div className="card text-center py-16">
      {Icon && <Icon className="w-12 h-12 text-slate-600 mx-auto mb-4" />}
      <h3 className="text-lg font-medium text-slate-400">{title}</h3>
      {description && <p className="text-sm text-slate-500 mt-2">{description}</p>}
    </div>
  )
}
