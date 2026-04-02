import React from 'react';
import { Link, useParams } from 'react-router-dom';
import useRunStore from '../store/useRunStore';
import { SeverityBadge, StatusBadge } from '../components/ui';
import {
  AlertTriangle,
  ArrowRight,
  Bell,
  CheckCircle2,
  Flame,
  Info,
  ShieldAlert,
  ShieldCheck,
  Wrench,
} from 'lucide-react';

const SEVERITY_STYLES = {
  CRITICAL: {
    icon: Flame,
    card: 'bg-red-500/10 border-red-500/30',
    iconWrap: 'bg-red-500/15',
    iconClass: 'text-red-400',
    title: 'text-red-300',
    body: 'text-red-100/85',
  },
  HIGH: {
    icon: AlertTriangle,
    card: 'bg-orange-500/10 border-orange-500/30',
    iconWrap: 'bg-orange-500/15',
    iconClass: 'text-orange-300',
    title: 'text-orange-200',
    body: 'text-orange-100/85',
  },
  MEDIUM: {
    icon: ShieldAlert,
    card: 'bg-amber-500/10 border-amber-500/30',
    iconWrap: 'bg-amber-500/15',
    iconClass: 'text-amber-300',
    title: 'text-amber-100',
    body: 'text-amber-100/80',
  },
  LOW: {
    icon: Info,
    card: 'bg-blue-500/10 border-blue-500/30',
    iconWrap: 'bg-blue-500/15',
    iconClass: 'text-blue-300',
    title: 'text-blue-100',
    body: 'text-blue-100/80',
  },
};

const HEALTH_STYLES = {
  HEALTHY: { icon: ShieldCheck, accent: 'text-emerald-400', border: 'border-emerald-500/30', bg: 'bg-emerald-500/10' },
  WARNING: { icon: ShieldAlert, accent: 'text-amber-400', border: 'border-amber-500/30', bg: 'bg-amber-500/10' },
  STALLED: { icon: AlertTriangle, accent: 'text-orange-400', border: 'border-orange-500/30', bg: 'bg-orange-500/10' },
  FAILED: { icon: Flame, accent: 'text-red-400', border: 'border-red-500/30', bg: 'bg-red-500/10' },
};

const STATUS_STYLES = {
  fail: 'text-red-300 border-red-500/30 bg-red-500/10',
  warn: 'text-amber-200 border-amber-500/30 bg-amber-500/10',
  info: 'text-blue-200 border-blue-500/30 bg-blue-500/10',
};

const CTA_LABELS = {
  '/training': 'Open Metrics',
  '/evaluation': 'Open Evaluation',
  '/infrastructure': 'Open Infrastructure',
  '/overview': 'Open Overview',
  '/data': 'Open Data',
};

function normalizeSeverity(severity) {
  return (severity || 'LOW').toString().trim().toUpperCase();
}

function normalizeStatus(status) {
  return (status || 'warn').toString().trim().toLowerCase();
}

function severityRank(severity) {
  const level = normalizeSeverity(severity);
  return { CRITICAL: 3, HIGH: 2, MEDIUM: 1, LOW: 0 }[level] || 0;
}

function statusRank(status) {
  const level = normalizeStatus(status);
  return { fail: 0, warn: 1, info: 2 }[level] ?? 9;
}

function buildFallbackSummary(alerts, overview) {
  const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
  let failCount = 0;
  let warnCount = 0;

  alerts.forEach((alert) => {
    const severity = normalizeSeverity(alert.severity);
    counts[severity] = (counts[severity] || 0) + 1;
    if (normalizeStatus(alert.status) === 'fail') failCount += 1;
    if (normalizeStatus(alert.status) === 'warn') warnCount += 1;
  });

  return {
    total: alerts.length,
    critical: counts.CRITICAL,
    high: counts.HIGH,
    medium: counts.MEDIUM,
    low: counts.LOW,
    high_severity: counts.CRITICAL + counts.HIGH,
    warnings: counts.MEDIUM + counts.LOW,
    fail_count: failCount,
    warn_count: warnCount,
    health_state: overview?.health_state || 'WARNING',
    health_reason: overview?.status_reason || overview?.eta_reason || null,
    top_alert_id: alerts[0]?.id || null,
  };
}

function getVerdict(summary, primaryAlert) {
  const healthState = summary?.health_state || 'WARNING';
  if (!primaryAlert) {
    return {
      title: 'All Clear',
      body: 'No active anomaly detections were found for this run.',
      recommendation: 'Keep logging metrics, gradients, and evaluation probes so GradGlass can continue checking for regressions.',
    };
  }

  if (summary?.high_severity > 0 || healthState === 'FAILED' || healthState === 'STALLED') {
    return {
      title: primaryAlert.title,
      body: primaryAlert.message,
      recommendation: primaryAlert.recommendation,
    };
  }

  return {
    title: 'Warnings Detected',
    body: primaryAlert.message,
    recommendation: primaryAlert.recommendation,
  };
}

export default function Alerts() {
  const { runId } = useParams();
  const { alerts, alertsSummary, metadata, overview } = useRunStore();

  if (!metadata || !overview) {
    return <div className="p-8 text-slate-400">Loading alerts data...</div>;
  }

  const sortedAlerts = [...alerts].sort((a, b) => {
    const severityDelta = severityRank(b.severity) - severityRank(a.severity);
    if (severityDelta !== 0) return severityDelta;
    const statusDelta = statusRank(a.status) - statusRank(b.status);
    if (statusDelta !== 0) return statusDelta;
    return (a.title || '').localeCompare(b.title || '');
  });

  const summary = alertsSummary || buildFallbackSummary(sortedAlerts, overview);
  const healthState = summary.health_state || overview.health_state || 'WARNING';
  const healthStyle = HEALTH_STYLES[healthState] || HEALTH_STYLES.WARNING;
  const HealthIcon = healthStyle.icon;
  const primaryAlert = sortedAlerts[0] || null;
  const verdict = getVerdict(summary, primaryAlert);
  const decodedRunId = decodeURIComponent(runId || metadata.run_id || '');

  return (
    <div className="space-y-6 max-w-[1100px] mx-auto">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-bold text-white tracking-tight">System Alerts</h1>
            <StatusBadge status={overview.status || metadata.status} />
          </div>
          <p className="text-sm text-slate-400 mt-2">
            Automatic anomaly detections for training run {decodedRunId}
          </p>
          <p className="text-sm text-slate-500 mt-1">
            {metadata.name ? `Run name: ${metadata.name}` : 'GradGlass combines live health signals and saved analysis warnings here.'}
          </p>
        </div>

        <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border bg-slate-900 ${healthStyle.border}`}>
          <div className={`p-2 rounded-lg ${healthStyle.bg}`}>
            <HealthIcon className={`w-5 h-5 ${healthStyle.accent}`} />
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">System Health</p>
            <p className={`text-sm font-semibold ${healthStyle.accent}`}>{healthState}</p>
          </div>
        </div>
      </div>

      <div className={`rounded-2xl border bg-slate-900 p-6 ${healthStyle.border}`}>
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="flex items-start gap-4">
            <div className={`p-3 rounded-xl ${healthStyle.bg}`}>
              <HealthIcon className={`w-6 h-6 ${healthStyle.accent}`} />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">{verdict.title}</h2>
              <p className="text-sm text-slate-300 mt-2 max-w-2xl">{verdict.body}</p>
              {summary.health_reason && (
                <p className="text-xs text-slate-500 mt-2">Health note: {summary.health_reason}</p>
              )}
            </div>
          </div>
          {primaryAlert?.cta_path && (
            <Link
              to={`/run/${encodeURIComponent(runId || '')}${primaryAlert.cta_path}`}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-700 bg-slate-950 text-sm text-slate-200 hover:border-slate-600 hover:bg-slate-800 transition-colors"
            >
              {CTA_LABELS[primaryAlert.cta_path] || 'Open Related View'}
              <ArrowRight className="w-4 h-4" />
            </Link>
          )}
        </div>

        {verdict.recommendation && (
          <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/80 p-4">
            <div className="flex items-start gap-3">
              <Wrench className="w-5 h-5 text-sky-300 shrink-0 mt-0.5" />
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Recommended Fix</p>
                <p className="text-sm text-slate-200 mt-1">{verdict.recommendation}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-slate-400">Total</p>
            <p className="text-3xl font-bold text-white mt-2">{summary.total || 0}</p>
          </div>
          <div className="p-3 bg-slate-800 rounded-full">
            <Bell className="w-6 h-6 text-slate-300" />
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-slate-400">High Severity</p>
            <p className="text-3xl font-bold text-red-400 mt-2">{summary.high_severity || 0}</p>
          </div>
          <div className="p-3 bg-red-500/10 rounded-full">
            <Flame className="w-6 h-6 text-red-400" />
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-slate-400">Warnings</p>
            <p className="text-3xl font-bold text-amber-300 mt-2">{summary.warnings || 0}</p>
          </div>
          <div className="p-3 bg-amber-500/10 rounded-full">
            <ShieldAlert className="w-6 h-6 text-amber-300" />
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-slate-400">System Health</p>
            <p className={`text-xl font-bold mt-2 ${healthStyle.accent}`}>{healthState}</p>
          </div>
          <div className={`p-3 rounded-full ${healthStyle.bg}`}>
            <HealthIcon className={`w-6 h-6 ${healthStyle.accent}`} />
          </div>
        </div>
      </div>

      {sortedAlerts.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-12 bg-slate-900 border border-emerald-500/20 rounded-2xl">
          <div className="w-16 h-16 bg-emerald-500/10 flex items-center justify-center rounded-full mb-4">
            <CheckCircle2 className="w-8 h-8 text-emerald-400" />
          </div>
          <h3 className="text-lg font-bold text-white">All Clear</h3>
          <p className="text-slate-400 mt-2 text-center max-w-xl">
            No anomalies or significant warnings are active for this run right now.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {sortedAlerts.map((alert) => {
            const severity = normalizeSeverity(alert.severity);
            const status = normalizeStatus(alert.status);
            const style = SEVERITY_STYLES[severity] || SEVERITY_STYLES.LOW;
            const Icon = style.icon;
            const statusStyle = STATUS_STYLES[status] || STATUS_STYLES.warn;

            return (
              <div key={alert.id} className={`rounded-2xl border p-5 ${style.card}`}>
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="flex items-start gap-4 min-w-0">
                    <div className={`p-3 rounded-xl ${style.iconWrap}`}>
                      <Icon className={`w-5 h-5 ${style.iconClass}`} />
                    </div>
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className={`text-lg font-semibold ${style.title}`}>{alert.title}</h3>
                        <SeverityBadge severity={severity} />
                        <span className={`text-[10px] uppercase tracking-[0.2em] border rounded-full px-2 py-1 ${statusStyle}`}>
                          {status}
                        </span>
                        <span className="text-[10px] uppercase tracking-[0.2em] border border-slate-700 text-slate-400 rounded-full px-2 py-1">
                          {alert.category}
                        </span>
                        <span className="text-[10px] uppercase tracking-[0.2em] border border-slate-700 text-slate-500 rounded-full px-2 py-1">
                          {alert.source}
                        </span>
                      </div>

                      <p className={`text-sm mt-3 leading-relaxed ${style.body}`}>
                        {alert.message || alert.recommendation}
                      </p>

                      {Array.isArray(alert.evidence) && alert.evidence.length > 0 && (
                        <div className="mt-4">
                          <p className="text-xs uppercase tracking-[0.18em] text-slate-500 mb-2">Evidence</p>
                          <div className="flex flex-wrap gap-2">
                            {alert.evidence.map((line, index) => (
                              <span
                                key={`${alert.id}-evidence-${index}`}
                                className="text-xs px-2.5 py-1.5 rounded-full border border-slate-700 bg-slate-950/70 text-slate-300"
                              >
                                {line}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 shrink-0">
                    {alert.step != null && (
                      <span className="text-xs px-2.5 py-1.5 rounded-full border border-slate-700 text-slate-300 bg-slate-950/70">
                        Step {alert.step}
                      </span>
                    )}
                    {alert.cta_path && (
                      <Link
                        to={`/run/${encodeURIComponent(runId || '')}${alert.cta_path}`}
                        className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-700 bg-slate-950 text-sm text-slate-200 hover:border-slate-600 hover:bg-slate-800 transition-colors"
                      >
                        {CTA_LABELS[alert.cta_path] || 'Open Related View'}
                        <ArrowRight className="w-4 h-4" />
                      </Link>
                    )}
                  </div>
                </div>

                {alert.recommendation && (
                  <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/80 p-4">
                    <div className="flex items-start gap-3">
                      <Wrench className="w-4 h-4 text-sky-300 shrink-0 mt-0.5" />
                      <div>
                        <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Recommended Fix</p>
                        <p className="text-sm text-slate-200 mt-1">{alert.recommendation}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
