import React, { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import useRunStore from '../store/useRunStore';
import { AlertTriangle, Info, Bell, ShieldCheck, Flame } from 'lucide-react';

export default function Alerts() {
  const { runId } = useParams();
  const { setActiveRun, alerts, metadata } = useRunStore();

  useEffect(() => {
    if (runId) setActiveRun(runId);
  }, [runId, setActiveRun]);

  if (!metadata) return <div className="p-8 text-slate-400">Loading alerts data...</div>;

  const getAlertIcon = (severity) => {
    if (severity === 'high') return <Flame className="w-5 h-5 text-red-500" />;
    if (severity === 'medium') return <AlertTriangle className="w-5 h-5 text-amber-500" />;
    return <Info className="w-5 h-5 text-blue-500" />;
  };

  const getAlertColors = (severity) => {
    if (severity === 'high') return { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400' };
    if (severity === 'medium') return { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-500' };
    return { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400' };
  };

  const highCount = alerts.filter(a => a.severity === 'high').length;
  const medCount = alerts.filter(a => a.severity === 'medium').length;

  return (
    <div className="space-y-6 max-w-[1000px] mx-auto">
      <div className="flex items-center justify-between">
        <div>
           <h1 className="text-2xl font-bold text-white tracking-tight">System Alerts</h1>
           <p className="text-sm text-slate-400 mt-1">Automatic anomaly detections for training run {decodeURIComponent(runId)}</p>
        </div>
        
        <div className="flex items-center gap-4">
           <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border bg-slate-900 ${alerts.length === 0 ? 'border-emerald-500/30' : 'border-slate-800'}`}>
             <Bell className={`w-4 h-4 ${alerts.length === 0 ? 'text-emerald-500' : 'text-slate-400'}`} />
             <span className="text-sm font-bold text-white">{alerts.length} Total</span>
           </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
         <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
            <div>
               <p className="text-sm font-medium text-slate-400">High Severity</p>
               <p className="text-3xl font-bold text-red-500 mt-2">{highCount}</p>
            </div>
            <div className="p-3 bg-red-500/10 rounded-full">
               <Flame className="w-6 h-6 text-red-500" />
            </div>
         </div>
         
         <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
            <div>
               <p className="text-sm font-medium text-slate-400">Warnings</p>
               <p className="text-3xl font-bold text-amber-500 mt-2">{medCount}</p>
            </div>
            <div className="p-3 bg-amber-500/10 rounded-full">
               <AlertTriangle className="w-6 h-6 text-amber-500" />
            </div>
         </div>
         
         <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
            <div>
               <p className="text-sm font-medium text-slate-400">System Health</p>
               <p className="text-xl font-bold text-white mt-2 capitalize">{metadata.status || 'Active'}</p>
            </div>
            <div className="p-3 bg-emerald-500/10 rounded-full">
               <ShieldCheck className="w-6 h-6 text-emerald-500" />
            </div>
         </div>
      </div>

      <div className="space-y-4">
         {alerts.length === 0 ? (
           <div className="flex flex-col items-center justify-center p-12 bg-slate-900 border border-slate-800 rounded-xl">
             <div className="w-16 h-16 bg-emerald-500/10 flex items-center justify-center rounded-full mb-4">
                <ShieldCheck className="w-8 h-8 text-emerald-500" />
             </div>
             <h3 className="text-lg font-bold text-white">All Clear</h3>
             <p className="text-slate-400 mt-2">No anomalies or significant warnings found for this run.</p>
           </div>
         ) : (
           alerts.map((a, i) => {
             const colors = getAlertColors(a.severity);
             return (
               <div key={i} className={`p-5 rounded-xl border flex items-start gap-4 transition-colors ${colors.bg} ${colors.border}`}>
                  <div className="shrink-0 mt-0.5">
                     {getAlertIcon(a.severity)}
                  </div>
                  <div className="flex-1 min-w-0">
                     <div className="flex items-center justify-between gap-4">
                        <h4 className={`text-base font-semibold ${colors.text}`}>{a.title}</h4>
                        <span className={`text-xs px-2 py-0.5 rounded-full uppercase font-bold tracking-wider border ${colors.border} ${colors.text}`}>
                           {a.severity}
                        </span>
                     </div>
                     <p className={`text-sm mt-1 mb-2 opacity-90 ${colors.text}`}>{a.message}</p>
                     
                     {a.severity === 'high' && (
                       <div className="mt-3 bg-slate-950 p-3 rounded text-xs text-red-300 font-mono border border-red-500/20">
                         <strong>Recommendation:</strong> Consider stopping the run or reverting to the last known good checkpoint. Gradients or loss may be diverging.
                       </div>
                     )}
                  </div>
               </div>
             );
         })
         )}
      </div>
    </div>
  );
}
