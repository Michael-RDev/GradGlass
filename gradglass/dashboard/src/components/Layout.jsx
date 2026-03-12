import { Link, useLocation } from 'react-router-dom';
import { Microscope, LayoutDashboard, Activity, BarChart2, Cpu, Database, GitCompare, AlertTriangle, Sun, Moon, Server, Lightbulb } from 'lucide-react';
import { useTheme } from './ThemeProvider';

const RUN_NAV_ITEMS = [
  { path: '/overview', label: 'Overview', icon: LayoutDashboard },
  { path: '/training', label: 'Training', icon: Activity },
  { path: '/evaluation', label: 'Evaluation', icon: BarChart2 },
  { path: '/internals', label: 'Model Internals', icon: Cpu },
  { path: '/data', label: 'Data', icon: Database },
  { path: '/infrastructure', label: 'Infrastructure', icon: Server },
  { path: '/interpretability', label: 'Interpretability', icon: Lightbulb },
  { path: '/compare', label: 'Compare', icon: GitCompare },
  { path: '/alerts', label: 'Alerts', icon: AlertTriangle },
];

export default function Layout({ children }) {
  const location = useLocation();
  const pathParts = location.pathname.split('/');
  const isRunPage = pathParts[1] === 'run' && pathParts[2];
  const runId = isRunPage ? pathParts[2] : null;

  const { theme, toggleTheme } = useTheme();

  return (
    <div className="min-h-screen flex bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-200 font-sans transition-colors duration-200">
      {/* Sidebar Navigation */}
      <aside className="w-64 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/50 flex flex-col shrink-0 transition-colors duration-200">
        <div className="h-16 flex items-center justify-between px-6 border-b border-slate-200 dark:border-slate-800 transition-colors duration-200">
          <Link to="/" className="flex items-center gap-2 text-xl font-bold tracking-tight text-slate-900 dark:text-white hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">
            <Microscope className="w-6 h-6 text-indigo-600 dark:text-indigo-500" />
            <span>GradGlass</span>
          </Link>
          <button 
            onClick={toggleTheme} 
            className="p-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>

        <div className="flex-1 overflow-y-auto py-4">
          <div className="px-4 mb-2">
            <p className="px-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">
              {runId ? 'Run Context' : 'Global Context'}
            </p>
          </div>

          <nav className="space-y-1 px-3">
            {!isRunPage && (
              <Link
                to="/"
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                  location.pathname === '/' 
                    ? 'bg-indigo-50 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-400' 
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/50'
                }`}
              >
                <Database className="w-4 h-4" />
                All Runs
              </Link>
            )}

            {isRunPage && RUN_NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const fullPath = `/run/${runId}${item.path}`;
              const active = location.pathname.startsWith(fullPath);
              return (
                <Link
                  key={item.path}
                  to={fullPath}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    active 
                      ? 'bg-indigo-50 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-400 shadow-sm shadow-indigo-500/5' 
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>

        {runId && (
          <div className="p-4 border-t border-slate-200 dark:border-slate-800 transition-colors duration-200">
            <div className="px-3 py-2 rounded-md bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 transition-colors duration-200">
              <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Active Run</p>
              <p className="text-sm font-mono text-slate-900 dark:text-slate-200 truncate" title={decodeURIComponent(runId)}>
                {decodeURIComponent(runId)}
              </p>
            </div>
          </div>
        )}
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden bg-slate-50 dark:bg-slate-950 transition-colors duration-200">
        <header className="h-16 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/30 flex items-center px-8 shrink-0 transition-colors duration-200">
          <div className="flex-1 flex gap-2 text-sm text-slate-500 dark:text-slate-400">
            {runId && (
               <>
                 <span>Run</span>
                 <span className="text-slate-400 dark:text-slate-600">/</span>
                 <span className="text-slate-800 dark:text-slate-300 font-mono">{decodeURIComponent(runId)}</span>
                 {location.pathname.split('/').slice(3).map(part => (
                   part && (
                     <span key={part} className="flex gap-2">
                       <span className="text-slate-400 dark:text-slate-600">/</span>
                       <span className="capitalize">{part}</span>
                     </span>
                   )
                 ))}
               </>
            )}
            {!runId && <span>Dashboard</span>}
          </div>
        </header>
        <div className="flex-1 overflow-auto p-8">
          <div className="max-w-[1600px] mx-auto pb-12">
            {children}
          </div>
        </div>
      </main>
    </div>
  );
}
