import { Link, useLocation } from 'react-router-dom';
import { Microscope, LayoutDashboard, Activity, BarChart2, Cpu, Database, GitCompare, AlertTriangle, Sun, Moon, Server, Lightbulb, User, Bell } from 'lucide-react';
import { useTheme } from './ThemeProvider';

const MAIN_NAV = [
  { globalLabel: 'Dashboard', runLabel: 'Overview', runPath: '/overview', globalPath: '/dashboard' },
  { globalLabel: 'Experiments', runLabel: 'Experiments', runPath: '/compare', globalPath: '/experiments' },
  { globalLabel: 'Models', runLabel: 'Visualizations', runPath: '/internals', globalPath: '/models' },
  { globalLabel: 'Datasets', runLabel: 'Datasets', runPath: '/data', globalPath: '/datasets' },
];

const SIDEBAR_SECTIONS = [
  {
    title: 'Monitor',
    items: [
      { path: '/overview', label: 'Overview', icon: LayoutDashboard },
      { path: '/training', label: 'Metrics', icon: Activity },
      { path: '/infrastructure', label: 'Infrastructure', icon: Server },
    ]
  },
  {
    title: 'Analyze',
    items: [
      { path: '/evaluation', label: 'Evaluation', icon: BarChart2 },
      { path: '/compare', label: 'Compare', icon: GitCompare },
      { path: '/alerts', label: 'Alerts', icon: AlertTriangle },
    ]
  },
  {
    title: 'Deep Dive',
    items: [
      { path: '/internals', label: 'Visualizations', icon: Cpu },
      { path: '/data', label: 'Data', icon: Database },
      { path: '/interpretability', label: 'Interpretability', icon: Lightbulb },
    ]
  }
];

export default function Layout({ children }) {
  const location = useLocation();
  const pathParts = location.pathname.split('/');
  const isRunPage = pathParts[1] === 'run' && pathParts[2];
  const runId = isRunPage ? pathParts[2] : null;

  const { theme, toggleTheme } = useTheme();

  return (
    <div className="flex flex-col h-screen h-[100dvh] overflow-hidden bg-theme-bg text-theme-text-primary transition-colors duration-250 ease-in-out">
      
      {/* Top Navigation */}
      <header className="h-[64px] border-b border-theme-border bg-theme-surface flex items-center justify-between px-6 shrink-0 z-20">
        
        {/* Left: Logo */}
        <div className="flex items-center gap-8">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-theme-primary flex items-center justify-center shadow-lg">
               <Microscope className="w-5 h-5 text-[#181A2F]" />
            </div>
            <span className="text-xl font-bold tracking-tight text-theme-text-primary hidden sm:block">GradGlass</span>
          </Link>

          {/* Center: Main Nav */}
          <nav className="hidden md:flex items-center gap-1">
            {MAIN_NAV.map((nav) => {
              const target = runId ? `/run/${runId}${nav.runPath}` : nav.globalPath;
              const label = runId ? nav.runLabel : nav.globalLabel;
              const isActive = runId
                ? location.pathname.includes(nav.runPath)
                : location.pathname === nav.globalPath;

              return (
                <Link
                  key={nav.globalLabel}
                  to={target}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'text-theme-primary bg-theme-primary/10'
                      : 'text-theme-text-secondary hover:text-theme-text-primary hover:bg-theme-surface-hover'
                  }`}
                >
                  {label}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-3">
          <button 
            onClick={toggleTheme} 
            className="p-2 rounded-full bg-theme-surface-hover text-theme-text-secondary hover:text-theme-primary transition-all duration-250 hover:-translate-y-[2px]"
          >
            {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
          
          <button className="p-2 rounded-full bg-theme-surface-hover text-theme-text-secondary hover:text-theme-primary transition-all duration-250 hover:-translate-y-[2px]">
             <Bell className="w-5 h-5" />
          </button>

          <button className="w-9 h-9 rounded-full bg-theme-accent border border-theme-border flex items-center justify-center text-white font-medium hover:ring-2 hover:ring-theme-primary transition-all cursor-pointer">
             <User className="w-5 h-5" />
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-[260px] border-r border-theme-border bg-theme-bg dark:bg-[#181A2F] flex flex-col shrink-0 z-10 transition-colors duration-250">
          <div className="flex-1 overflow-y-auto py-6 px-4 custom-scrollbar">
            
            {runId ? (
              <div className="mb-6 p-4 rounded-xl bg-theme-surface border border-theme-border shadow-sm">
                <p className="text-[12px] font-semibold text-theme-text-muted uppercase tracking-wider mb-1">Active Run</p>
                <p className="text-sm font-mono text-theme-text-primary truncate" title={decodeURIComponent(runId)}>
                  {decodeURIComponent(runId)}
                </p>
              </div>
            ) : (
              <div className="mb-6 px-2">
                 <p className="text-[12px] font-semibold text-theme-text-muted uppercase tracking-wider">Global Dashboard</p>
              </div>
            )}

            <div className="space-y-8">
              {SIDEBAR_SECTIONS.map((section, sIdx) => (
                 <div key={sIdx}>
                   <h4 className="px-3 text-[12px] font-semibold text-theme-text-muted uppercase tracking-wider mb-2">
                     {section.title}
                   </h4>
                   <nav className="space-y-1">
                     {section.items.map((item) => {
                        const Icon = item.icon;
                        const fullPath = runId ? `/run/${runId}${item.path}` : '/';
                        // Keep simple active state matching for now
                        const active = location.pathname.includes(item.path);
                        
                        return (
                          <Link
                            key={item.path}
                            to={fullPath}
                            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-[14px] font-medium transition-all duration-250 ${
                              active 
                                ? 'bg-theme-surface border border-theme-border text-theme-primary shadow-sm' 
                                : 'text-theme-text-secondary hover:text-theme-text-primary hover:bg-theme-surface-hover'
                            }`}
                          >
                            <Icon className={`w-4 h-4 ${active ? 'text-theme-primary' : 'text-theme-text-muted'}`} />
                            {item.label}
                          </Link>
                        );
                     })}
                   </nav>
                 </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 flex flex-col h-full overflow-hidden bg-theme-bg relative">
          {/* subtle background gradient overlay */}
          <div className="absolute inset-0 bg-theme-bg opacity-50 pointer-events-none" />
          
          <div className="flex-1 overflow-auto p-8 relative z-10 custom-scrollbar">
            <div className="max-w-[1600px] mx-auto pb-12">
              {children}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
