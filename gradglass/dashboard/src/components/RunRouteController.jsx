import { useEffect, useRef } from 'react';
import { Outlet, useLocation, useNavigate, useParams } from 'react-router-dom';
import useRunStore from '../store/useRunStore';

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled', 'interrupted']);
const RUN_STATUS_ALIASES = {
  complete: 'completed',
  finished: 'completed',
  error: 'failed',
  canceled: 'cancelled',
  stopped: 'cancelled',
  terminated: 'interrupted',
};

function normalizeRunStatus(status) {
  const normalized = (status || '').toString().trim().toLowerCase();
  return RUN_STATUS_ALIASES[normalized] || normalized;
}

function buildOverviewPath(runId) {
  return `/run/${encodeURIComponent(runId || '')}/overview`;
}

export default function RunRouteController() {
  const { runId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const { setActiveRun, clearActiveRun, metadata, overview } = useRunStore();
  const previousStatusRef = useRef(null);
  const sawLiveStatusRef = useRef(false);
  const redirectedToOverviewRef = useRef(false);

  useEffect(() => {
    if (!runId) {
      return undefined;
    }

    previousStatusRef.current = null;
    sawLiveStatusRef.current = false;
    redirectedToOverviewRef.current = false;
    setActiveRun(runId);

    return () => {
      clearActiveRun();
    };
  }, [clearActiveRun, runId, setActiveRun]);

  const currentStatus = normalizeRunStatus(overview?.status || metadata?.status);
  const overviewPath = buildOverviewPath(runId || '');

  useEffect(() => {
    if (!currentStatus) {
      return;
    }

    const previousStatus = previousStatusRef.current;
    if (!TERMINAL_STATUSES.has(currentStatus)) {
      sawLiveStatusRef.current = true;
    }

    const transitionedToTerminal =
      Boolean(previousStatus) &&
      previousStatus !== currentStatus &&
      !TERMINAL_STATUSES.has(previousStatus) &&
      TERMINAL_STATUSES.has(currentStatus);

    if (
      transitionedToTerminal &&
      sawLiveStatusRef.current &&
      !redirectedToOverviewRef.current &&
      location.pathname !== overviewPath
    ) {
      redirectedToOverviewRef.current = true;
      navigate(overviewPath, { replace: true });
    }

    previousStatusRef.current = currentStatus;
  }, [currentStatus, location.pathname, navigate, overviewPath]);

  return <Outlet />;
}
