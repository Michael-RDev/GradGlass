import { Routes, Route, Navigate, useParams } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import ModelInternals from './pages/ModelInternals';
import Data from './pages/Data';
import Compare from './pages/Compare';
import Alerts from './pages/Alerts';
import Infrastructure from './pages/Infrastructure';
import Interpretability from './pages/Interpretability';

function RunInternalsRedirect() {
  const { runId } = useParams();
  return <Navigate to={`/run/${encodeURIComponent(runId || '')}/overview`} replace />;
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Home />} />
        <Route path="/experiments" element={<Home />} />
        <Route path="/models" element={<Home />} />
        <Route path="/datasets" element={<Home />} />
        <Route path="/run/:runId" element={<Navigate to="overview" replace />} />
        <Route path="/run/:runId/overview" element={<ModelInternals />} />
        <Route path="/run/:runId/training" element={<Training />} />
        <Route path="/run/:runId/evaluation" element={<Evaluation />} />
        <Route path="/run/:runId/internals" element={<RunInternalsRedirect />} />
        <Route path="/run/:runId/data" element={<Data />} />
        <Route path="/run/:runId/infrastructure" element={<Infrastructure />} />
        <Route path="/run/:runId/interpretability" element={<Interpretability />} />
        <Route path="/run/:runId/compare" element={<Compare />} />
        <Route path="/run/:runId/alerts" element={<Alerts />} />
      </Routes>
    </Layout>
  );
}
