import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import RunRouteController from './components/RunRouteController';
import Home from './pages/Home';
import Overview from './pages/Overview';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import ModelInternals from './pages/ModelInternals';
import Data from './pages/Data';
import Compare from './pages/Compare';
import Alerts from './pages/Alerts';
import Infrastructure from './pages/Infrastructure';
import Interpretability from './pages/Interpretability';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Home />} />
        <Route path="/experiments" element={<Home />} />
        <Route path="/models" element={<Home />} />
        <Route path="/datasets" element={<Home />} />
        <Route path="/run/:runId" element={<RunRouteController />}>
          <Route index element={<Navigate to="overview" replace />} />
          <Route path="overview" element={<Overview />} />
          <Route path="training" element={<Training />} />
          <Route path="evaluation" element={<Evaluation />} />
          <Route path="internals" element={<ModelInternals />} />
          <Route path="data" element={<Data />} />
          <Route path="leakage" element={<Data routeMode="leakage" />} />
          <Route path="infrastructure" element={<Infrastructure />} />
          <Route path="interpretability" element={<Interpretability />} />
          <Route path="compare" element={<Compare />} />
          <Route path="alerts" element={<Alerts />} />
        </Route>
      </Routes>
    </Layout>
  );
}
