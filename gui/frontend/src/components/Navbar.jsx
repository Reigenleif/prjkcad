import React, { useState, useEffect } from 'react';
import { Box, Save, Download, FileText, RefreshCw, Cpu, Layers, HardDrive } from 'lucide-react';
import { getCuratedSamples, exportCAD } from '../services/api';

export default function Navbar({
  systemHealth,
  dualseqTuples,
  onSelectSample,
  onNewModel,
  onOpenEditor,
}) {
  const [samples, setSamples] = useState([]);
  const [selectedSampleUid, setSelectedSampleUid] = useState('');
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    getCuratedSamples()
      .then((data) => {
        if (data && data.samples) {
          setSamples(data.samples);
          if (data.samples.length > 0) {
            setSelectedSampleUid(data.samples[0].uid);
          }
        }
      })
      .catch((err) => console.error('Failed to load samples in navbar:', err));
  }, []);

  const handleSampleChange = (e) => {
    const uid = e.target.value;
    setSelectedSampleUid(uid);
    if (onSelectSample) {
      onSelectSample(uid);
    }
  };

  const handleExportSTL = async () => {
    if (!dualseqTuples || dualseqTuples.length === 0) return;
    setExporting(true);
    try {
      await exportCAD(dualseqTuples, 'stl', 'text2cad_model');
    } catch (err) {
      alert(`Export STL failed: ${err.message}`);
    } finally {
      setExporting(false);
    }
  };

  const handleExportJSON = async () => {
    if (!dualseqTuples || dualseqTuples.length === 0) return;
    setExporting(true);
    try {
      await exportCAD(dualseqTuples, 'json', 'dualseq_sequence');
    } catch (err) {
      alert(`Export JSON failed: ${err.message}`);
    } finally {
      setExporting(false);
    }
  };

  return (
    <header className="navbar">
      <div className="navbar-brand">
        <Box size={18} color="#3b82f6" />
        <span style={{ fontWeight: 700, letterSpacing: '0.5px' }}>MINI CAD</span>
      </div>

      <div className="navbar-menu">
        <button className="nav-item-btn" onClick={onNewModel} title="Create New Empty CAD Model">
          <FileText size={13} />
          <span>New</span>
        </button>

        <button className="nav-item-btn" onClick={onOpenEditor} title="Open Sequence Code Editor">
          <Layers size={13} />
          <span>Edit Sequence</span>
        </button>

        <button className="nav-item-btn" onClick={handleExportSTL} disabled={exporting} title="Export Binary STL for 3D Printing / CAD">
          <Download size={13} />
          <span>Export STL</span>
        </button>

        <button className="nav-item-btn" onClick={handleExportJSON} disabled={exporting} title="Export DualSeq JSON format">
          <Save size={13} />
          <span>Save JSON</span>
        </button>

        {samples.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', marginLeft: 12, gap: 6 }}>
            <span style={{ fontSize: 11, color: '#94a3b8' }}>Sample:</span>
            <select
              value={selectedSampleUid}
              onChange={handleSampleChange}
              style={{
                background: '#1e293b',
                color: '#f8fafc',
                border: '1px solid #334155',
                borderRadius: 4,
                padding: '3px 8px',
                fontSize: 11.5,
                maxWidth: 220,
                outline: 'none',
              }}
            >
              {samples.map((s) => (
                <option key={s.uid} value={s.uid}>
                  {s.label || s.uid}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div className="navbar-actions">
        {systemHealth && (
          <div className="status-badge" title="Connection status">
            <div className={`status-indicator ${systemHealth?.status === 'offline' ? 'offline' : ''}`} />
            <span style={{ fontSize: 11 }}>
              {systemHealth?.status === 'offline' ? 'Offline' : 'Connected'}
            </span>
          </div>
        )}
      </div>
    </header>
  );
}
