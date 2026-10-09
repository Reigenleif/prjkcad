import React, { useState } from 'react';
import { Box, Save, Download, FileText, Layers } from 'lucide-react';
import { exportCAD } from '../services/api';

export default function Navbar({
  systemHealth,
  dualseqTuples,
  onNewModel,
  onOpenEditor,
}) {
  const [exporting, setExporting] = useState(false);

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
