import React from 'react';
import { X, BarChart3 } from 'lucide-react';

export default function EvaluateModal({
  properties,
  treeData,
  dualseqTuples,
  onClose,
}) {
  const bbox = properties?.bbox || { min: [0, 0, 0], max: [0, 0, 0] };
  const dx = (bbox.max[0] - bbox.min[0]).toFixed(3);
  const dy = (bbox.max[1] - bbox.min[1]).toFixed(3);
  const dz = (bbox.max[2] - bbox.min[2]).toFixed(3);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" style={{ width: 480 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <BarChart3 size={16} color="#2563eb" />
            <span>Mass Properties & Evaluation</span>
          </div>
          <button className="header-icon-btn" onClick={onClose}><X size={16} /></button>
        </div>

        <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ background: '#f8fafc', padding: 12, borderRadius: 6, border: '1px solid #e2e8f0' }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#0f172a', marginBottom: 8 }}>
              Geometric Properties (OpenCASCADE)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 12 }}>
              <div>
                <span style={{ color: '#64748b' }}>Volume: </span>
                <b>{properties?.volume?.toFixed(4) || '0.0000'}</b>
              </div>
              <div>
                <span style={{ color: '#64748b' }}>Solid Parts: </span>
                <b>{treeData?.part_count || 0}</b>
              </div>
              <div style={{ gridColumn: 'span 2' }}>
                <span style={{ color: '#64748b' }}>Bounding Box (L × W × H): </span>
                <b>{dx} × {dy} × {dz}</b>
              </div>
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn-primary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
