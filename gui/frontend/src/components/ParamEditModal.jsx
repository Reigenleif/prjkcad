import React, { useState } from 'react';
import { X, Check } from 'lucide-react';

const PARAM_DESCRIPTIONS = {
  coor_tx: 'Translation X (X-axis offset)',
  coor_ty: 'Translation Y (Y-axis offset)',
  coor_tz: 'Translation Z (Z-axis offset)',
  coor_euax: 'Euler Rotation X (degrees)',
  coor_euay: 'Euler Rotation Y (degrees)',
  coor_euaz: 'Euler Rotation Z (degrees)',
  line_sx: 'Start X coordinate',
  line_sy: 'Start Y coordinate',
  line_ex: 'End X coordinate',
  line_ey: 'End Y coordinate',
  circle_cx: 'Center X coordinate',
  circle_cy: 'Center Y coordinate',
  circle_r: 'Radius',
  arc_sx: 'Start X coordinate',
  arc_sy: 'Start Y coordinate',
  arc_mx: 'Midpoint X coordinate',
  arc_my: 'Midpoint Y coordinate',
  arc_ex: 'End X coordinate',
  arc_ey: 'End Y coordinate',
  extrude_new_dtn: 'Extrude Depth (towards normal)',
  extrude_new_don: 'Reverse Depth (opposite normal)',
  extrude_new_scale: 'Sketch Scale factor (1.0)',
  extrude_join_dtn: 'Extrude Depth (towards normal)',
  extrude_join_don: 'Reverse Depth (opposite normal)',
  extrude_join_scale: 'Sketch Scale factor',
  extrude_cut_dtn: 'Cut Depth (towards normal)',
  extrude_cut_don: 'Reverse Cut Depth',
  extrude_cut_scale: 'Sketch Scale factor',
  extrude_thread_dtn: 'Threaded Depth (towards normal)',
  extrude_thread_don: 'Reverse Depth',
  extrude_thread_scale: 'Sketch Scale factor',
  extrude_thread_pitch: 'Thread Pitch (mm/turn)',
  extrude_thread_depth: 'Thread Tooth Depth',
};

export default function ParamEditModal({
  node,
  onSave,
  onClose,
}) {
  if (!node) return null;

  const [params, setParams] = useState({ ...(node.params || {}) });

  const handleChange = (key, val) => {
    setParams((prev) => ({
      ...prev,
      [key]: val === '' ? 0.0 : parseFloat(val),
    }));
  };

  const handleSave = () => {
    onSave(node.cmd_index, params);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" style={{ width: 460 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span>Edit Parameters: {node.name}</span>
          <button className="header-icon-btn" onClick={onClose}><X size={16} /></button>
        </div>

        <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 12, color: '#64748b' }}>
            Command: <b>{node.cmd}</b> (Feature Index: {node.cmd_index})
          </div>

          {Object.keys(params).length === 0 ? (
            <div style={{ fontSize: 12, color: '#94a3b8', padding: '12px 0' }}>
              This command does not require any numeric parameters.
            </div>
          ) : (
            Object.entries(params).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <label style={{ fontSize: 11.5, fontWeight: 500, color: '#334155', display: 'flex', justifyContent: 'space-between' }}>
                  <span>{PARAM_DESCRIPTIONS[k] || k}</span>
                  <code style={{ fontSize: 10, color: '#64748b' }}>{k}</code>
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={v !== undefined ? v : 0}
                  onChange={(e) => handleChange(k, e.target.value)}
                  style={{
                    padding: '6px 10px',
                    border: '1px solid #cbd5e1',
                    borderRadius: 4,
                    fontSize: 12,
                    fontFamily: 'var(--font-mono)',
                    outline: 'none',
                  }}
                />
              </div>
            ))
          )}
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={handleSave}>Apply Changes</button>
        </div>
      </div>
    </div>
  );
}
