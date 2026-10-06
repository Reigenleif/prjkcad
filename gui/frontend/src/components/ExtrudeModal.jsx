import React, { useState } from 'react';
import { Box, Scissors, Layers, Sliders, X, Check, ArrowUpDown } from 'lucide-react';

export default function ExtrudeModal({
  isOpen,
  facesResult,
  initialType = 'EXTRUDE_NEW',
  onConfirm,
  onClose,
}) {
  if (!isOpen) return null;

  const [extrudeType, setExtrudeType] = useState(initialType);
  const [depth, setDepth] = useState(0.4);
  const [offset, setOffset] = useState(0.0);
  const [scale, setScale] = useState(1.0);
  const [reverseDirection, setReverseDirection] = useState(false);

  // Thread specific parameters
  const [threadPitch, setThreadPitch] = useState(0.1);
  const [threadDepth, setThreadDepth] = useState(0.04);

  const faces = facesResult?.faces || [];
  const facesCount = facesResult?.facesCount || 0;
  const loopsCount = facesResult?.loopsCount || 0;

  const handleApply = () => {
    const finalDepth = reverseDirection ? -Math.abs(depth) : Math.abs(depth);
    const params = {};

    if (extrudeType === 'EXTRUDE_NEW') {
      params.extrude_new_dtn = Number(finalDepth.toFixed(4));
      params.extrude_new_don = Number(offset.toFixed(4));
      params.extrude_new_scale = Number(scale.toFixed(4));
    } else if (extrudeType === 'EXTRUDE_JOIN') {
      params.extrude_join_dtn = Number(finalDepth.toFixed(4));
      params.extrude_join_don = Number(offset.toFixed(4));
      params.extrude_join_scale = Number(scale.toFixed(4));
    } else if (extrudeType === 'EXTRUDE_CUT') {
      params.extrude_cut_dtn = Number(finalDepth.toFixed(4));
      params.extrude_cut_don = Number(offset.toFixed(4));
      params.extrude_cut_scale = Number(scale.toFixed(4));
    } else if (extrudeType === 'EXTRUDE_THREAD') {
      params.extrude_thread_dtn = Number(finalDepth.toFixed(4));
      params.extrude_thread_don = Number(offset.toFixed(4));
      params.extrude_thread_scale = Number(scale.toFixed(4));
      params.extrude_thread_pitch = Number(threadPitch.toFixed(4));
      params.extrude_thread_depth = Number(threadDepth.toFixed(4));
    } else if (extrudeType === 'EXTRUDE_THREAD_CUT') {
      params.extrude_thread_cut_dtn = Number(finalDepth.toFixed(4));
      params.extrude_thread_cut_don = Number(offset.toFixed(4));
      params.extrude_thread_cut_scale = Number(scale.toFixed(4));
      params.extrude_thread_cut_pitch = Number(threadPitch.toFixed(4));
      params.extrude_thread_cut_depth = Number(threadDepth.toFixed(4));
    }

    onConfirm({
      extrudeCmd: extrudeType,
      extrudeParams: params,
    });
  };

  return (
    <div className="modal-backdrop">
      <div className="modal-window" style={{ maxWidth: 540 }}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Box size={18} color="#2563eb" />
            <span style={{ fontWeight: 600, fontSize: 14 }}>Extrude Feature PropertyManager</span>
          </div>
          <button className="header-icon-btn" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="modal-content" style={{ padding: 18 }}>
          {/* Loop Inference Topology Summary */}
          <div
            style={{
              padding: '10px 12px',
              background: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: 6,
              marginBottom: 16,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: 12,
            }}
          >
            <div>
              <span style={{ fontWeight: 600, color: '#0f172a' }}>Detected Topology: </span>
              <span style={{ color: '#2563eb', fontWeight: 600 }}>
                {facesCount} {facesCount === 1 ? 'Face' : 'Faces'}
              </span>
              <span style={{ color: '#64748b' }}> ({loopsCount} closed {loopsCount === 1 ? 'loop' : 'loops'})</span>
            </div>
            {faces.some((f) => f.inners?.length > 0) && (
              <span
                style={{
                  fontSize: 11,
                  background: '#dbeafe',
                  color: '#1e40af',
                  padding: '2px 8px',
                  borderRadius: 4,
                  fontWeight: 600,
                }}
              >
                Holes Detected
              </span>
            )}
          </div>

          {/* Extrusion Type Selector */}
          <div style={{ marginBottom: 16 }}>
            <label style={{ fontSize: 11.5, fontWeight: 600, color: '#475569', display: 'block', marginBottom: 6 }}>
              OPERATION TYPE
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6 }}>
              <button
                type="button"
                className={`ribbon-btn ${extrudeType === 'EXTRUDE_NEW' ? 'primary' : ''}`}
                style={{ height: 'auto', padding: '10px 4px', flexDirection: 'column', gap: 4 }}
                onClick={() => setExtrudeType('EXTRUDE_NEW')}
              >
                <Box size={18} />
                <span style={{ fontSize: 10.5, fontWeight: 600 }}>Boss / Base</span>
              </button>

              <button
                type="button"
                className={`ribbon-btn ${extrudeType === 'EXTRUDE_JOIN' ? 'primary' : ''}`}
                style={{ height: 'auto', padding: '10px 4px', flexDirection: 'column', gap: 4 }}
                onClick={() => setExtrudeType('EXTRUDE_JOIN')}
              >
                <Layers size={18} />
                <span style={{ fontSize: 10.5, fontWeight: 600 }}>Join</span>
              </button>

              <button
                type="button"
                className={`ribbon-btn ${extrudeType === 'EXTRUDE_CUT' ? 'primary' : ''}`}
                style={{ height: 'auto', padding: '10px 4px', flexDirection: 'column', gap: 4 }}
                onClick={() => setExtrudeType('EXTRUDE_CUT')}
              >
                <Scissors size={18} color={extrudeType === 'EXTRUDE_CUT' ? '#ffffff' : '#dc2626'} />
                <span style={{ fontSize: 10.5, fontWeight: 600 }}>Cut</span>
              </button>

              <button
                type="button"
                className={`ribbon-btn ${extrudeType === 'EXTRUDE_THREAD' ? 'primary' : ''}`}
                style={{ height: 'auto', padding: '10px 4px', flexDirection: 'column', gap: 4 }}
                onClick={() => setExtrudeType('EXTRUDE_THREAD')}
              >
                <Sliders size={18} />
                <span style={{ fontSize: 10.5, fontWeight: 600 }}>Thread</span>
              </button>

              <button
                type="button"
                className={`ribbon-btn ${extrudeType === 'EXTRUDE_THREAD_CUT' ? 'primary' : ''}`}
                style={{ height: 'auto', padding: '10px 4px', flexDirection: 'column', gap: 4 }}
                onClick={() => setExtrudeType('EXTRUDE_THREAD_CUT')}
              >
                <Scissors size={18} color={extrudeType === 'EXTRUDE_THREAD_CUT' ? '#ffffff' : '#d97706'} />
                <span style={{ fontSize: 10.5, fontWeight: 600 }}>Thread Cut</span>
              </button>
            </div>
          </div>

          {/* Extrusion Parameters */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {/* Depth Slider */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <label style={{ fontSize: 12, fontWeight: 500, color: '#334155' }}>
                  Extrude Depth (dtn)
                </label>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input
                    type="number"
                    step="0.05"
                    min="0.01"
                    max="10.0"
                    value={depth}
                    onChange={(e) => setDepth(Math.max(0.01, parseFloat(e.target.value) || 0.1))}
                    style={{
                      width: 68,
                      padding: '3px 6px',
                      fontSize: 12,
                      border: '1px solid #cbd5e1',
                      borderRadius: 4,
                      textAlign: 'right',
                      fontFamily: 'var(--font-mono)',
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setReverseDirection(!reverseDirection)}
                    title="Reverse Extrusion Direction"
                    style={{
                      border: '1px solid #cbd5e1',
                      background: reverseDirection ? '#eff6ff' : '#f8fafc',
                      color: reverseDirection ? '#2563eb' : '#475569',
                      padding: '3px 6px',
                      borderRadius: 4,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 2,
                      fontSize: 11,
                    }}
                  >
                    <ArrowUpDown size={12} />
                    <span>{reverseDirection ? 'Rev' : 'Fwd'}</span>
                  </button>
                </div>
              </div>
              <input
                type="range"
                min="0.05"
                max="2.0"
                step="0.05"
                value={depth}
                onChange={(e) => setDepth(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#2563eb' }}
              />
            </div>

            {/* Offset / Draft Scale */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div>
                <label style={{ fontSize: 11.5, fontWeight: 500, color: '#475569', display: 'block', marginBottom: 4 }}>
                  Offset Start (don)
                </label>
                <input
                  type="number"
                  step="0.05"
                  value={offset}
                  onChange={(e) => setOffset(parseFloat(e.target.value) || 0.0)}
                  style={{
                    width: '100%',
                    padding: '5px 8px',
                    fontSize: 12,
                    border: '1px solid #cbd5e1',
                    borderRadius: 4,
                    fontFamily: 'var(--font-mono)',
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: 11.5, fontWeight: 500, color: '#475569', display: 'block', marginBottom: 4 }}>
                  Draft / Taper Scale
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="3.0"
                  value={scale}
                  onChange={(e) => setScale(parseFloat(e.target.value) || 1.0)}
                  style={{
                    width: '100%',
                    padding: '5px 8px',
                    fontSize: 12,
                    border: '1px solid #cbd5e1',
                    borderRadius: 4,
                    fontFamily: 'var(--font-mono)',
                  }}
                />
              </div>
            </div>

            {/* Thread specific parameters if thread selected */}
            {(extrudeType === 'EXTRUDE_THREAD' || extrudeType === 'EXTRUDE_THREAD_CUT') && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, padding: 10, background: '#fffbeb', borderRadius: 6, border: '1px solid #fef3c7' }}>
                <div>
                  <label style={{ fontSize: 11.5, fontWeight: 600, color: '#92400e', display: 'block', marginBottom: 4 }}>
                    Thread Pitch
                  </label>
                  <input
                    type="number"
                    step="0.02"
                    min="0.02"
                    max="1.0"
                    value={threadPitch}
                    onChange={(e) => setThreadPitch(parseFloat(e.target.value) || 0.1)}
                    style={{
                      width: '100%',
                      padding: '4px 8px',
                      fontSize: 12,
                      border: '1px solid #fde68a',
                      borderRadius: 4,
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: 11.5, fontWeight: 600, color: '#92400e', display: 'block', marginBottom: 4 }}>
                    Thread Depth
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    min="0.01"
                    max="0.5"
                    value={threadDepth}
                    onChange={(e) => setThreadDepth(parseFloat(e.target.value) || 0.04)}
                    style={{
                      width: '100%',
                      padding: '4px 8px',
                      fontSize: 12,
                      border: '1px solid #fde68a',
                      borderRadius: 4,
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="modal-footer" style={{ justifyContent: 'space-between' }}>
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn-primary"
            onClick={handleApply}
            style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#16a34a' }}
          >
            <Check size={16} />
            <span>Generate Extrude Solid</span>
          </button>
        </div>
      </div>
    </div>
  );
}
