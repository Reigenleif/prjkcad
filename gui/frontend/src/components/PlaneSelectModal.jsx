import React from 'react';
import { Layers, X, Compass, MousePointerClick, CheckCircle2 } from 'lucide-react';

export default function PlaneSelectModal({
  isOpen,
  onSelectPlane,
  onSelectSurface,
  selectedSurface,
  onClose,
  hasExistingParts,
}) {
  if (!isOpen) return null;

  const planes = [
    {
      id: 'xy',
      name: 'Front Plane (XY)',
      desc: 'Frontal reference plane (Normal +Z, Vertical: Y)',
      coor: { coor_tx: 0.0, coor_ty: 0.0, coor_tz: 0.0, coor_euax: 0.0, coor_euay: 0.0, coor_euaz: 0.0 },
      color: '#2563eb',
      normal: [0, 0, 1],
    },
    {
      id: 'xz',
      name: 'Top Plane (XZ)',
      desc: 'Horizontal reference plane (Normal +Y, Vertical: -Z)',
      coor: { coor_tx: 0.0, coor_ty: 0.0, coor_tz: 0.0, coor_euax: -90.0, coor_euay: 0.0, coor_euaz: 0.0 },
      color: '#d97706',
      normal: [0, 1, 0],
    },
    {
      id: 'yz',
      name: 'Right Plane (YZ)',
      desc: 'Side profile plane (Normal +X, Vertical: Y)',
      coor: { coor_tx: 0.0, coor_ty: 0.0, coor_tz: 0.0, coor_euax: 0.0, coor_euay: 90.0, coor_euaz: 0.0 },
      color: '#16a34a',
      normal: [1, 0, 0],
    },
  ];

  return (
    <div className="modal-backdrop">
      <div className="modal-window" style={{ maxWidth: 540 }}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Compass size={18} color="#2563eb" />
            <span style={{ fontWeight: 600, fontSize: 14 }}>Start New Sketch - Select Plane or Surface</span>
          </div>
          <button className="header-icon-btn" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="modal-content" style={{ padding: 18 }}>
          {selectedSurface && (
            <div
              style={{
                marginBottom: 16,
                padding: '12px 14px',
                background: '#eff6ff',
                borderRadius: 8,
                border: '1.5px solid #3b82f6',
                boxShadow: 'var(--shadow-sm)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div
                    style={{
                      width: 36,
                      height: 36,
                      borderRadius: 6,
                      background: '#dbeafe',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    <MousePointerClick size={20} color="#2563eb" />
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e3a8a', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span>Selected 3D Face ({selectedSurface.isFlat ? 'Flat Surface' : 'Curved Surface'})</span>
                      <CheckCircle2 size={14} color="#16a34a" />
                    </div>
                    <div style={{ fontSize: 11, color: '#3b82f6', marginTop: 2 }}>
                      Normal: [{selectedSurface.normal.x.toFixed(2)}, {selectedSurface.normal.y.toFixed(2)}, {selectedSurface.normal.z.toFixed(2)}]
                    </div>
                  </div>
                </div>

                <button
                  className="btn-primary"
                  style={{ padding: '7px 14px', fontSize: 12, flexShrink: 0 }}
                  onClick={() => onSelectSurface?.(selectedSurface)}
                >
                  Start Sketch on This Face
                </button>
              </div>
            </div>
          )}

          <p style={{ fontSize: 12.5, color: '#475569', marginBottom: 12 }}>
            {selectedSurface
              ? 'Or choose an orthogonal reference plane below:'
              : 'Choose an orthogonal coordinate plane to start drawing your 2D sketch profile:'}
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
            {planes.map((p) => (
              <button
                key={p.id}
                onClick={() => onSelectPlane(p)}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  padding: '16px 10px',
                  background: '#ffffff',
                  border: '1.5px solid #e2e8f0',
                  borderRadius: 8,
                  cursor: 'pointer',
                  textAlign: 'center',
                  transition: 'all 0.15s ease',
                  boxShadow: 'var(--shadow-sm)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = p.color;
                  e.currentTarget.style.backgroundColor = '#f8fafc';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#e2e8f0';
                  e.currentTarget.style.backgroundColor = '#ffffff';
                  e.currentTarget.style.transform = 'none';
                }}
              >
                <div
                  style={{
                    width: 38,
                    height: 38,
                    borderRadius: 6,
                    background: `${p.color}15`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: 10,
                  }}
                >
                  <Layers size={20} color={p.color} />
                </div>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#0f172a', marginBottom: 4 }}>
                  {p.name}
                </div>
                <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.3 }}>
                  {p.desc}
                </div>
              </button>
            ))}
          </div>

          {hasExistingParts && !selectedSurface && (
            <div
              style={{
                marginTop: 16,
                padding: '10px 14px',
                background: '#f1f5f9',
                borderRadius: 6,
                border: '1px dashed #cbd5e1',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                fontSize: 12,
                color: '#334155',
              }}
            >
              <MousePointerClick size={18} color="#2563eb" style={{ flexShrink: 0 }} />
              <div>
                <b>Tip:</b> You can also click directly on any flat surface of the 3D model in the viewport to sketch on that face!
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer" style={{ justifyContent: 'flex-end' }}>
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
