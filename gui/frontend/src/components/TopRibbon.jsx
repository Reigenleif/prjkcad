import React, { useState } from 'react';
import {
  Box,
  Layers,
  Scissors,
  Circle,
  Minus,
  Spline,
  Square,
  RotateCcw,
  Compass,
  Grid,
  Eye,
  Sliders,
  BarChart3,
  FileCode2,
  PenTool,
  Check,
  XCircle,
} from 'lucide-react';

export default function TopRibbon({
  viewMode,
  onViewModeChange,
  onResetCamera,
  onSetCameraView,
  showGrid,
  onToggleGrid,
  onOpenEditor,
  onEvaluateModal,
  onSelectQuickPrompt,
  // CAD Sub-part Sketch Creation System
  inSketchMode = false,
  sketchTool = 'line',
  onSetSketchTool,
  onNewSketch,
  onTriggerExtrude,
  onExitSketch,
}) {
  const [activeTab, setActiveTab] = useState('features');

  return (
    <div className="ribbon-container">
      <div className="ribbon-tabs">
        <button
          className={`ribbon-tab ${activeTab === 'features' ? 'active' : ''}`}
          onClick={() => setActiveTab('features')}
        >
          Features
        </button>
        <button
          className={`ribbon-tab ${activeTab === 'sketch' ? 'active' : ''}`}
          onClick={() => setActiveTab('sketch')}
        >
          Sketch {inSketchMode && '●'}
        </button>
        <button
          className={`ribbon-tab ${activeTab === 'evaluate' ? 'active' : ''}`}
          onClick={() => setActiveTab('evaluate')}
        >
          Evaluate
        </button>
        <button
          className={`ribbon-tab ${activeTab === 'view' ? 'active' : ''}`}
          onClick={() => setActiveTab('view')}
        >
          View
        </button>
      </div>

      <div className="ribbon-toolbar">
        {activeTab === 'features' && (
          <>
            <div className="ribbon-group">
              <button
                className="ribbon-btn primary"
                onClick={onNewSketch}
                title="Create a new 2D sketch on a reference plane or face"
              >
                <PenTool size={18} />
                <span>New Sketch</span>
              </button>
            </div>

            <div className="ribbon-group">
              <button
                className="ribbon-btn"
                onClick={() => onTriggerExtrude('EXTRUDE_NEW')}
                title="Extruded Boss/Base: Add solid material"
              >
                <Box size={18} color="#2563eb" />
                <span>Extrude Boss</span>
              </button>

              <button
                className="ribbon-btn"
                onClick={() => onTriggerExtrude('EXTRUDE_JOIN')}
                title="Extrude Join: Union with existing solid"
              >
                <Layers size={18} color="#16a34a" />
                <span>Extrude Join</span>
              </button>

              <button
                className="ribbon-btn"
                onClick={() => onTriggerExtrude('EXTRUDE_CUT')}
                title="Extruded Cut: Subtract material from existing solid"
              >
                <Scissors size={18} color="#dc2626" />
                <span>Extrude Cut</span>
              </button>

              <button
                className="ribbon-btn"
                onClick={() => onTriggerExtrude('EXTRUDE_THREAD')}
                title="Extrude Threaded Cylinder"
              >
                <Sliders size={18} color="#d97706" />
                <span>Thread Boss</span>
              </button>
            </div>

            {inSketchMode && (
              <div className="ribbon-group" style={{ marginLeft: 'auto' }}>
                <span style={{ fontSize: 11.5, color: '#2563eb', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
                  ● Sketch Active
                </span>
                <button
                  className="ribbon-btn"
                  onClick={() => onTriggerExtrude('EXTRUDE_NEW')}
                  style={{ background: '#16a34a', color: 'white' }}
                >
                  <Check size={16} />
                  <span>Extrude...</span>
                </button>
                <button
                  className="ribbon-btn"
                  onClick={onExitSketch}
                  title="Exit sketch mode"
                >
                  <XCircle size={16} color="#ef4444" />
                  <span>Exit Sketch</span>
                </button>
              </div>
            )}
          </>
        )}

        {activeTab === 'sketch' && (
          <>
            <div className="ribbon-group">
              <button
                className={`ribbon-btn ${!inSketchMode ? 'primary' : ''}`}
                onClick={onNewSketch}
                title="Start a new sketch"
              >
                <PenTool size={18} />
                <span>New Sketch</span>
              </button>
            </div>

            {inSketchMode ? (
              <>
                <div className="ribbon-group">
                  <button
                    className={`ribbon-btn ${sketchTool === 'line' ? 'primary' : ''}`}
                    onClick={() => onSetSketchTool?.('line')}
                    title="Continuous line segments"
                  >
                    <Minus size={18} />
                    <span>Line</span>
                  </button>

                  <button
                    className={`ribbon-btn ${sketchTool === 'rectangle' ? 'primary' : ''}`}
                    onClick={() => onSetSketchTool?.('rectangle')}
                    title="Corner-to-corner rectangle"
                  >
                    <Square size={18} />
                    <span>Rectangle</span>
                  </button>

                  <button
                    className={`ribbon-btn ${sketchTool === 'circle' ? 'primary' : ''}`}
                    onClick={() => onSetSketchTool?.('circle')}
                    title="Center and radius circle"
                  >
                    <Circle size={18} />
                    <span>Circle</span>
                  </button>

                  <button
                    className={`ribbon-btn ${sketchTool === 'arc' ? 'primary' : ''}`}
                    onClick={() => onSetSketchTool?.('arc')}
                    title="3-Point Arc (Start, Mid, End)"
                  >
                    <Spline size={18} />
                    <span>3-Pt Arc</span>
                  </button>
                </div>

                <div className="ribbon-group">
                  <button
                    className="ribbon-btn"
                    onClick={() => onTriggerExtrude('EXTRUDE_NEW')}
                    style={{ background: '#16a34a', color: 'white', fontWeight: 600 }}
                    title="Finish and Extrude sketch"
                  >
                    <Check size={16} />
                    <span>Extrude Sketch</span>
                  </button>

                  <button
                    className="ribbon-btn"
                    onClick={onExitSketch}
                    title="Exit sketch mode without extruding"
                  >
                    <XCircle size={16} color="#ef4444" />
                    <span>Cancel</span>
                  </button>
                </div>
              </>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', fontSize: 12, color: '#64748b', padding: '0 8px' }}>
                Click "New Sketch" above or click any face in the 3D viewport to start drawing.
              </div>
            )}
          </>
        )}

        {activeTab === 'evaluate' && (
          <>
            <div className="ribbon-group">
              <button className="ribbon-btn" onClick={onEvaluateModal} title="Measure Mass Properties & Dimensions">
                <BarChart3 size={20} color="#2563eb" />
                <span>Mass Properties</span>
              </button>
            </div>
            <div className="ribbon-group">
              <button className="ribbon-btn" onClick={onOpenEditor} title="View Raw DualSeq Representation">
                <FileCode2 size={20} color="#0f172a" />
                <span>DualSeq Tree</span>
              </button>
            </div>
          </>
        )}

        {activeTab === 'view' && (
          <>
            <div className="ribbon-group">
              <button className="ribbon-btn" onClick={onToggleGrid} title="Toggle Ground Grid">
                <Grid size={18} color={showGrid ? '#2563eb' : '#64748b'} />
                <span>Grid: {showGrid ? 'On' : 'Off'}</span>
              </button>
              <button className="ribbon-btn" onClick={onResetCamera} title="Reset Camera to Isometric View">
                <RotateCcw size={18} />
                <span>Reset View</span>
              </button>
            </div>

            <div className="ribbon-group">
              <button className="ribbon-btn" onClick={() => onSetCameraView?.('top')} title="Top View">
                <span style={{ fontSize: 11, fontWeight: 700 }}>Top</span>
              </button>
              <button className="ribbon-btn" onClick={() => onSetCameraView?.('front')} title="Front View">
                <span style={{ fontSize: 11, fontWeight: 700 }}>Front</span>
              </button>
              <button className="ribbon-btn" onClick={() => onSetCameraView?.('right')} title="Right View">
                <span style={{ fontSize: 11, fontWeight: 700 }}>Right</span>
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
