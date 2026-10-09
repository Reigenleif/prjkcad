import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import TopRibbon from './components/TopRibbon';
import ResizableLayout from './components/ResizableLayout';
import PartManager from './components/PartManager';
import ChatDivision from './components/ChatDivision';
import CADViewport from './components/CADViewport';
import SequenceEditor from './components/SequenceEditor';
import ParamEditModal from './components/ParamEditModal';
import EvaluateModal from './components/EvaluateModal';
import PlaneSelectModal from './components/PlaneSelectModal';
import ExtrudeModal from './components/ExtrudeModal';
import { checkHealth, getSampleDetails, renderDualSeq } from './services/api';
import { inferFacesAndLoops, buildDualSeqSubpart } from './utils/sketchInference';

const DEFAULT_INITIAL_SEQUENCE = [
  ['COOR', { coor_euax: 0.0, coor_euay: 0.0, coor_euaz: 0.0, coor_tx: 0.0, coor_ty: 0.0, coor_tz: 0.0 }],
  ['FACE', {}],
  ['LOOP', {}],
  ['LINE', { line_sx: 0.0, line_sy: 0.0, line_ex: 1.0, line_ey: 0.0 }],
  ['LINE', { line_sx: 1.0, line_sy: 0.0, line_ex: 1.0, line_ey: 0.6 }],
  ['LINE', { line_sx: 1.0, line_sy: 0.6, line_ex: 0.0, line_ey: 0.6 }],
  ['LINE', { line_sx: 0.0, line_sy: 0.6, line_ex: 0.0, line_ey: 0.0 }],
  ['EXTRUDE_NEW', { extrude_new_dtn: 0.4, extrude_new_don: 0.0, extrude_new_scale: 1.0 }],
];

export default function App() {
  const [systemHealth, setSystemHealth] = useState(null);
  const [dualseqTuples, setDualseqTuples] = useState(DEFAULT_INITIAL_SEQUENCE);
  const [stlUrl, setStlUrl] = useState(null);
  const [treeData, setTreeData] = useState(null);
  const [properties, setProperties] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);

  const [viewMode, setViewMode] = useState('shaded_edges');
  const [showGrid, setShowGrid] = useState(true);
  const [cameraTrigger, setCameraTrigger] = useState(null);

  // Modals & Panels
  const [showBottomEditor, setShowBottomEditor] = useState(false);
  const [showEditorModal, setShowEditorModal] = useState(false);
  const [showEvaluateModal, setShowEvaluateModal] = useState(false);
  const [editingNode, setEditingNode] = useState(null);
  const [externalPrompt, setExternalPrompt] = useState('');

  // Interactive CAD Sketch System State
  const [showPlaneModal, setShowPlaneModal] = useState(false);
  const [showExtrudeModal, setShowExtrudeModal] = useState(false);
  const [pendingExtrudeType, setPendingExtrudeType] = useState('EXTRUDE_NEW');
  const [cachedFacesResult, setCachedFacesResult] = useState(null);

  const [inSketchMode, setInSketchMode] = useState(false);
  const [sketchPlane, setSketchPlane] = useState(null);
  const [sketchTool, setSketchTool] = useState('line');
  const [sketchEntities, setSketchEntities] = useState([]);
  const [selectedSurface, setSelectedSurface] = useState(null);

  // Initial startup
  useEffect(() => {
    checkHealth().then(setSystemHealth);

    renderDualSeq(DEFAULT_INITIAL_SEQUENCE, 'initial_workspace_model')
      .then((res) => {
        if (res.success) {
          setDualseqTuples(res.dualseq_tuples || DEFAULT_INITIAL_SEQUENCE);
          setStlUrl(res.stl_url);
          setTreeData(res.tree_data);
          setProperties(res.properties);
        }
      })
      .catch((err) => console.error('Failed initial render:', err));
  }, []);

  const handleSelectSample = async (uid) => {
    try {
      const data = await getSampleDetails(uid);
      if (data.dualseq_tuples && data.dualseq_tuples.length > 0) {
        setDualseqTuples(data.dualseq_tuples);
      }
      if (data.stl_url) {
        setStlUrl(data.stl_url);
      } else if (data.dualseq_tuples && data.dualseq_tuples.length > 0) {
        const renderRes = await renderDualSeq(data.dualseq_tuples, uid.replace('/', '_'));
        if (renderRes.success) {
          setDualseqTuples(renderRes.dualseq_tuples || data.dualseq_tuples);
          setStlUrl(renderRes.stl_url);
          setTreeData(renderRes.tree_data);
          setProperties(renderRes.properties);
        }
      }
      if (data.tree_data) {
        setTreeData(data.tree_data);
      }
      // Reset sketch mode if active
      setInSketchMode(false);
      setSketchEntities([]);
    } catch (err) {
      console.error('Error loading sample:', err);
    }
  };

  const handleNewModel = async () => {
    setInSketchMode(false);
    setSketchEntities([]);
    setDualseqTuples([]);
    setStlUrl(null);
    setTreeData(null);
    setProperties(null);
    setShowPlaneModal(true);
  };

  const handleApplyDualSeq = (tuples, renderRes, newTreeData) => {
    const activeTuples = renderRes?.dualseq_tuples || tuples;
    setDualseqTuples(activeTuples);
    if (renderRes && renderRes.stl_url) {
      setStlUrl(renderRes.stl_url);
      setProperties(renderRes.properties);
    }
    const finalTree = newTreeData || renderRes?.tree_data;
    if (finalTree) {
      setTreeData(finalTree);
    }
  };

  const handleDeleteCommand = async (cmdIndex) => {
    const updated = dualseqTuples.filter((_, idx) => idx !== cmdIndex);
    try {
      const res = await renderDualSeq(updated, 'deleted_cmd_model');
      if (res.success) {
        setDualseqTuples(res.dualseq_tuples || updated);
        setStlUrl(res.stl_url);
        setTreeData(res.tree_data);
        setProperties(res.properties);
      } else {
        alert(`Delete warning: ${res.error || 'Remaining commands do not form a valid solid'}`);
      }
    } catch (err) {
      console.error('Render error after deleting command:', err);
    }
  };

  const handleSaveParamEdit = async (cmdIndex, newParams) => {
    const updated = dualseqTuples.map((tuple, idx) => {
      if (idx === cmdIndex) {
        return [tuple[0], newParams];
      }
      return tuple;
    });

    setEditingNode(null);

    try {
      const res = await renderDualSeq(updated, 'edited_param_model');
      if (res.success) {
        setDualseqTuples(res.dualseq_tuples || updated);
        setStlUrl(res.stl_url);
        setTreeData(res.tree_data);
        setProperties(res.properties);
      } else {
        alert(`Parameter update error: ${res.error || 'Solid construction failed with these parameters.'}`);
      }
    } catch (err) {
      alert(`Save error: ${err.message}`);
    }
  };

  // Sketch Workflow Handlers
  const handleNewSketch = () => {
    setShowPlaneModal(true);
  };

  const handleSelectPlane = (planeDef) => {
    setShowPlaneModal(false);
    setSelectedSurface(null);
    setSketchPlane({
      name: planeDef.name,
      coor: planeDef.coor,
      normal: planeDef.normal,
    });
    setSketchEntities([]);
    setSketchTool('line');
    setInSketchMode(true);
  };

  const handleStartSketchOnSurface = (coor, normal) => {
    setShowPlaneModal(false);
    setSelectedSurface(null);
    setSketchPlane({
      name: 'Surface Face',
      coor,
      normal: Array.isArray(normal) ? normal : [normal.x, normal.y, normal.z],
    });
    setSketchEntities([]);
    setSketchTool('line');
    setInSketchMode(true);
  };

  const handleTriggerExtrude = (extrudeType = 'EXTRUDE_NEW') => {
    if (!inSketchMode) {
      alert('Please start a sketch first by clicking "New Sketch" or picking a part face.');
      return;
    }

    if (sketchEntities.length === 0) {
      alert('Please draw at least one closed sketch profile (lines, circle, or arc) before extruding.');
      return;
    }

    const loopsRes = inferFacesAndLoops(sketchEntities);
    if (!loopsRes.success) {
      alert(loopsRes.error || 'No closed loops detected. Please ensure line segments connect to form closed cycles.');
      return;
    }

    setCachedFacesResult(loopsRes);
    setPendingExtrudeType(extrudeType);
    setShowExtrudeModal(true);
  };

  const handleConfirmExtrude = async ({ extrudeCmd, extrudeParams }) => {
    if (!sketchPlane || !cachedFacesResult) return;

    const newSubpart = buildDualSeqSubpart({
      coorParams: sketchPlane.coor,
      facesResult: cachedFacesResult,
      extrudeCmd,
      extrudeParams,
    });

    const updated = [...dualseqTuples, ...newSubpart];

    try {
      const res = await renderDualSeq(updated, 'sketch_extrude_model');
      if (res.success) {
        setDualseqTuples(res.dualseq_tuples || updated);
        setStlUrl(res.stl_url);
        setTreeData(res.tree_data);
        setProperties(res.properties);

        // Exit sketch mode successfully
        setInSketchMode(false);
        setSketchPlane(null);
        setSketchEntities([]);
        setShowExtrudeModal(false);
        setCachedFacesResult(null);
      } else {
        alert(`Solid construction failed: ${res.error || 'Could not extrude geometry.'}`);
      }
    } catch (err) {
      alert(`Render error: ${err.message}`);
    }
  };

  const handleExitSketch = () => {
    setInSketchMode(false);
    setSketchPlane(null);
    setSketchEntities([]);
    setShowExtrudeModal(false);
    setCachedFacesResult(null);
  };

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <Navbar
        systemHealth={systemHealth}
        dualseqTuples={dualseqTuples}
        onNewModel={handleNewModel}
        onOpenEditor={() => setShowEditorModal(true)}
      />

      {/* Top CAD SolidWorks-style Ribbon */}
      <TopRibbon
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        onResetCamera={() => setCameraTrigger(Date.now().toString() + '_iso')}
        onSetCameraView={(v) => setCameraTrigger(Date.now().toString() + '_' + v)}
        showGrid={showGrid}
        onToggleGrid={() => setShowGrid(!showGrid)}
        onOpenEditor={() => setShowBottomEditor(!showBottomEditor)}
        onEvaluateModal={() => setShowEvaluateModal(true)}
        onSelectQuickPrompt={(p) => setExternalPrompt(p)}
        // CAD Sub-part Sketch Creation System
        inSketchMode={inSketchMode}
        sketchTool={sketchTool}
        onSetSketchTool={setSketchTool}
        onNewSketch={handleNewSketch}
        onTriggerExtrude={handleTriggerExtrude}
        onExitSketch={handleExitSketch}
      />

      {/* Resizable 2-Division Left Sidebar & Center 3D CAD Editor */}
      <ResizableLayout
        partManagerComponent={
          <PartManager
            treeData={treeData}
            selectedNodeId={selectedNode?.id}
            onSelectNode={setSelectedNode}
            onEditParam={(node) => setEditingNode(node)}
            onDeleteCmd={handleDeleteCommand}
            onNewSketch={handleNewSketch}
          />
        }
        chatDivisionComponent={
          <ChatDivision
            onApplyDualSeq={handleApplyDualSeq}
            externalPrompt={externalPrompt}
            onClearExternalPrompt={() => setExternalPrompt('')}
          />
        }
        centerViewportComponent={
          <CADViewport
            stlUrl={stlUrl}
            viewMode={viewMode}
            showGrid={showGrid}
            properties={properties}
            cameraTrigger={cameraTrigger ? cameraTrigger.split('_')[1] : null}
            selectedSurface={selectedSurface}
            onSelectSurface={setSelectedSurface}
            // Sketch Mode Props
            inSketchMode={inSketchMode}
            sketchPlane={sketchPlane}
            sketchTool={sketchTool}
            onSetSketchTool={setSketchTool}
            sketchEntities={sketchEntities}
            onUpdateSketchEntities={setSketchEntities}
            onStartSketchOnSurface={handleStartSketchOnSurface}
            onOpenExtrudeModal={() => handleTriggerExtrude('EXTRUDE_NEW')}
            onExitSketch={handleExitSketch}
          />
        }
        bottomEditorComponent={
          <SequenceEditor
            dualseqTuples={dualseqTuples}
            onUpdateSequence={(tuples, res) => handleApplyDualSeq(tuples, res, res.tree_data)}
            onClose={() => setShowBottomEditor(false)}
          />
        }
        showBottomEditor={showBottomEditor}
      />

      {/* Plane Selection Modal (Draw from nothing or on selected surface) */}
      <PlaneSelectModal
        isOpen={showPlaneModal}
        selectedSurface={selectedSurface}
        onSelectSurface={(surf) => handleStartSketchOnSurface(surf.coor, surf.normal)}
        onSelectPlane={handleSelectPlane}
        onClose={() => setShowPlaneModal(false)}
        hasExistingParts={dualseqTuples && dualseqTuples.length > 0}
      />

      {/* Extrude PropertyManager Modal */}
      <ExtrudeModal
        isOpen={showExtrudeModal}
        facesResult={cachedFacesResult}
        initialType={pendingExtrudeType}
        onConfirm={handleConfirmExtrude}
        onClose={() => setShowExtrudeModal(false)}
      />

      {/* Full Modal Sequence Editor */}
      {showEditorModal && (
        <SequenceEditor
          dualseqTuples={dualseqTuples}
          onUpdateSequence={(tuples, res) => handleApplyDualSeq(tuples, res, res.tree_data)}
          onClose={() => setShowEditorModal(false)}
          isModal={true}
        />
      )}

      {/* Parameter Edit Modal */}
      {editingNode && (
        <ParamEditModal
          node={editingNode}
          onSave={handleSaveParamEdit}
          onClose={() => setEditingNode(null)}
        />
      )}

      {/* Geometry / Mass Properties Modal */}
      {showEvaluateModal && (
        <EvaluateModal
          properties={properties}
          treeData={treeData}
          dualseqTuples={dualseqTuples}
          onClose={() => setShowEvaluateModal(false)}
        />
      )}
    </div>
  );
}
