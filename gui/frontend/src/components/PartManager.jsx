import React, { useState } from 'react';
import {
  ChevronRight,
  ChevronDown,
  Box,
  Compass,
  Layers,
  Circle,
  Minus,
  Spline,
  Scissors,
  Sliders,
  Plus,
  Trash2,
  Edit2,
} from 'lucide-react';

function getIconForType(type, cmd) {
  switch (type) {
    case 'part':
      return <Box size={14} color="#0f172a" />;
    case 'coor':
      return <Compass size={14} color="#2563eb" />;
    case 'face':
      return <Layers size={14} color="#0284c7" />;
    case 'loop':
      return <Circle size={13} color="#475569" />;
    case 'line':
      return <Minus size={13} color="#64748b" />;
    case 'circle':
      return <Circle size={13} color="#64748b" />;
    case 'arc':
      return <Spline size={13} color="#64748b" />;
    case 'extrude':
      if (cmd === 'EXTRUDE_CUT') return <Scissors size={14} color="#dc2626" />;
      if (cmd?.includes('THREAD')) return <Sliders size={14} color="#d97706" />;
      return <Box size={14} color="#16a34a" />;
    default:
      return <Box size={14} color="#94a3b8" />;
  }
}

function TreeNode({
  node,
  selectedId,
  onSelect,
  onEditParam,
  onDeleteCmd,
}) {
  const [expanded, setExpanded] = useState(true);
  const hasChildren = node.children && node.children.length > 0;
  const isSelected = selectedId === node.id;
  const hasParams = Boolean(node.params && Object.keys(node.params).length > 0);
  const isEditable = node.cmd_index !== undefined && hasParams;
  const canDelete = node.cmd_index !== undefined && node.type !== 'part';

  return (
    <div className="tree-node">
      <div
        className={`tree-row ${isSelected ? 'selected' : ''}`}
        onClick={() => onSelect(node)}
        onDoubleClick={() => {
          if (isEditable && onEditParam) {
            onEditParam(node);
          }
        }}
        title={isEditable ? 'Double-click to edit parameters' : ''}
      >
        <span
          style={{ width: 14, display: 'inline-flex', cursor: 'pointer' }}
          onClick={(e) => {
            e.stopPropagation();
            setExpanded(!expanded);
          }}
        >
          {hasChildren ? (
            expanded ? <ChevronDown size={12} color="#64748b" /> : <ChevronRight size={12} color="#64748b" />
          ) : (
            <span style={{ width: 12 }} />
          )}
        </span>

        {getIconForType(node.type, node.cmd)}
        <span className="tree-row-name" title={node.name}>{node.name}</span>

        {hasParams && (
          <span className="tree-row-param">
            {(node.type === 'extrude' || node.type === 'part') && `d=${node.params.extrude_new_dtn || node.params.extrude_cut_dtn || node.params.extrude_join_dtn || node.params.extrude_thread_dtn || 0}`}
            {node.type === 'line' && `(${node.params.line_sx},${node.params.line_sy})`}
            {node.type === 'circle' && `r=${node.params.circle_r}`}
            {node.type === 'coor' && `[${node.params.coor_tx},${node.params.coor_ty},${node.params.coor_tz}]`}
          </span>
        )}

        <div style={{ display: 'flex', gap: 4, marginLeft: 'auto' }}>
          {isEditable && (
            <button
              className="header-icon-btn"
              title="Edit Parameters"
              onClick={(e) => {
                e.stopPropagation();
                onEditParam(node);
              }}
            >
              <Edit2 size={11} />
            </button>
          )}
          {canDelete && (
            <button
              className="header-icon-btn"
              title="Delete Command"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteCmd(node.cmd_index);
              }}
            >
              <Trash2 size={11} color="#ef4444" />
            </button>
          )}
        </div>
      </div>

      {hasChildren && expanded && (
        <div className="tree-children">
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              selectedId={selectedId}
              onSelect={onSelect}
              onEditParam={onEditParam}
              onDeleteCmd={onDeleteCmd}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function PartManager({
  treeData,
  selectedNodeId,
  onSelectNode,
  onEditParam,
  onDeleteCmd,
  onNewSketch,
}) {
  const parts = treeData?.tree || [];
  const partCount = treeData?.part_count || 0;
  const totalCmds = treeData?.total_commands || 0;

  return (
    <div className="part-tree-container">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 6px', borderBottom: '1px solid #f1f5f9', marginBottom: 6 }}>
        <div style={{ display: 'flex', gap: 8, fontSize: 11, color: '#64748b' }}>
          <span>Parts: <b>{partCount}</b></span>
          <span>Operations: <b>{totalCmds}</b></span>
        </div>
        <div>
          <button
            className="header-icon-btn"
            onClick={onNewSketch}
            title="Create New Sketch"
            style={{ background: '#eff6ff', color: '#2563eb', padding: '3px 8px', borderRadius: 4, fontSize: 11, display: 'flex', gap: 4, alignItems: 'center', border: '1px solid #bfdbfe' }}
          >
            <Plus size={12} />
            <span>New Sketch</span>
          </button>
        </div>
      </div>

      {parts.length === 0 ? (
        <div style={{ padding: '24px 12px', textAlign: 'center', color: '#94a3b8', fontSize: 12 }}>
          <Box size={28} style={{ margin: '0 auto 8px', opacity: 0.5 }} />
          <div>No CAD parts defined.</div>
          <div style={{ fontSize: 11, marginTop: 4 }}>Ask the AI assistant or load a sample.</div>
        </div>
      ) : (
        parts.map((part) => (
          <TreeNode
            key={part.id}
            node={part}
            selectedId={selectedNodeId}
            onSelect={onSelectNode}
            onEditParam={onEditParam}
            onDeleteCmd={onDeleteCmd}
          />
        ))
      )}
    </div>
  );
}
