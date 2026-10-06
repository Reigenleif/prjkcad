import React, { useState, useEffect } from 'react';
import { Play, Check, AlertCircle, RefreshCw, X, Copy } from 'lucide-react';
import { renderDualSeq } from '../services/api';

export default function SequenceEditor({
  dualseqTuples,
  onUpdateSequence,
  onClose,
  isModal = false,
}) {
  const [jsonText, setJsonText] = useState('');
  const [error, setError] = useState(null);
  const [rendering, setRendering] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (dualseqTuples) {
      setJsonText(JSON.stringify(dualseqTuples, null, 2));
    }
  }, [dualseqTuples]);

  const handleApply = async () => {
    setError(null);
    let parsed;
    try {
      parsed = JSON.parse(jsonText);
    } catch (e) {
      setError(`JSON Syntax Error: ${e.message}`);
      return;
    }

    if (!Array.isArray(parsed)) {
      setError('Sequence must be an array of [cmd, arg_dict] pairs');
      return;
    }

    setRendering(true);
    try {
      const res = await renderDualSeq(parsed);
      if (res.success) {
        if (onUpdateSequence) {
          onUpdateSequence(parsed, res);
        }
      } else {
        setError(res.error || 'OpenCASCADE rendering failed for this sequence.');
      }
    } catch (err) {
      setError(err.message || 'Render request failed');
    } finally {
      setRendering(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(jsonText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const editorContent = (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#334155' }}>
          DualSeq Representation: List of <code>[cmd, arg_dict]</code> tuples
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <button
            className="btn-secondary"
            onClick={handleCopy}
            title="Copy sequence JSON"
            style={{ display: 'flex', alignItems: 'center', gap: 4 }}
          >
            {copied ? <Check size={13} color="#16a34a" /> : <Copy size={13} />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>

          <button
            className="btn-primary"
            onClick={handleApply}
            disabled={rendering}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            {rendering ? <RefreshCw size={13} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={13} />}
            <span>Apply & Render 3D</span>
          </button>

          {isModal && onClose && (
            <button className="btn-secondary" onClick={onClose}>
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {error && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#fee2e2', color: '#b91c1c', padding: '6px 10px', borderRadius: 4, fontSize: 11.5 }}>
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      <textarea
        value={jsonText}
        onChange={(e) => setJsonText(e.target.value)}
        style={{
          flex: 1,
          fontFamily: 'var(--font-mono)',
          fontSize: 11.5,
          padding: 10,
          border: '1px solid #cbd5e1',
          borderRadius: 6,
          background: '#0f172a',
          color: '#f8fafc',
          lineHeight: 1.45,
          resize: 'none',
          outline: 'none',
          minHeight: isModal ? 380 : 120,
        }}
      />
    </div>
  );

  if (isModal) {
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal-card" style={{ width: 720, height: 560 }} onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <span>DualSeq Sequence Editor</span>
            <button className="header-icon-btn" onClick={onClose}><X size={16} /></button>
          </div>
          <div className="modal-body" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            {editorContent}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bottom-editor-bar">
      <div className="editor-bar-header">
        <span>DualSeq Sequence Code Editor</span>
        <button className="header-icon-btn" onClick={onClose}><X size={14} /></button>
      </div>
      <div style={{ padding: 10, flex: 1, display: 'flex', flexDirection: 'column' }}>
        {editorContent}
      </div>
    </div>
  );
}
