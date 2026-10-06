import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Code2,
  Check,
  Play,
  RotateCw,
  Info,
} from 'lucide-react';
import { streamChatMessage } from '../services/api';

export default function ChatDivision({
  onApplyDualSeq,
  externalPrompt,
  onClearExternalPrompt,
}) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Welcome to **Text2CAD Assistant**! Describe any 3D mechanical part (e.g., *“Create an L-bracket with thickness 0.2”* or *“Threaded bolt with head”*) or choose a preset below. I will stream the step-by-step geometric reasoning and generate a validated DualSeq operation sequence.',
      thought: null,
      dualseqTuples: null,
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingTokens, setStreamingTokens] = useState('');
  const [expandedThoughts, setExpandedThoughts] = useState({ 0: false });
  const messagesEndRef = useRef(null);

  const quickPrompts = [
    'Mounting Plate with Thread Cut',
    'Threaded Bolt with Head and Shaft',
    'Threaded Rod (R 0.2, L 0.8, Pitch 0.1)',
    'Hollow Cylinder Tube (OD 0.5, ID 0.3)',
    'L-Bracket Profile (W 0.8, H 0.8, T 0.2)',
  ];

  useEffect(() => {
    if (externalPrompt) {
      setInput(externalPrompt);
      if (onClearExternalPrompt) onClearExternalPrompt();
    }
  }, [externalPrompt, onClearExternalPrompt]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading, streamingTokens]);

  const toggleThought = (idx) => {
    setExpandedThoughts((prev) => ({
      ...prev,
      [idx]: !prev[idx],
    }));
  };

  const handleSend = async (textToSend) => {
    const prompt = (textToSend || input).trim();
    if (!prompt || loading) return;

    const userMsg = { role: 'user', content: prompt };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setStreamingTokens('');

    const history = messages
      .filter((m) => m.content && !m.error)
      .map((m) => ({
        role: m.role,
        content: m.content,
        thought: m.thought || null,
        dualseq_tuples: m.dualseqTuples || null,
      }));

    await streamChatMessage(
      prompt,
      history,
      (token) => {
        setStreamingTokens((prev) => prev + token);
      },
      (res) => {
        const nextIndex = messages.length + 1;
        const aiMsg = {
          role: 'assistant',
          content: res.assistant_message || 'Model generated successfully.',
          thought: res.thought,
          dualseqTuples: res.dualseq_tuples,
          treeData: res.tree_data,
          renderResult: res.render_result,
          error: res.error,
        };

        setMessages((prev) => [...prev, aiMsg]);
        setExpandedThoughts((prev) => ({
          ...prev,
          [nextIndex]: true,
        }));

        if (res.dualseq_tuples && res.dualseq_tuples.length > 0 && onApplyDualSeq) {
          onApplyDualSeq(res.dualseq_tuples, res.render_result, res.tree_data);
        }

        setStreamingTokens('');
        setLoading(false);
      },
      (err) => {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `Generation error: ${err.message}`,
            error: true,
          },
        ]);
        setStreamingTokens('');
        setLoading(false);
      }
    );
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Parse streaming tokens for live thinking preview
  let liveThought = '';
  let liveCode = '';
  if (loading && streamingTokens) {
    const thoughtMatch = streamingTokens.match(/<(?:thought|think)>([\s\S]*?)(?:<\/(?:thought|think)>|$)/i);
    if (thoughtMatch) {
      liveThought = thoughtMatch[1].trim();
    }
    const codeMatch = streamingTokens.match(/```(?:json)?([\s\S]*?)(?:```|$)/i);
    if (codeMatch) {
      liveCode = codeMatch[1].trim();
    }
  }

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.role}`}>
            <div className="message-bubble">
              <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>

              {msg.thought && (
                <div className="thought-card">
                  <div className="thought-header" onClick={() => toggleThought(idx)}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Sparkles size={12} color="#2563eb" />
                      <span>Geometric Reasoning</span>
                    </div>
                    {expandedThoughts[idx] ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                  </div>
                  {expandedThoughts[idx] && (
                    <div className="thought-content">{msg.thought}</div>
                  )}
                </div>
              )}

              {msg.dualseqTuples && msg.dualseqTuples.length > 0 && (
                <div className="dualseq-preview-card">
                  <div className="dualseq-preview-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Code2 size={12} />
                      <span>DualSeq Operations ({msg.dualseqTuples.length} cmds)</span>
                    </div>
                  </div>
                  <div className="dualseq-code-view">
                    {JSON.stringify(msg.dualseqTuples, null, 2)}
                  </div>
                  <button
                    className="apply-cad-btn"
                    onClick={() =>
                      onApplyDualSeq(msg.dualseqTuples, msg.renderResult, msg.treeData)
                    }
                  >
                    <Play size={12} />
                    <span>Load into CAD Viewport</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message assistant">
            <div className="message-bubble" style={{ width: '100%' }}>
              {liveThought && (
                <div className="streaming-thought-card">
                  <div className="streaming-thought-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Sparkles size={12} color="#2563eb" />
                      <span>Thinking Process (Streaming)...</span>
                    </div>
                    <span className="pulsing-dot" />
                  </div>
                  <div className="streaming-thought-content">{liveThought}</div>
                </div>
              )}

              {liveCode && (
                <div className="dualseq-preview-card" style={{ marginTop: 6, borderColor: '#93c5fd' }}>
                  <div className="dualseq-preview-header" style={{ background: '#eff6ff', color: '#1d4ed8' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Code2 size={12} />
                      <span>DualSeq Operations (Synthesizing...)</span>
                    </div>
                    <span className="pulsing-dot" />
                  </div>
                  <div className="dualseq-code-view">{liveCode}</div>
                </div>
              )}

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  marginTop: liveThought || liveCode ? 8 : 0,
                  fontSize: 12,
                  color: '#64748b',
                }}
              >
                <RotateCw size={13} className="spin-icon" style={{ animation: 'spin 1s linear infinite' }} />
                <span>
                  {liveCode
                    ? 'Synthesizing DualSeq operations & OCC solid...'
                    : liveThought
                    ? 'Analyzing geometry & topological constraints...'
                    : 'Reasoning & generating CAD sequence...'}
                </span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-bar">
        <div className="quick-prompts-row">
          {quickPrompts.map((qp, i) => (
            <button key={i} className="quick-chip" onClick={() => handleSend(qp)}>
              {qp}
            </button>
          ))}
        </div>

        <div className="chat-input-wrapper">
          <textarea
            className="chat-textarea"
            placeholder="Type CAD instruction (e.g. Threaded bolt with head)..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
          />
          <button
            className="chat-send-btn"
            onClick={() => handleSend()}
            disabled={!input.trim() || loading}
            title="Send to AI Assistant"
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}
