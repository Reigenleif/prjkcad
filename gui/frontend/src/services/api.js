const API_BASE = '/api';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('Health check failed:', err);
    return { status: 'offline', device: 'unknown' };
  }
}

export async function getCuratedSamples() {
  const res = await fetch(`${API_BASE}/samples/curated`);
  if (!res.ok) throw new Error('Failed to load curated samples');
  return await res.json();
}

export async function getSampleDetails(uid) {
  const res = await fetch(`${API_BASE}/samples/${encodeURIComponent(uid)}`);
  if (!res.ok) throw new Error(`Failed to load sample ${uid}`);
  return await res.json();
}

export async function sendChatMessage(prompt, history = []) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, history }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Chat request failed');
  }
  return await res.json();
}

export async function streamChatMessage(prompt, history = [], onChunk, onDone, onError) {
  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, history }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop();

      for (const block of blocks) {
        const trimmed = block.trim();
        if (!trimmed.startsWith('data:')) continue;
        const jsonStr = trimmed.slice(5).trim();
        if (!jsonStr) continue;

        try {
          const payload = JSON.parse(jsonStr);
          if (payload.type === 'token') {
            onChunk && onChunk(payload.token);
          } else if (payload.type === 'done') {
            onDone && onDone(payload);
          } else if (payload.type === 'error') {
            onError && onError(new Error(payload.error));
          }
        } catch (e) {
          console.warn('Failed to parse SSE event:', e, jsonStr);
        }
      }
    }
  } catch (err) {
    console.error('Streaming chat failed:', err);
    onError && onError(err);
  }
}

export async function renderDualSeq(dualseq_tuples, name = 'cad_model') {
  const res = await fetch(`${API_BASE}/dualseq/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dualseq_tuples, name }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Render request failed');
  }
  return await res.json();
}

export async function exportCAD(dualseq_tuples, format = 'stl', filename = 'cad_model') {
  if (format === 'json') {
    const res = await fetch(`${API_BASE}/cad/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dualseq_tuples, format, filename }),
    });
    if (!res.ok) throw new Error('Failed to export JSON');
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    downloadBlob(blob, `${filename}.json`);
    return true;
  } else if (format === 'stl') {
    const res = await fetch(`${API_BASE}/cad/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dualseq_tuples, format, filename }),
    });
    if (!res.ok) throw new Error('Failed to export STL');
    const blob = await res.blob();
    downloadBlob(blob, `${filename}.stl`);
    return true;
  }
  return false;
}

function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}
