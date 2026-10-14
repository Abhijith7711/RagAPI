/* ================================================================
   RAG Knowledge Assistant — Client Application
   Vanilla JS – handles uploads, chat, and settings
   ================================================================ */

const $ = (id) => document.getElementById(id);

const state = { documents: [] };

/* ---- Notifications ---- */

function notify(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `notification ${type}`;
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => {
        el.style.animation = 'fadeOut 200ms ease forwards';
        setTimeout(() => el.remove(), 200);
    }, 2500);
}

/* ---- API Calls ---- */

async function uploadFile(file) {
    const form = new FormData();
    form.append('file', file);
    try {
        const res = await fetch('/documents', { method: 'POST', body: form });
        const data = await res.json();
        if (res.ok) {
            addDocument(data);
            notify(
                `${data.filename}: ${data.status}`,
                data.is_duplicate ? 'info' : 'success',
            );
        } else {
            notify(data.detail || 'Upload failed', 'error');
        }
    } catch {
        notify('Upload failed — server unreachable', 'error');
    }
}


async function askQuestion(question) {
    if (!question.trim()) return;
    appendMessage('user', question);
    showTyping();

    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question.trim() }),
        });
        removeTyping();
        const data = await res.json();
        appendMessage(res.ok ? 'assistant' : 'error', res.ok ? data : data.detail || 'Request failed');
    } catch {
        removeTyping();
        appendMessage('error', 'Could not reach the server');
    }
}

async function checkReady() {
    try {
        const res = await fetch('/ready');
        const data = await res.json();
        const dot = $('statusDot');
        dot.classList.toggle('ready', data.status === 'ready');
        dot.title = data.status === 'ready' ? 'Ready' : 'Not ready';
    } catch {
        /* ignore */
    }
}

/* ---- Document List ---- */

function addDocument(doc) {
    if (state.documents.some((d) => d.document_id === doc.document_id)) return;
    state.documents.push(doc);
    renderDocList();
}

function renderDocList() {
    const el = $('docList');
    if (!state.documents.length) {
        el.innerHTML = '<p class="empty-state">No documents indexed</p>';
        return;
    }
    el.innerHTML = state.documents
        .map(
            (d) =>
                `<div class="doc-item">
                    <span class="doc-name" title="${esc(d.filename)}">${esc(d.filename)}</span>
                    <span class="doc-badge">${d.chunk_count} chunks</span>
                </div>`,
        )
        .join('');
}

/* ---- Chat Rendering ---- */

function appendMessage(role, data) {
    const container = $('messages');
    const welcome = container.querySelector('.welcome');
    if (welcome) welcome.remove();

    const div = document.createElement('div');
    div.className = `message ${role === 'error' ? 'assistant' : role}`;

    if (role === 'user') {
        div.innerHTML = `<div class="bubble"><p>${esc(data)}</p></div>`;
    } else if (role === 'assistant') {
        let html = `<div class="bubble"><p>${esc(data.answer)}</p>`;
        html += data.found_in_documents
            ? '<span class="badge found">Found in documents</span>'
            : '<span class="badge not-found">Not found in documents</span>';

        if (data.sources && data.sources.length) {
            html += '<div class="sources">';
            data.sources.forEach((s) => {
                const page = s.page ? ` &middot; Page ${s.page}` : '';
                html += `<div class="source">
                    <div class="source-header">${esc(s.document)}${page}</div>
                    <p class="source-excerpt">${esc(truncate(s.excerpt, 220))}</p>
                </div>`;
            });
            html += '</div>';
        }
        html += '</div>';
        div.innerHTML = html;
    } else {
        div.innerHTML = `<div class="bubble error-bubble"><p>${esc(data)}</p></div>`;
    }

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function showTyping() {
    const container = $('messages');
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'typing';
    div.innerHTML =
        '<div class="bubble typing-dots"><span></span><span></span><span></span></div>';
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function removeTyping() {
    const el = $('typing');
    if (el) el.remove();
}

/* ---- Helpers ---- */

function esc(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

function truncate(s, n) {
    return s.length > n ? s.slice(0, n) + '…' : s;
}

/* ---- Event Listeners ---- */

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = $('dropZone');
    const fileInput = $('fileInput');

    /* File upload — drag & drop + click */
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        [...e.dataTransfer.files].forEach(uploadFile);
    });
    fileInput.addEventListener('change', () => {
        [...fileInput.files].forEach(uploadFile);
        fileInput.value = '';
    });

    /* Chat form */
    $('inputForm').addEventListener('submit', (e) => {
        e.preventDefault();
        const input = $('questionInput');
        askQuestion(input.value);
        input.value = '';
        input.style.height = 'auto';
    });


    /* Auto-resize textarea */
    const textarea = $('questionInput');
    textarea.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    /* Enter to send (Shift+Enter for newline) */
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            $('inputForm').dispatchEvent(new Event('submit'));
        }
    });

    /* Initial status check */
    checkReady();
});
