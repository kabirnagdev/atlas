document.addEventListener('DOMContentLoaded', () => {
  const API_BASE = window.location.origin;

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const selectedFilePill = document.getElementById('selectedFilePill');
  
  const pageScope = document.getElementById('pageScope');
  const questionInput = document.getElementById('questionInput');
  const submitBtn = document.getElementById('submitBtn');
  const modeSelect = document.getElementById('modeSelect');
  
  const docLibraryList = document.getElementById('docLibraryList');
  const outputContainer = document.getElementById('outputContainer');
  const sourcesBlock = document.getElementById('sourcesBlock');
  const sourcesList = document.getElementById('sourcesList');

  const guideBtn = document.getElementById('guideBtn');
  const guideModal = document.getElementById('guideModal');
  const closeGuideModalBtn = document.getElementById('closeGuideModalBtn');

  // Initial load: fetch PDF library
  fetchLibraryDocuments();

  // 1. Fetch & Render PDF Library List
  async function fetchLibraryDocuments() {
    try {
      const res = await fetch(`${API_BASE}/api/documents`);
      if (res.ok) {
        const data = await res.json();
        renderLibraryDocuments(data.documents || []);
      }
    } catch (err) {
      console.error('Failed to fetch document library', err);
    }
  }

  function renderLibraryDocuments(documents) {
    if (!documents || documents.length === 0) {
      docLibraryList.innerHTML = `<p class="empty-state" style="font-size: 0.85rem;">No PDFs in library. Upload a PDF to add it to your context library.</p>`;
      return;
    }

    docLibraryList.innerHTML = documents.map(doc => `
      <div class="doc-library-item ${doc.is_active ? 'active' : ''}">
        <label class="doc-library-checkbox-label">
          <input 
            type="checkbox" 
            class="doc-library-checkbox" 
            data-filename="${escapeAttr(doc.filename)}" 
            ${doc.is_active ? 'checked' : ''}
          >
          <div class="doc-info-group">
            <span class="doc-name" title="${escapeAttr(doc.filename)}">${escapeHtml(doc.filename)}</span>
            <span class="doc-meta">${doc.file_size} • ${doc.page_count || 1} pages • ${doc.total_chunks || 0} chunks</span>
          </div>
        </label>
        <button class="btn-delete-doc" data-filename="${escapeAttr(doc.filename)}" title="Permanently delete from disk & context">Delete</button>
      </div>
    `).join('');

    // Attach checkbox toggle handlers
    document.querySelectorAll('.doc-library-checkbox').forEach(cb => {
      cb.addEventListener('change', (e) => {
        const filename = cb.dataset.filename;
        const isChecked = cb.checked;
        toggleDocumentActive(filename, isChecked);
      });
    });

    // Attach delete button handlers
    document.querySelectorAll('.btn-delete-doc').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const filename = btn.dataset.filename;
        deleteDocumentFromLibrary(filename);
      });
    });
  }

  function escapeHtml(str) {
    return (str || '')
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function escapeAttr(str) {
    return (str || '').replace(/"/g, '&quot;');
  }

  // 2. Toggle Active Checkbox Status
  async function toggleDocumentActive(filename, isActive) {
    try {
      const res = await fetch(`${API_BASE}/api/documents/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, is_active: isActive })
      });
      if (res.ok) {
        const data = await res.json();
        renderLibraryDocuments(data.library || []);
      }
    } catch (err) {
      alert('Failed to toggle document context status.');
    }
  }

  // 3. Permanently Delete PDF from Library & Disk
  async function deleteDocumentFromLibrary(filename) {
    if (!confirm(`Are you sure you want to permanently delete '${filename}' from disk and context library?`)) return;

    try {
      const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        const data = await res.json();
        renderLibraryDocuments(data.library || []);
        sourcesBlock.style.display = 'none';
      }
    } catch (err) {
      alert('Failed to delete document from library.');
    }
  }

  // 4. File Dropzone & Ingestion
  dropzone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please select a PDF document.');
      return;
    }

    selectedFilePill.textContent = `Uploaded: ${file.name}`;
    selectedFilePill.style.display = 'inline-flex';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/upload-pdf`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        fetchLibraryDocuments();
      }
    } catch (err) {
      console.error('PDF upload error:', err);
    }
  }

  // 5. Submit Question Handler
  submitBtn.addEventListener('click', async () => {
    const question = questionInput.value.trim();
    if (!question) {
      alert('Please type a question before submitting.');
      return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Generating Answer...';
    outputContainer.innerHTML = '<p class="empty-state">Processing query across active PDF contexts...</p>';

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: question,
          page_selection: pageScope.value.trim() || null,
          context_type: 'notebook_lm'
        })
      });

      const data = await res.json();
      renderResponseData(data);

    } catch (err) {
      outputContainer.innerHTML = '<p style="color: #dc2626;">Failed to connect to backend server.</p>';
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Ask Question';
    }
  });

  function renderResponseData(data) {
    let text = data.output_text
      .replace(/### /g, '<h3>')
      .replace(/#### /g, '<h4>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/🎙️|📓|🔍|🔑|✨|⚡|🚀|ℹ️|📄|📥|✂️|🧬|🔎|🤖/g, '')
      .replace(/\n/g, '<br>');

    outputContainer.innerHTML = text;

    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
      sourcesBlock.style.display = 'block';
      sourcesList.innerHTML = data.retrieved_chunks.map(chunk => `
        <div class="source-card">
          <div class="source-card-header">
            <span class="page-tag">Page ${chunk.page_number}</span>
            <span class="score-tag">Match: ${(chunk.similarity_score * 100).toFixed(1)}%</span>
          </div>
          <div class="source-card-text">${escapeHtml(chunk.content)}</div>
        </div>
      `).join('');
    } else {
      sourcesBlock.style.display = 'none';
    }
  }

  // Engine Mode Selector
  modeSelect.addEventListener('change', async (e) => {
    const newMode = e.target.value;
    try {
      await fetch(`${API_BASE}/api/toggle-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: newMode })
      });
      fetchLibraryDocuments();
    } catch (err) {
      console.error('Could not switch model mode', err);
    }
  });

  // Modal Handler
  guideBtn.addEventListener('click', () => guideModal.classList.add('active'));
  closeGuideModalBtn.addEventListener('click', () => guideModal.classList.remove('active'));
});
