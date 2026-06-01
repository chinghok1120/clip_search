// CLIP Search Web Interface - Frontend Logic

const API_BASE = '/api';

// State
const state = {
    mode: 'single',
    models: [],
    loras: [],
    datasets: [],
    currentResults: null
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await init();
    setupEventListeners();
});

// Initialize app
async function init() {
    console.log('Initializing app...');

    // Check health
    try {
        const health = await apiCall('/health');
        updateStatus(true, `${health.device.toUpperCase()}: ${health.gpu_name || 'CPU'}`);
    } catch (error) {
        updateStatus(false, 'API Offline');
        return;
    }

    // Load models first, then datasets (datasets filtering depends on models being loaded)
    await loadModels();
    await loadDatasets();
}

// API call helper
async function apiCall(endpoint, options = {}) {
    const response = await fetch(API_BASE + endpoint, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        }
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API Error');
    }

    return response.json();
}

// Update status indicator
function updateStatus(online, text) {
    const indicator = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.status-text');

    if (online) {
        indicator.classList.add('online');
        statusText.textContent = text;
    } else {
        indicator.classList.remove('online');
        statusText.textContent = text;
    }
}

// Load models
async function loadModels() {
    try {
        const data = await apiCall('/models');
        state.models = data.base_models;
        state.loras = data.lora_adapters;

        // Only populate single-mode model select here.
        // Compare mode model selects are owned by filterSlotModel (called from loadDatasets).
        const singleModelSelect = document.getElementById('model1');
        singleModelSelect.innerHTML = '<option value="">Select model...</option>';
        state.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.display_name;
            singleModelSelect.appendChild(option);
        });

        // Populate LoRA selects
        const loraSelects = ['lora1', 'compare-lora1', 'compare-lora2'];
        loraSelects.forEach(id => {
            const select = document.getElementById(id);
            select.innerHTML = '<option value="">None</option>';
            state.loras.forEach(lora => {
                const option = document.createElement('option');
                option.value = lora.id;
                option.textContent = lora.name;
                select.appendChild(option);
            });
        });

        console.log(`Loaded ${state.models.length} models, ${state.loras.length} LoRA adapters`);
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

// Populate a dataset <select> element with all available datasets
function populateDatasetSelect(selectEl) {
    selectEl.innerHTML = '';
    if (state.datasets.length === 0) {
        selectEl.innerHTML = '<option value="">No datasets found</option>';
        return;
    }
    state.datasets.forEach(dataset => {
        const option = document.createElement('option');
        option.value = dataset.id;
        option.textContent = `${dataset.name} (${dataset.num_embeddings.toLocaleString()} images)`;
        selectEl.appendChild(option);
    });
}

// Load datasets and wire up all dataset dropdowns
async function loadDatasets() {
    try {
        const data = await apiCall('/datasets');
        state.datasets = data.datasets;

        // Single mode dataset
        const singleDataset = document.getElementById('dataset');
        populateDatasetSelect(singleDataset);
        singleDataset.selectedIndex = 0;
        filterSlotModel('model1', singleDataset.value);
        singleDataset.addEventListener('change', () => filterSlotModel('model1', singleDataset.value));

        // Compare mode: each slot has its own dataset select
        document.querySelectorAll('.compare-dataset').forEach(sel => {
            populateDatasetSelect(sel);
            sel.selectedIndex = 0;
            const slot = sel.dataset.slot;
            filterSlotModel(`compare-model${slot}`, sel.value);
            sel.addEventListener('change', () => filterSlotModel(`compare-model${slot}`, sel.value));
        });

        // Default slot 2 to the second dataset if available
        const slot2Dataset = document.getElementById('compare-dataset2');
        if (state.datasets.length > 1) {
            slot2Dataset.selectedIndex = 1;
            filterSlotModel('compare-model2', slot2Dataset.value);
        }

        console.log(`Loaded ${state.datasets.length} datasets`);
    } catch (error) {
        console.error('Failed to load datasets:', error);
    }
}

// Set a model <select> to only show the model compatible with the given dataset
function filterSlotModel(modelSelectId, datasetId) {
    const dataset = state.datasets.find(d => d.id === datasetId);
    const select = document.getElementById(modelSelectId);
    if (!dataset || !select) return;

    const compatibleModelId = dataset.model;
    select.innerHTML = '';
    state.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.display_name;
        option.disabled = model.id !== compatibleModelId;
        if (option.disabled) option.textContent += ' (incompatible)';
        select.appendChild(option);
    });
    select.value = compatibleModelId;
}

// Setup event listeners
function setupEventListeners() {
    // Mode toggle
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            switchMode(mode);
        });
    });

    // Search button
    document.getElementById('search-btn').addEventListener('click', handleSearch);

    // Query input - Enter key
    document.getElementById('query').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });

    // Example queries
    document.addEventListener('click', (e) => {
        if (e.target.matches('.example-queries li')) {
            document.getElementById('query').value = e.target.textContent.replace(/"/g, '');
            handleSearch();
        }
    });

    // Modal
    const modal = document.getElementById('image-modal');
    const modalClose = document.querySelector('.modal-close');

    modalClose.addEventListener('click', () => {
        modal.classList.remove('show');
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('show');
        }
    });
}

// Switch mode
function switchMode(mode) {
    state.mode = mode;

    // Update buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Show/hide mode content
    document.getElementById('single-mode').style.display = mode === 'single' ? 'block' : 'none';
    document.getElementById('compare-mode').style.display = mode === 'compare' ? 'block' : 'none';
}

// Handle search
async function handleSearch() {
    const query = document.getElementById('query').value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }

    const topK = parseInt(document.getElementById('topk').value);

    // Validate dataset selection depending on mode
    if (state.mode === 'single') {
        const dataset = document.getElementById('dataset').value;
        if (!dataset) { alert('Please select a dataset'); return; }
    }

    // Show loading
    setSearching(true);

    try {
        if (state.mode === 'single') {
            const dataset = document.getElementById('dataset').value;
            await performSingleSearch(query, dataset, topK);
        } else {
            await performComparison(query, null, topK);
        }
    } catch (error) {
        console.error('Search error:', error);
        alert('Search failed: ' + error.message);
    } finally {
        setSearching(false);
    }
}

// Perform single model search
async function performSingleSearch(query, dataset, topK) {
    const model = document.getElementById('model1').value;
    const lora = document.getElementById('lora1').value || null;

    if (!model) {
        alert('Please select a model');
        return;
    }

    const result = await apiCall('/search', {
        method: 'POST',
        body: JSON.stringify({
            query,
            model,
            dataset,
            lora,
            top_k: topK
        })
    });

    displaySingleResults(result);
}

// Perform model comparison — each slot has its own dataset
async function performComparison(query, _unused, topK) {
    const dataset1 = document.getElementById('compare-dataset1').value;
    const model1   = document.getElementById('compare-model1').value;
    const lora1    = document.getElementById('compare-lora1').value || null;
    const dataset2 = document.getElementById('compare-dataset2').value;
    const model2   = document.getElementById('compare-model2').value;
    const lora2    = document.getElementById('compare-lora2').value || null;

    if (!dataset1 || !model1 || !dataset2 || !model2) {
        alert('Please select a dataset and model for both slots');
        return;
    }

    // Run both searches independently and combine
    const [result1, result2] = await Promise.all([
        apiCall('/search', {
            method: 'POST',
            body: JSON.stringify({ query, model: model1, dataset: dataset1, lora: lora1, top_k: topK })
        }),
        apiCall('/search', {
            method: 'POST',
            body: JSON.stringify({ query, model: model2, dataset: dataset2, lora: lora2, top_k: topK })
        })
    ]);

    displayComparisonResults({ query, comparisons: [
        { model: model1, lora: lora1, dataset: dataset1, results: result1.results, timing: result1.timing },
        { model: model2, lora: lora2, dataset: dataset2, results: result2.results, timing: result2.timing }
    ]});
}

// Display single model results
function displaySingleResults(data) {
    const container = document.getElementById('results-container');

    const modelName = state.models.find(m => m.id === data.model)?.display_name || data.model;
    const loraName = data.lora ? state.loras.find(l => l.id === data.lora)?.name : null;

    container.innerHTML = `
        <div class="results-header">
            <div class="results-info">
                <h2>"${data.query}"</h2>
                <div class="results-meta">
                    ${data.num_results} results • ${modelName}${loraName ? ` + ${loraName}` : ''}
                </div>
            </div>
            <div class="results-timing">
                <div>Text: ${data.timing.text_encoding_ms.toFixed(1)}ms</div>
                <div>Search: ${data.timing.search_ms.toFixed(1)}ms</div>
                <div><strong>Total: ${data.timing.total_ms.toFixed(1)}ms</strong></div>
            </div>
        </div>
        <div class="results-grid">
            ${data.results.map(result => createResultCard(result)).join('')}
        </div>
    `;

    // Add click handlers
    container.querySelectorAll('.result-card').forEach((card, index) => {
        card.addEventListener('click', () => showImageModal(data.results[index]));
    });
}

// Calculate score statistics for a result set
function calcStats(results) {
    if (!results.length) return { avg: 0, top1: 0, top5avg: 0 };
    const scores = results.map(r => r.score);
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const top5 = scores.slice(0, 5);
    return {
        top1:    scores[0],
        top5avg: top5.reduce((a, b) => a + b, 0) / top5.length,
        avg:     avg
    };
}

// Find image paths that appear in more than one model's results
function findOverlapPaths(comparisons) {
    const pathCounts = new Map();
    comparisons.forEach(comp => {
        const seen = new Set();
        comp.results.forEach(r => {
            if (!seen.has(r.image_path)) {
                seen.add(r.image_path);
                pathCounts.set(r.image_path, (pathCounts.get(r.image_path) || 0) + 1);
            }
        });
    });
    return new Set([...pathCounts.entries()].filter(([, count]) => count > 1).map(([path]) => path));
}

// Display comparison results
function displayComparisonResults(data) {
    const container = document.getElementById('results-container');

    // Compute overlap and stats
    const overlapPaths = findOverlapPaths(data.comparisons);
    const overlapCount = [...data.comparisons[0]?.results || []].filter(r => overlapPaths.has(r.image_path)).length;

    container.innerHTML = `
        <div class="results-header">
            <div class="results-info">
                <h2>"${data.query}"</h2>
                <div class="results-meta">Model Comparison • ${overlapCount} shared results in top-${data.comparisons[0]?.results.length || 0}</div>
            </div>
        </div>
        <div class="comparison-container">
            ${data.comparisons.map(comp => createComparisonColumn(comp, overlapPaths)).join('')}
        </div>
    `;

    // Add click handlers
    data.comparisons.forEach((comp, compIndex) => {
        const column = container.querySelectorAll('.comparison-column')[compIndex];
        column.querySelectorAll('.result-card').forEach((card, resultIndex) => {
            card.addEventListener('click', () => showImageModal(comp.results[resultIndex]));
        });
    });
}

// Create result card HTML — overlapPaths is optional (only used in compare mode)
function createResultCard(result, overlapPaths = null) {
    const isOverlap = overlapPaths && overlapPaths.has(result.image_path);
    return `
        <div class="result-card${isOverlap ? ' result-overlap' : ''}" data-rank="${result.rank}">
            ${isOverlap ? '<div class="overlap-badge">Both</div>' : ''}
            <img class="result-image" src="${result.thumbnail_url}" alt="Result ${result.rank}">
            <div class="result-info">
                <div class="result-rank">#${result.rank}</div>
                <div class="result-score">${(result.score * 100).toFixed(1)}%</div>
            </div>
        </div>
    `;
}

// Create comparison column HTML with stats bar and overlap badges
function createComparisonColumn(comp, overlapPaths = null) {
    const modelName = state.models.find(m => m.id === comp.model)?.display_name || comp.model;
    const loraName = comp.lora ? state.loras.find(l => l.id === comp.lora)?.name : null;
    const datasetName = comp.dataset
        ? (state.datasets.find(d => d.id === comp.dataset)?.name || comp.dataset)
        : '';

    const stats = calcStats(comp.results);
    const overlapCount = overlapPaths
        ? comp.results.filter(r => overlapPaths.has(r.image_path)).length
        : 0;

    return `
        <div class="comparison-column">
            <div class="comparison-header">
                <h3>${modelName}${loraName ? ` + ${loraName}` : ''}</h3>
                <div class="comparison-meta">
                    ${datasetName ? `${datasetName} • ` : ''}${comp.timing.total_ms.toFixed(1)}ms
                </div>
            </div>

            <div class="stats-bar">
                <div class="stat">
                    <span class="stat-label">Top-1</span>
                    <span class="stat-value">${(stats.top1 * 100).toFixed(1)}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Top-5 avg</span>
                    <span class="stat-value">${(stats.top5avg * 100).toFixed(1)}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">All avg</span>
                    <span class="stat-value">${(stats.avg * 100).toFixed(1)}%</span>
                </div>
                ${overlapPaths ? `
                <div class="stat stat-overlap">
                    <span class="stat-label">Shared</span>
                    <span class="stat-value">${overlapCount}/${comp.results.length}</span>
                </div>` : ''}
            </div>

            <div class="comparison-grid">
                ${comp.results.map(result => createResultCard(result, overlapPaths)).join('')}
            </div>
        </div>
    `;
}

// Show image modal
function showImageModal(result) {
    const modal = document.getElementById('image-modal');
    const img = document.getElementById('modal-image');
    const rank = document.getElementById('modal-rank');
    const score = document.getElementById('modal-score');
    const path = document.getElementById('modal-path');

    img.src = result.image_url;
    rank.textContent = `#${result.rank}`;
    score.textContent = `${(result.score * 100).toFixed(2)}%`;
    path.textContent = result.image_path;

    modal.classList.add('show');
}

// Set searching state
function setSearching(searching) {
    const btn = document.getElementById('search-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');

    btn.disabled = searching;
    btnText.style.display = searching ? 'none' : 'inline';
    btnLoading.style.display = searching ? 'inline' : 'none';

    if (searching) {
        const container = document.getElementById('results-container');
        container.innerHTML = `
            <div class="loading">
                <div class="loading-spinner"></div>
                <p>Searching...</p>
            </div>
        `;
    }
}
