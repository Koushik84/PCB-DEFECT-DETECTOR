document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const masterInput = document.getElementById('master-input');
    const testInput = document.getElementById('test-input');
    const masterDropzone = document.getElementById('master-dropzone');
    const testDropzone = document.getElementById('test-dropzone');
    const masterPreview = document.getElementById('master-preview');
    const testPreview = document.getElementById('test-preview');
    const removeMasterBtn = document.getElementById('remove-master');
    const removeTestBtn = document.getElementById('remove-test');

    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdVal = document.getElementById('threshold-val');
    const areaSlider = document.getElementById('area-slider');
    const areaVal = document.getElementById('area-val');
    const inspectBtn = document.getElementById('inspect-btn');

    const stateInitial = document.getElementById('state-initial');
    const stateScanning = document.getElementById('state-scanning');
    const stateResults = document.getElementById('state-results');

    const scanMasterBox = document.getElementById('scan-master-box');
    const scanTestBox = document.getElementById('scan-test-box');

    const headerStats = document.getElementById('header-stats');
    const latencyVal = document.getElementById('latency-val');

    const verdictBanner = document.getElementById('verdict-banner');
    const verdictBadge = document.getElementById('verdict-badge');
    const verdictText = document.getElementById('verdict-text');
    const defectCountSummary = document.getElementById('defect-count-summary');

    const resMaster = document.getElementById('res-master');
    const resTest = document.getElementById('res-test');
    const resHeatmap = document.getElementById('res-heatmap');
    const resBbox = document.getElementById('res-bbox');

    const defectsList = document.getElementById('defects-list');
    const detailsCount = document.getElementById('details-count');
    const noDefectsMsg = document.getElementById('no-defects-msg');

    // State variables
    let masterFile = null;
    let testFile = null;
    let isInspecting = false;
    let isUpdatingSliders = false;

    // --- File Drag & Drop Handlers ---
    setupDropzone(masterDropzone, masterInput, (file) => {
        masterFile = file;
        showPreview(file, masterPreview, removeMasterBtn, masterDropzone);
        checkInputs();
    });

    setupDropzone(testDropzone, testInput, (file) => {
        testFile = file;
        showPreview(file, testPreview, removeTestBtn, testDropzone);
        checkInputs();
    });

    removeMasterBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        masterFile = null;
        masterInput.value = '';
        masterPreview.style.display = 'none';
        removeMasterBtn.style.display = 'none';
        masterDropzone.querySelector('.upload-placeholder').style.opacity = '1';
        checkInputs();
    });

    removeTestBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        testFile = null;
        testInput.value = '';
        testPreview.style.display = 'none';
        removeTestBtn.style.display = 'none';
        testDropzone.querySelector('.upload-placeholder').style.opacity = '1';
        checkInputs();
    });

    function setupDropzone(dropzone, input, onFileSelect) {
        dropzone.addEventListener('click', () => input.click());

        input.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                onFileSelect(e.target.files[0]);
            }
        });

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        ['dragleave', 'dragend'].forEach(type => {
            dropzone.addEventListener(type, () => {
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.type.startsWith('image/')) {
                    onFileSelect(file);
                }
            }
        });
    }

    function showPreview(file, previewImg, removeBtn, dropzone) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            removeBtn.style.display = 'flex';
            dropzone.querySelector('.upload-placeholder').style.opacity = '0';
        };
        reader.readAsDataURL(file);
    }

    function checkInputs() {
        inspectBtn.disabled = !(masterFile && testFile) || isInspecting;
    }

    // --- Slider Event Listeners ---
    thresholdSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value).toFixed(2);
        thresholdVal.textContent = val;
        debounceSliderUpdate();
    });

    areaSlider.addEventListener('input', (e) => {
        areaVal.textContent = `${e.target.value} px`;
        debounceSliderUpdate();
    });

    // Debounce slider updates to prevent spamming backend during dragging
    let sliderDebounceTimeout;
    function debounceSliderUpdate() {
        if (!stateResults.style.display || stateResults.style.display === 'none') {
            return; // Only update if results are currently visible
        }
        clearTimeout(sliderDebounceTimeout);
        sliderDebounceTimeout = setTimeout(() => {
            updateParameters();
        }, 150);
    }

    // --- Trigger Inspection API ---
    inspectBtn.addEventListener('click', async () => {
        if (isInspecting || !masterFile || !testFile) return;

        isInspecting = true;
        inspectBtn.disabled = true;

        // Set scanning backgrounds
        scanMasterBox.style.backgroundImage = `url(${masterPreview.src})`;
        scanTestBox.style.backgroundImage = `url(${testPreview.src})`;

        // Toggle states
        stateInitial.style.display = 'none';
        stateResults.style.display = 'none';
        stateScanning.style.display = 'flex';
        headerStats.style.display = 'none';

        const startTime = performance.now();

        const formData = new FormData();
        formData.append('master', masterFile);
        formData.append('test', testFile);
        formData.append('threshold', thresholdSlider.value);
        formData.append('min_area', areaSlider.value);

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Inspection failed');
            }

            const data = await response.json();
            const endTime = performance.now();
            const latency = ((endTime - startTime) / 1000).toFixed(2);

            // Display latency
            latencyVal.textContent = `${latency}s`;

            // Populate Results
            renderResults(data);

            // Move to results state
            stateScanning.style.display = 'none';
            stateResults.style.display = 'flex';
            headerStats.style.display = 'flex';

        } catch (error) {
            console.error(error);
            alert(`Error: ${error.message}`);
            stateScanning.style.display = 'none';
            stateInitial.style.display = 'flex';
        } finally {
            isInspecting = false;
            checkInputs();
        }
    });

    // --- Update Parameters API ---
    async function updateParameters() {
        if (isUpdatingSliders) return;
        isUpdatingSliders = true;

        const formData = new FormData();
        formData.append('threshold', thresholdSlider.value);
        formData.append('min_area', areaSlider.value);

        try {
            const response = await fetch('/api/update_threshold', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to update threshold');
            }

            const data = await response.json();
            
            // Update BBox image
            resBbox.src = data.bbox_img;

            // Update Verdict and Table
            updateVerdictUI(data.verdict, data.anomalies.length);
            renderDefectsTable(data.anomalies);

        } catch (error) {
            console.error('Threshold update failed:', error);
        } finally {
            isUpdatingSliders = false;
        }
    }

    // --- Helper Renderers ---
    function renderResults(data) {
        resMaster.src = data.master;
        resTest.src = data.test;
        resHeatmap.src = data.heatmap;
        resBbox.src = data.bbox_img;

        updateVerdictUI(data.verdict, data.anomalies.length);
        renderDefectsTable(data.anomalies);
    }

    function updateVerdictUI(verdict, count) {
        if (verdict === 'PASSED') {
            verdictBanner.className = 'verdict-banner passed';
            verdictBadge.textContent = 'PASSED';
            verdictText.textContent = 'Inspection Passed: No anomalies detected';
            defectCountSummary.textContent = 'Trace paths match master template';
        } else {
            verdictBanner.className = 'verdict-banner defect';
            verdictBadge.textContent = 'DEFECT';
            verdictText.textContent = 'Anomaly Detected in Target Board';
            defectCountSummary.textContent = `${count} defect ${count === 1 ? 'area' : 'areas'} highlighted`;
        }
    }

    function renderDefectsTable(anomalies) {
        defectsList.innerHTML = '';
        detailsCount.textContent = `${anomalies.length} ${anomalies.length === 1 ? 'defect' : 'defects'}`;

        if (anomalies.length === 0) {
            noDefectsMsg.style.display = 'flex';
            return;
        }

        noDefectsMsg.style.display = 'none';

        anomalies.forEach((anomaly) => {
            const tr = document.createElement('tr');
            
            const position = `X: ${anomaly.x}, Y: ${anomaly.y}`;
            const dimensions = `${anomaly.w} &times; ${anomaly.h} px`;
            const area = `${anomaly.area} px&sup2;`;

            tr.innerHTML = `
                <td><span class="defect-id-badge">#${anomaly.id}</span></td>
                <td>${position}</td>
                <td>${dimensions}</td>
                <td>${area}</td>
                <td><span class="status-indicator">Active</span></td>
            `;

            // Hover effect to simulate highlighting
            tr.addEventListener('mouseenter', () => {
                tr.classList.add('selected');
            });
            tr.addEventListener('mouseleave', () => {
                tr.classList.remove('selected');
            });

            defectsList.appendChild(tr);
        });
    }
});
