// Global variables
let selectedVoiceId = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordedBlob = null;
let allVoices = [];
let allAudioFiles = [];
let deleteCallback = null;

// API Base URL
const API_BASE = '/api';

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    loadVoices();
    loadAudioLibraryCount();
    setupEventListeners();
    initializeTheme();
    checkModelStatus();
});

// Check model status
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();

        if (!data.model_loaded) {
            showToast('Model is loading, this may take a few minutes...', 'warning');
        }
    } catch (error) {
        console.error('Failed to check model status:', error);
    }
}

// Theme Management
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const icon = document.getElementById('theme-icon');
    icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
}

// Event Listeners
function setupEventListeners() {
    // Text input listeners
    const textInput = document.getElementById('text-input');
    textInput.addEventListener('input', updateTextStats);

    // CFG Scale slider
    const cfgSlider = document.getElementById('cfg-scale');
    cfgSlider.addEventListener('input', (e) => {
        document.getElementById('cfg-value').textContent = e.target.value;
    });

    // Voice file upload
    const voiceUploadInput = document.getElementById('voice-upload');
    voiceUploadInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const uploadArea = document.getElementById('upload-drop');
            uploadArea.querySelector('p').textContent = `Selected: ${file.name}`;
        }
    });

    // File upload drag & drop
    const uploadArea = document.getElementById('upload-drop');
    uploadArea.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') {
            return;
        }
        document.getElementById('voice-upload').click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragging');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragging');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragging');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const voiceUploadInput = document.getElementById('voice-upload');
            voiceUploadInput.files = files;
            uploadArea.querySelector('p').textContent = `Selected: ${files[0].name}`;
        }
    });

    // Text file upload
    document.getElementById('text-file').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            document.getElementById('file-name').textContent = `Selected: ${file.name}`;
        }
    });

    // Delete modal confirm button
    document.getElementById('confirm-delete-btn').addEventListener('click', () => {
        if (deleteCallback) {
            deleteCallback();
            closeDeleteModal();
        }
    });
}

// Tab Switching
function switchTab(tab) {
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    document.getElementById(`${tab}-tab`).style.display = 'block';

    const clickedBtn = Array.from(document.querySelectorAll('.tab-btn')).find(btn =>
        btn.textContent.toLowerCase().includes(tab)
    );
    if (clickedBtn) {
        clickedBtn.classList.add('active');
    }
}

function switchTextTab(tab) {
    document.querySelectorAll('.text-tab-content').forEach(content => {
        content.style.display = 'none';
    });

    document.querySelectorAll('.text-tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    document.getElementById(`${tab}-tab`).style.display = 'block';

    const clickedBtn = Array.from(document.querySelectorAll('.text-tab-btn')).find(btn =>
        btn.textContent.toLowerCase().includes(tab === 'manual' ? 'manual' : 'file')
    );
    if (clickedBtn) {
        clickedBtn.classList.add('active');
    }
}

// Voice Management
async function loadVoices(searchQuery = '') {
    try {
        const url = searchQuery ? `${API_BASE}/voices?search=${encodeURIComponent(searchQuery)}` : `${API_BASE}/voices`;
        const response = await fetch(url);
        const voices = await response.json();
        allVoices = voices;
        displayVoices(voices);

        // Select first voice if none selected
        if (!selectedVoiceId && voices.length > 0) {
            selectVoice(voices[0].id);
        } else if (selectedVoiceId) {
            // Re-select current voice to maintain selection
            const voiceCard = document.querySelector(`[data-voice-id="${selectedVoiceId}"]`);
            if (voiceCard) {
                voiceCard.classList.add('selected');
            }
        }

    } catch (error) {
        showToast('Failed to load voices', 'error');
        console.error(error);
    }
}

function displayVoices(voices) {
    const voiceList = document.getElementById('voice-list');
    const noVoices = document.getElementById('no-voices');

    if (voices.length === 0) {
        voiceList.style.display = 'none';
        noVoices.style.display = 'block';
        return;
    }

    voiceList.style.display = 'grid';
    noVoices.style.display = 'none';
    voiceList.innerHTML = '';

    voices.forEach(voice => {
        const voiceCard = document.createElement('div');
        voiceCard.className = 'voice-card';
        voiceCard.dataset.voiceId = voice.id;
        voiceCard.onclick = (e) => {
            if (!e.target.classList.contains('delete-btn') && !e.target.closest('.delete-btn')) {
                selectVoice(voice.id);
            }
        };

        const icon = voice.type === 'recorded' ? 'fa-microphone' :
            voice.type === 'uploaded' ? 'fa-upload' : 'fa-user-circle';

        voiceCard.innerHTML = `
            <button class="delete-btn" onclick="confirmDeleteVoice('${voice.id}', '${voice.name}')" title="Delete Voice">
                <i class="fas fa-trash"></i>
            </button>
            <i class="fas ${icon} voice-icon"></i>
            <div class="voice-name">${voice.name}</div>
            <div class="voice-type">${voice.type}</div>
        `;

        voiceList.appendChild(voiceCard);
    });
}

function searchVoices() {
    const searchQuery = document.getElementById('voice-search').value;
    loadVoices(searchQuery);
}

function refreshVoices() {
    document.getElementById('voice-search').value = '';
    loadVoices();
    showToast('Voices refreshed', 'success');
}

function selectVoice(voiceId) {
    selectedVoiceId = voiceId;

    document.querySelectorAll('.voice-card').forEach(card => {
        card.classList.remove('selected');
    });

    const selectedCard = document.querySelector(`[data-voice-id="${voiceId}"]`);
    if (selectedCard) {
        selectedCard.classList.add('selected');
    }
}

function confirmDeleteVoice(voiceId, voiceName) {
    document.getElementById('delete-message').textContent = `Are you sure you want to delete the voice "${voiceName}"?`;
    deleteCallback = () => deleteVoice(voiceId);
    document.getElementById('delete-modal').style.display = 'flex';
}

async function deleteVoice(voiceId) {
    showLoading('Deleting voice...');

    try {
        const response = await fetch(`${API_BASE}/voices/${voiceId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            showToast('Voice deleted successfully', 'success');

            // Clear selection if deleted voice was selected
            if (selectedVoiceId === voiceId) {
                selectedVoiceId = null;
            }

            await loadVoices();
        } else {
            showToast('Failed to delete voice', 'error');
        }
    } catch (error) {
        showToast('Delete error', 'error');
        console.error(error);
    } finally {
        hideLoading();
    }
}

async function uploadVoice() {
    const fileInput = document.getElementById('voice-upload');
    const nameInput = document.getElementById('upload-name');

    if (!fileInput.files || !fileInput.files.length) {
        showToast('Please select an audio file', 'warning');
        return;
    }

    const voiceName = nameInput.value.trim();
    if (!voiceName) {
        showToast('Please enter a voice name', 'warning');
        return;
    }

    const file = fileInput.files[0];
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('File size must be less than 50MB', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', voiceName);

    showLoading('Uploading voice...');

    try {
        const response = await fetch(`${API_BASE}/voices/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const result = await response.json();

        if (result.success) {
            showToast('Voice uploaded successfully', 'success');
            await loadVoices();

            fileInput.value = '';
            nameInput.value = '';
            document.getElementById('upload-drop').querySelector('p').textContent = 'Drag & drop audio file or click to browse';

            if (result.voice && result.voice.id) {
                selectVoice(result.voice.id);
            }
        } else {
            showToast(result.message || 'Upload failed', 'error');
        }

    } catch (error) {
        showToast(`Upload error: ${error.message}`, 'error');
        console.error('Upload error:', error);
    } finally {
        hideLoading();
    }
}

// Recording
async function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, {type: 'audio/webm'});
            recordedBlob = audioBlob;
            document.getElementById('save-recording').style.display = 'inline-block';
        };

        mediaRecorder.start(100);
        isRecording = true;

        const recordBtn = document.getElementById('record-btn');
        recordBtn.classList.add('recording');
        document.getElementById('record-status').textContent = 'Recording... Click to stop';

        startVisualizer(stream);

    } catch (error) {
        showToast('Microphone access denied', 'error');
        console.error(error);
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;

        const recordBtn = document.getElementById('record-btn');
        recordBtn.classList.remove('recording');
        document.getElementById('record-status').textContent = 'Recording complete';
    }
}

async function saveRecording() {
    const nameInput = document.getElementById('record-name');
    const voiceName = nameInput.value.trim();

    if (!voiceName) {
        showToast('Please enter a voice name', 'warning');
        return;
    }

    if (!recordedBlob) {
        showToast('No recording available', 'warning');
        return;
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
        const base64Audio = reader.result.split(',')[1];

        showLoading('Saving recording...');

        try {
            const response = await fetch(`${API_BASE}/voices/record`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: voiceName,
                    audio_data: base64Audio,
                    format: 'webm'
                })
            });

            const result = await response.json();

            if (result.success) {
                showToast('Recording saved successfully', 'success');
                await loadVoices();

                nameInput.value = '';
                document.getElementById('save-recording').style.display = 'none';
                document.getElementById('record-status').textContent = 'Click to start recording';
                recordedBlob = null;

                if (result.voice && result.voice.id) {
                    selectVoice(result.voice.id);
                }
            } else {
                showToast('Save failed', 'error');
            }

        } catch (error) {
            showToast('Save error', 'error');
            console.error(error);
        } finally {
            hideLoading();
        }
    };

    reader.readAsDataURL(recordedBlob);
}

function startVisualizer(stream) {
    const canvas = document.getElementById('waveform');
    const ctx = canvas.getContext('2d');
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);

    source.connect(analyser);
    analyser.fftSize = 256;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    function draw() {
        if (!isRecording) return;

        requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--bg-secondary');
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;

            const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
            gradient.addColorStop(0, '#667eea');
            gradient.addColorStop(1, '#764ba2');

            ctx.fillStyle = gradient;
            ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

            x += barWidth + 1;
        }
    }

    draw();
}

// Audio Library Management
async function loadAudioLibraryCount() {
    try {
        const response = await fetch(`${API_BASE}/audio/library`);
        const data = await response.json();

        if (data.success && data.total > 0) {
            const badge = document.getElementById('library-count');
            badge.textContent = data.total;
            badge.style.display = 'flex';
        }
    } catch (error) {
        console.error('Failed to load audio library count:', error);
    }
}

async function openAudioLibrary() {
    document.getElementById('audio-library-modal').style.display = 'flex';
    await loadAudioLibrary();
}

function closeAudioLibrary() {
    document.getElementById('audio-library-modal').style.display = 'none';
}

async function loadAudioLibrary(searchQuery = '') {
    showLoading('Loading audio library...');

    try {
        const url = searchQuery ?
            `${API_BASE}/audio/library?search=${encodeURIComponent(searchQuery)}` :
            `${API_BASE}/audio/library`;

        const response = await fetch(url);
        const data = await response.json();

        if (data.success) {
            allAudioFiles = data.audio_files;
            displayAudioLibrary(data.audio_files);

            // Update badge count
            const badge = document.getElementById('library-count');
            if (data.total > 0) {
                badge.textContent = data.total;
                badge.style.display = 'flex';
            } else {
                badge.style.display = 'none';
            }
        }
    } catch (error) {
        showToast('Failed to load audio library', 'error');
        console.error(error);
    } finally {
        hideLoading();
    }
}

function displayAudioLibrary(audioFiles) {
    const libraryList = document.getElementById('audio-library-list');
    const noAudio = document.getElementById('no-audio');

    if (audioFiles.length === 0) {
        libraryList.style.display = 'none';
        noAudio.style.display = 'block';
        return;
    }

    libraryList.style.display = 'grid';
    noAudio.style.display = 'none';
    libraryList.innerHTML = '';

    audioFiles.forEach(audio => {
        const audioItem = document.createElement('div');
        audioItem.className = 'audio-item';

        const createdDate = new Date(audio.created_at).toLocaleDateString();
        const fileSize = (audio.size / 1024).toFixed(1) + ' KB';

        audioItem.innerHTML = `
            <div class="audio-item-icon">
                <i class="fas fa-file-audio"></i>
            </div>
            <div class="audio-item-details">
                <div class="audio-item-name">${audio.filename}</div>
                <div class="audio-item-meta">
                    <span><i class="fas fa-microphone"></i> ${audio.voice_name}</span>
                    <span><i class="fas fa-clock"></i> ${audio.duration.toFixed(1)}s</span>
                    <span><i class="fas fa-hdd"></i> ${fileSize}</span>
                    <span><i class="fas fa-calendar"></i> ${createdDate}</span>
                </div>
                ${audio.text_preview ? `<div class="audio-item-preview">"${audio.text_preview}"</div>` : ''}
            </div>
            <div class="audio-item-actions">
                <button class="btn-icon" onclick="playAudioFile('${audio.filename}')" title="Play">
                    <i class="fas fa-play"></i>
                </button>
                <button class="btn-icon" onclick="downloadAudioFile('${audio.filename}')" title="Download">
                    <i class="fas fa-download"></i>
                </button>
                <button class="btn-icon" onclick="confirmDeleteAudio('${audio.filename}')" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;

        libraryList.appendChild(audioItem);
    });
}

function searchAudioLibrary() {
    const searchQuery = document.getElementById('audio-search').value;
    const filtered = allAudioFiles.filter(audio => {
        const query = searchQuery.toLowerCase();
        return audio.filename.toLowerCase().includes(query) ||
            audio.voice_name.toLowerCase().includes(query) ||
            (audio.text_preview && audio.text_preview.toLowerCase().includes(query));
    });
    displayAudioLibrary(filtered);
}

function refreshAudioLibrary() {
    document.getElementById('audio-search').value = '';
    loadAudioLibrary();
    showToast('Audio library refreshed', 'success');
}

function playAudioFile(filename) {
    const audio = new Audio(`${API_BASE}/audio/${filename}`);
    audio.play();
}

function downloadAudioFile(filename) {
    const link = document.createElement('a');
    link.href = `${API_BASE}/audio/${filename}`;
    link.download = filename;
    link.click();
}

function confirmDeleteAudio(filename) {
    document.getElementById('delete-message').textContent = `Are you sure you want to delete "${filename}"?`;
    deleteCallback = () => deleteAudioFile(filename);
    document.getElementById('delete-modal').style.display = 'flex';
}

async function deleteAudioFile(filename) {
    showLoading('Deleting audio file...');

    try {
        const response = await fetch(`${API_BASE}/audio/${filename}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            showToast('Audio file deleted successfully', 'success');
            await loadAudioLibrary();
            await loadAudioLibraryCount();
        } else {
            showToast('Failed to delete audio file', 'error');
        }
    } catch (error) {
        showToast('Delete error', 'error');
        console.error(error);
    } finally {
        hideLoading();
    }
}

// Delete Modal
function closeDeleteModal() {
    document.getElementById('delete-modal').style.display = 'none';
    deleteCallback = null;
}

// Text Management
function updateTextStats() {
    const text = document.getElementById('text-input').value;
    const charCount = text.length;
    const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

    document.getElementById('char-count').textContent = `${charCount} characters`;
    document.getElementById('word-count').textContent = `${wordCount} words`;
}

// Speech Generation
async function generateSpeech() {
    if (!selectedVoiceId) {
        showToast('Please select a voice', 'warning');
        return;
    }

    let text = '';
    const textFileInput = document.getElementById('text-file');

    const fileTab = document.getElementById('file-tab');
    if (fileTab.style.display !== 'none' && textFileInput.files.length > 0) {
        await generateFromFile();
        return;
    } else {
        text = document.getElementById('text-input').value.trim();
    }

    if (!text) {
        showToast('Please enter some text', 'warning');
        return;
    }

    const cfgScale = parseFloat(document.getElementById('cfg-scale').value);
    const numSpeakers = parseInt(document.getElementById('num-speakers').value);

    showLoading('Generating speech... This may take a moment...');

    try {
        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                voice_id: selectedVoiceId,
                num_speakers: numSpeakers,
                cfg_scale: cfgScale
            })
        });

        const result = await response.json();

        if (result.success) {
            showToast('Speech generated successfully', 'success');
            displayAudio(result);
            await loadAudioLibraryCount();
        } else {
            showToast(result.message || 'Generation failed', 'error');
        }

    } catch (error) {
        showToast('Generation error', 'error');
        console.error(error);
    } finally {
        hideLoading();
    }
}

async function generateFromFile() {
    const fileInput = document.getElementById('text-file');

    if (!fileInput.files.length) {
        showToast('Please select a text file', 'warning');
        return;
    }

    const cfgScale = parseFloat(document.getElementById('cfg-scale').value);

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('voice_id', selectedVoiceId);
    formData.append('cfg_scale', cfgScale);

    showLoading('Generating speech from file...');

    try {
        const response = await fetch(`${API_BASE}/generate/file`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            showToast('Speech generated successfully', 'success');
            displayAudio(result);
            await loadAudioLibraryCount();
        } else {
            showToast(result.message || 'Generation failed', 'error');
        }

    } catch (error) {
        showToast('Generation error', 'error');
        console.error(error);
    } finally {
        hideLoading();
    }
}

// Audio Display
function displayAudio(result) {
    const outputSection = document.getElementById('output-section');
    outputSection.style.display = 'block';

    const audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = result.audio_url;
    audioPlayer.load();

    if (result.duration) {
        document.getElementById('audio-duration').textContent = `Duration: ${result.duration.toFixed(2)}s`;
    }
    document.getElementById('generation-time').textContent = `Generated at: ${new Date().toLocaleTimeString()}`;

    window.currentAudioUrl = result.audio_url;

    outputSection.scrollIntoView({behavior: 'smooth'});
}

function downloadAudio() {
    if (!window.currentAudioUrl) {
        showToast('No audio to download', 'warning');
        return;
    }

    const link = document.createElement('a');
    link.href = window.currentAudioUrl;
    link.download = `vibevoice_${Date.now()}.wav`;
    link.click();
}

function saveToLibrary() {
    showToast('Audio already saved to library', 'success');
}

// UI Helpers
function showLoading(text = 'Loading...') {
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('loading-text').textContent = text;
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = {
        success: 'fa-check-circle',
        error: 'fa-times-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    }[type];

    toast.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;

    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}