// Global variables
let selectedVoiceId = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordedBlob = null;

// API Base URL
const API_BASE = '/api';

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    loadVoices();
    setupEventListeners();
    initializeTheme();
});

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

    // File upload drag & drop
    const uploadArea = document.getElementById('upload-drop');
    uploadArea.addEventListener('click', () => {
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
            document.getElementById('voice-upload').files = files;
        }
    });

    // Text file upload
    document.getElementById('text-file').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            document.getElementById('file-name').textContent = `Selected: ${file.name}`;
        }
    });
}

// Tab Switching
function switchTab(tab) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tab}-tab`).style.display = 'block';

    // Add active class to clicked button
    event.target.classList.add('active');
}

function switchTextTab(tab) {
    document.querySelectorAll('.text-tab-content').forEach(content => {
        content.style.display = 'none';
    });

    document.querySelectorAll('.text-tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    document.getElementById(`${tab}-tab`).style.display = 'block';
    event.target.classList.add('active');
}

// Voice Management
async function loadVoices() {
    try {
        const response = await fetch(`${API_BASE}/voices`);
        const voices = await response.json();

        const voiceList = document.getElementById('voice-list');
        voiceList.innerHTML = '';

        voices.forEach(voice => {
            const voiceCard = document.createElement('div');
            voiceCard.className = 'voice-card';
            voiceCard.dataset.voiceId = voice.id;
            voiceCard.onclick = () => selectVoice(voice.id);

            voiceCard.innerHTML = `
                <i class="fas fa-user-circle"></i>
                <div class="voice-name">${voice.name}</div>
            `;

            voiceList.appendChild(voiceCard);
        });

        // Select first voice by default
        if (voices.length > 0) {
            selectVoice(voices[0].id);
        }

    } catch (error) {
        showToast('Failed to load voices', 'error');
        console.error(error);
    }
}

function selectVoice(voiceId) {
    selectedVoiceId = voiceId;

    // Update UI
    document.querySelectorAll('.voice-card').forEach(card => {
        card.classList.remove('selected');
    });

    const selectedCard = document.querySelector(`[data-voice-id="${voiceId}"]`);
    if (selectedCard) {
        selectedCard.classList.add('selected');
    }
}

async function uploadVoice() {
    const fileInput = document.getElementById('voice-upload');
    const nameInput = document.getElementById('upload-name');

    if (!fileInput.files.length) {
        showToast('Please select an audio file', 'warning');
        return;
    }

    if (!nameInput.value.trim()) {
        showToast('Please enter a voice name', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('name', nameInput.value.trim());

    showLoading('Uploading voice...');

    try {
        const response = await fetch(`${API_BASE}/voices/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            showToast('Voice uploaded successfully', 'success');
            loadVoices();
            fileInput.value = '';
            nameInput.value = '';
        } else {
            showToast('Upload failed', 'error');
        }

    } catch (error) {
        showToast('Upload error', 'error');
        console.error(error);
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
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(audioChunks, {type: 'audio/wav'});
            document.getElementById('save-recording').style.display = 'inline-block';
        };

        mediaRecorder.start();
        isRecording = true;

        // Update UI
        const recordBtn = document.getElementById('record-btn');
        recordBtn.classList.add('recording');
        document.getElementById('record-status').textContent = 'Recording...';

        // Start visualizer
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

        // Update UI
        const recordBtn = document.getElementById('record-btn');
        recordBtn.classList.remove('recording');
        document.getElementById('record-status').textContent = 'Recording complete';
    }
}

async function saveRecording() {
    const nameInput = document.getElementById('record-name');

    if (!nameInput.value.trim()) {
        showToast('Please enter a voice name', 'warning');
        return;
    }

    if (!recordedBlob) {
        showToast('No recording available', 'warning');
        return;
    }

    // Convert blob to base64
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
                    name: nameInput.value.trim(),
                    audio_data: base64Audio,
                    format: 'wav'
                })
            });

            const result = await response.json();

            if (result.success) {
                showToast('Recording saved successfully', 'success');
                loadVoices();
                nameInput.value = '';
                document.getElementById('save-recording').style.display = 'none';
                document.getElementById('record-status').textContent = 'Click to start recording';
                recordedBlob = null;
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
    const audioContext = new AudioContext();
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

    // Check if file tab is active and has file
    const fileTab = document.getElementById('file-tab');
    if (fileTab.style.display !== 'none' && textFileInput.files.length > 0) {
        // Generate from file
        await generateFromFile();
        return;
    } else {
        // Generate from text input
        text = document.getElementById('text-input').value.trim();
    }

    if (!text) {
        showToast('Please enter some text', 'warning');
        return;
    }

    const cfgScale = parseFloat(document.getElementById('cfg-scale').value);
    const numSpeakers = parseInt(document.getElementById('num-speakers').value);

    showLoading('Generating speech...');

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

    document.getElementById('audio-duration').textContent = `Duration: ${result.duration.toFixed(2)}s`;
    document.getElementById('generation-time').textContent = `Generated at: ${new Date().toLocaleTimeString()}`;

    // Store current audio URL for download
    window.currentAudioUrl = result.audio_url;
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
    showToast('Audio saved to library', 'success');
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