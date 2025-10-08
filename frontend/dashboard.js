// Face Recognition Dashboard JavaScript

class FaceRecognitionDashboard {
    constructor() {
        this.socket = null;
        this.charts = {};
        this.isConnected = false;
        this.currentTab = 'live-feed';
        this.faceDatabase = [];
        this.recognitionResults = [];
        this.systemStats = {
            totalFaces: 0,
            recognizedFaces: 0,
            currentFPS: 0,
            accuracy: 0
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.connectWebSocket();
        this.loadSettings();
        this.startPerformanceMonitoring();
    }
    
    // WebSocket Connection
    connectWebSocket() {
        try {
            this.socket = new WebSocket('ws://localhost:8765');
            
            this.socket.onopen = () => {
                this.isConnected = true;
                this.updateConnectionStatus(true);
                this.showNotification('Connected to face recognition system', 'success');
            };
            
            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.socket.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.showNotification('Disconnected from face recognition system', 'error');
                
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showNotification('Connection error occurred', 'error');
            };
            
        } catch (error) {
            console.error('Failed to connect to WebSocket:', error);
            this.showNotification('Failed to connect to server', 'error');
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'recognition_result':
                this.handleRecognitionResult(data.data);
                break;
            case 'stats':
                this.updateSystemStats(data.data);
                break;
            case 'face_added':
                this.showNotification(`Face added for ${data.identity}`, 'success');
                this.loadFaceDatabase();
                break;
            case 'error':
                this.showNotification(data.message, 'error');
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    handleRecognitionResult(result) {
        this.recognitionResults.unshift(result);
        if (this.recognitionResults.length > 100) {
            this.recognitionResults = this.recognitionResults.slice(0, 100);
        }
        
        this.updateRecognitionDisplay(result);
        this.updateLiveFeed(result);
        this.updateAnalytics(result);
    }
    
    // Event Listeners
    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const tabId = link.getAttribute('data-tab');
                this.switchTab(tabId);
            });
        });
        
        // Feed controls
        document.getElementById('startFeed').addEventListener('click', () => this.startFeed());
        document.getElementById('stopFeed').addEventListener('click', () => this.stopFeed());
        document.getElementById('captureFrame').addEventListener('click', () => this.captureFrame());
        
        // Database controls
        document.getElementById('addPerson').addEventListener('click', () => this.showAddPersonModal());
        document.getElementById('importDatabase').addEventListener('click', () => this.importDatabase());
        document.getElementById('exportDatabase').addEventListener('click', () => this.exportDatabase());
        
        // Settings controls
        document.getElementById('saveSettings').addEventListener('click', () => this.saveSettings());
        document.getElementById('resetSettings').addEventListener('click', () => this.resetSettings());
        
        // Modal controls
        document.querySelector('.close').addEventListener('click', () => this.hideModal());
        document.getElementById('cancelAdd').addEventListener('click', () => this.hideModal());
        document.getElementById('confirmAdd').addEventListener('click', () => this.addPerson());
        
        // File input for person photo
        document.getElementById('personPhoto').addEventListener('change', (e) => this.previewPhoto(e));
        
        // Search functionality
        document.getElementById('searchDatabase').addEventListener('input', (e) => this.searchDatabase(e.target.value));
        
        // Range input updates
        document.getElementById('fps').addEventListener('input', (e) => {
            document.getElementById('fpsValue').textContent = e.target.value;
        });
        
        document.getElementById('confidenceThreshold').addEventListener('input', (e) => {
            document.getElementById('confidenceValue').textContent = e.target.value;
        });
        
        // Log controls
        document.getElementById('clearLogs').addEventListener('click', () => this.clearLogs());
        document.getElementById('exportLogs').addEventListener('click', () => this.exportLogs());
        
        // Analytics controls
        document.getElementById('timeRange').addEventListener('change', (e) => this.updateAnalyticsTimeRange(e.target.value));
        document.getElementById('exportData').addEventListener('click', () => this.exportAnalyticsData());
    }
    
    // Tab Management
    switchTab(tabId) {
        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
        document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        
        this.currentTab = tabId;
        
        // Load tab-specific data
        this.loadTabData(tabId);
    }
    
    loadTabData(tabId) {
        switch (tabId) {
            case 'analytics':
                this.updateCharts();
                break;
            case 'database':
                this.loadFaceDatabase();
                break;
            case 'logs':
                this.loadSystemLogs();
                break;
        }
    }
    
    // Live Feed Management
    startFeed() {
        if (this.socket && this.isConnected) {
            this.socket.send(JSON.stringify({
                type: 'start_feed'
            }));
            this.showNotification('Live feed started', 'success');
        } else {
            this.showNotification('Not connected to server', 'error');
        }
    }
    
    stopFeed() {
        if (this.socket && this.isConnected) {
            this.socket.send(JSON.stringify({
                type: 'stop_feed'
            }));
            this.showNotification('Live feed stopped', 'warning');
        }
    }
    
    captureFrame() {
        if (this.socket && this.isConnected) {
            this.socket.send(JSON.stringify({
                type: 'capture_frame'
            }));
            this.showNotification('Frame captured', 'info');
        }
    }
    
    updateLiveFeed(result) {
        const canvas = document.getElementById('videoCanvas');
        const ctx = canvas.getContext('2d');
        
        // Update overlay information
        document.getElementById('faceCount').textContent = `Faces: ${result.faces.length}`;
        document.getElementById('processingTime').textContent = `Processing: ${Math.round(result.processing_time * 1000)}ms`;
        
        // Draw face detection results
        this.drawFaceResults(ctx, result.faces);
    }
    
    drawFaceResults(ctx, faces) {
        // Clear previous drawings
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        
        faces.forEach(face => {
            if (face.bbox) {
                const [x, y, w, h] = face.bbox;
                
                // Draw bounding box
                ctx.strokeStyle = face.identity ? '#00ff00' : '#ff0000';
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, w, h);
                
                // Draw label
                if (face.identity) {
                    const label = `${face.identity} (${Math.round(face.match_confidence * 100)}%)`;
                    ctx.fillStyle = '#00ff00';
                    ctx.font = '14px Arial';
                    ctx.fillText(label, x, y - 5);
                }
            }
        });
    }
    
    updateRecognitionDisplay(result) {
        const resultsList = document.getElementById('recognitionList');
        
        result.faces.forEach(face => {
            const resultItem = this.createResultItem(face, result.timestamp);
            resultsList.insertBefore(resultItem, resultsList.firstChild);
        });
        
        // Keep only last 20 results
        while (resultsList.children.length > 20) {
            resultsList.removeChild(resultsList.lastChild);
        }
    }
    
    createResultItem(face, timestamp) {
        const item = document.createElement('div');
        item.className = 'result-item';
        
        const confidence = Math.round((face.match_confidence || 0) * 100);
        const timeStr = new Date(timestamp * 1000).toLocaleTimeString();
        
        item.innerHTML = `
            <div class="result-header">
                <span class="result-name">${face.identity || 'Unknown'}</span>
                <span class="result-confidence">${confidence}%</span>
            </div>
            <div class="result-timestamp">${timeStr}</div>
        `;
        
        return item;
    }
    
    // Charts and Analytics
    initializeCharts() {
        this.initRecognitionChart();
        this.initPerformanceChart();
        this.initSystemChart();
        this.initDistributionChart();
    }
    
    initRecognitionChart() {
        const ctx = document.getElementById('recognitionChart').getContext('2d');
        this.charts.recognition = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Recognition Rate',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
    
    initPerformanceChart() {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        this.charts.performance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Processing Time', 'Queue Size', 'Memory Usage', 'CPU Usage'],
                datasets: [{
                    label: 'Performance Metrics',
                    data: [0, 0, 0, 0],
                    backgroundColor: ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
    
    initSystemChart() {
        const ctx = document.getElementById('systemChart').getContext('2d');
        this.charts.system = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Successful', 'Failed', 'Pending'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#2ecc71', '#e74c3c', '#f39c12']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
    
    initDistributionChart() {
        const data = [{
            x: [],
            y: [],
            type: 'histogram',
            marker: {
                color: '#3498db'
            }
        }];
        
        const layout = {
            title: 'Confidence Score Distribution',
            xaxis: { title: 'Confidence Score' },
            yaxis: { title: 'Frequency' },
            margin: { t: 40, b: 40, l: 40, r: 40 }
        };
        
        Plotly.newPlot('distributionChart', data, layout, {responsive: true});
    }
    
    updateCharts() {
        this.updateRecognitionChart();
        this.updatePerformanceChart();
        this.updateSystemChart();
        this.updateDistributionChart();
    }
    
    updateRecognitionChart() {
        // Generate sample data based on recent results
        const labels = [];
        const data = [];
        const now = Date.now();
        
        for (let i = 23; i >= 0; i--) {
            const time = new Date(now - i * 60 * 60 * 1000);
            labels.push(time.getHours() + ':00');
            
            // Calculate recognition rate for this hour
            const hourResults = this.recognitionResults.filter(result => {
                const resultTime = new Date(result.timestamp * 1000);
                return resultTime.getHours() === time.getHours();
            });
            
            const recognized = hourResults.filter(result => 
                result.faces.some(face => face.identity)
            ).length;
            
            const rate = hourResults.length > 0 ? (recognized / hourResults.length) * 100 : 0;
            data.push(rate);
        }
        
        this.charts.recognition.data.labels = labels;
        this.charts.recognition.data.datasets[0].data = data;
        this.charts.recognition.update();
    }
    
    updatePerformanceChart() {
        const stats = this.systemStats;
        this.charts.performance.data.datasets[0].data = [
            stats.avgProcessingTime || 0,
            stats.queueSize || 0,
            stats.memoryUsage || 0,
            stats.cpuUsage || 0
        ];
        this.charts.performance.update();
    }
    
    updateSystemChart() {
        const successful = this.recognitionResults.filter(r => r.faces.length > 0).length;
        const failed = this.recognitionResults.filter(r => r.faces.length === 0).length;
        const pending = 0; // Would come from actual system state
        
        this.charts.system.data.datasets[0].data = [successful, failed, pending];
        this.charts.system.update();
    }
    
    updateDistributionChart() {
        const confidenceScores = this.recognitionResults
            .flatMap(result => result.faces)
            .map(face => face.match_confidence || 0)
            .filter(score => score > 0);
        
        const data = [{
            x: confidenceScores,
            type: 'histogram',
            marker: { color: '#3498db' }
        }];
        
        Plotly.redraw('distributionChart');
    }
    
    // Database Management
    loadFaceDatabase() {
        // Simulate loading face database
        this.renderFaceDatabase();
    }
    
    renderFaceDatabase() {
        const personList = document.getElementById('personList');
        personList.innerHTML = '';
        
        this.faceDatabase.forEach(person => {
            const personCard = this.createPersonCard(person);
            personList.appendChild(personCard);
        });
    }
    
    createPersonCard(person) {
        const card = document.createElement('div');
        card.className = 'person-card';
        
        card.innerHTML = `
            <div class="person-photo">
                ${person.photo ? `<img src="${person.photo}" alt="${person.name}">` : '👤'}
            </div>
            <div class="person-name">${person.name}</div>
            <div class="person-details">
                ${person.email || 'No email'}
                <br>
                Added: ${new Date(person.dateAdded).toLocaleDateString()}
            </div>
        `;
        
        return card;
    }
    
    showAddPersonModal() {
        document.getElementById('addPersonModal').style.display = 'block';
    }
    
    hideModal() {
        document.getElementById('addPersonModal').style.display = 'none';
        document.getElementById('addPersonForm').reset();
        document.getElementById('photoPreview').innerHTML = '';
    }
    
    previewPhoto(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('photoPreview').innerHTML = 
                    `<img src="${e.target.result}" alt="Preview">`;
            };
            reader.readAsDataURL(file);
        }
    }
    
    addPerson() {
        const form = document.getElementById('addPersonForm');
        const formData = new FormData(form);
        
        const name = document.getElementById('personName').value;
        const email = document.getElementById('personEmail').value;
        const notes = document.getElementById('personNotes').value;
        const photoFile = document.getElementById('personPhoto').files[0];
        
        if (!name || !photoFile) {
            this.showNotification('Name and photo are required', 'error');
            return;
        }
        
        // Convert image to base64
        const reader = new FileReader();
        reader.onload = (e) => {
            const base64Data = e.target.result.split(',')[1];
            
            if (this.socket && this.isConnected) {
                this.socket.send(JSON.stringify({
                    type: 'add_face',
                    identity: name,
                    face_data: base64Data,
                    email: email,
                    notes: notes
                }));
            }
            
            // Add to local database
            this.faceDatabase.push({
                name: name,
                email: email,
                notes: notes,
                photo: e.target.result,
                dateAdded: Date.now()
            });
            
            this.hideModal();
            this.renderFaceDatabase();
        };
        
        reader.readAsDataURL(photoFile);
    }
    
    searchDatabase(query) {
        const cards = document.querySelectorAll('.person-card');
        cards.forEach(card => {
            const name = card.querySelector('.person-name').textContent.toLowerCase();
            const details = card.querySelector('.person-details').textContent.toLowerCase();
            
            if (name.includes(query.toLowerCase()) || details.includes(query.toLowerCase())) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
    }
    
    // Settings Management
    saveSettings() {
        const settings = {
            cameraSource: document.getElementById('cameraSource').value,
            resolution: document.getElementById('resolution').value,
            fps: document.getElementById('fps').value,
            confidenceThreshold: document.getElementById('confidenceThreshold').value,
            maxFaces: document.getElementById('maxFaces').value,
            enableGPU: document.getElementById('enableGPU').checked,
            enableAlerts: document.getElementById('enableAlerts').checked,
            emailNotifications: document.getElementById('emailNotifications').checked,
            alertSound: document.getElementById('alertSound').value,
            enableLogging: document.getElementById('enableLogging').checked,
            anonymizeData: document.getElementById('anonymizeData').checked,
            dataRetention: document.getElementById('dataRetention').value
        };
        
        localStorage.setItem('faceRecognitionSettings', JSON.stringify(settings));
        
        if (this.socket && this.isConnected) {
            this.socket.send(JSON.stringify({
                type: 'configure',
                config: settings
            }));
        }
        
        this.showNotification('Settings saved successfully', 'success');
    }
    
    loadSettings() {
        const savedSettings = localStorage.getItem('faceRecognitionSettings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            
            // Apply settings to form
            Object.keys(settings).forEach(key => {
                const element = document.getElementById(key);
                if (element) {
                    if (element.type === 'checkbox') {
                        element.checked = settings[key];
                    } else {
                        element.value = settings[key];
                    }
                }
            });
            
            // Update range display values
            document.getElementById('fpsValue').textContent = settings.fps || 30;
            document.getElementById('confidenceValue').textContent = settings.confidenceThreshold || 70;
        }
    }
    
    resetSettings() {
        localStorage.removeItem('faceRecognitionSettings');
        location.reload();
    }
    
    // System Monitoring
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (connected) {
            statusElement.textContent = 'Connected';
            statusElement.className = 'status-indicator online';
        } else {
            statusElement.textContent = 'Disconnected';
            statusElement.className = 'status-indicator offline';
        }
    }
    
    updateSystemStats(stats) {
        this.systemStats = { ...this.systemStats, ...stats };
        
        document.getElementById('currentFPS').textContent = Math.round(stats.fps || 0);
        document.getElementById('totalFaces').textContent = this.recognitionResults.length;
        
        const recognized = this.recognitionResults.filter(r => 
            r.faces.some(face => face.identity)
        ).length;
        document.getElementById('recognizedFaces').textContent = recognized;
        
        const accuracy = this.recognitionResults.length > 0 ? 
            Math.round((recognized / this.recognitionResults.length) * 100) : 0;
        document.getElementById('accuracy').textContent = `${accuracy}%`;
    }
    
    updateAnalytics(result) {
        // Update analytics data based on new result
        this.updateCharts();
    }
    
    startPerformanceMonitoring() {
        setInterval(() => {
            if (this.socket && this.isConnected) {
                this.socket.send(JSON.stringify({
                    type: 'get_stats'
                }));
            }
        }, 5000); // Update every 5 seconds
    }
    
    // Utility Functions
    showNotification(message, type = 'info') {
        const container = document.getElementById('notificationContainer');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            container.removeChild(notification);
        }, 5000);
    }
    
    exportAnalyticsData() {
        const data = {
            recognitionResults: this.recognitionResults,
            systemStats: this.systemStats,
            faceDatabase: this.faceDatabase.map(person => ({
                name: person.name,
                email: person.email,
                dateAdded: person.dateAdded
            }))
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `face_recognition_data_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    // Logs Management
    loadSystemLogs() {
        const logOutput = document.getElementById('logOutput');
        // Simulate loading logs
        const sampleLogs = [
            { level: 'info', message: 'Face recognition system started', timestamp: new Date() },
            { level: 'debug', message: 'Loading face detection model', timestamp: new Date() },
            { level: 'info', message: 'WebSocket server listening on port 8765', timestamp: new Date() },
            { level: 'warning', message: 'High CPU usage detected', timestamp: new Date() }
        ];
        
        logOutput.innerHTML = sampleLogs.map(log => 
            `<div class="log-entry ${log.level}">
                [${log.timestamp.toLocaleTimeString()}] ${log.level.toUpperCase()}: ${log.message}
            </div>`
        ).join('');
    }
    
    clearLogs() {
        document.getElementById('logOutput').innerHTML = '';
        this.showNotification('Logs cleared', 'info');
    }
    
    exportLogs() {
        const logContent = document.getElementById('logOutput').textContent;
        const blob = new Blob([logContent], { type: 'text/plain' });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `face_recognition_logs_${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new FaceRecognitionDashboard();
});

// Handle window resize for charts
window.addEventListener('resize', () => {
    if (window.dashboard && window.dashboard.charts) {
        Object.values(window.dashboard.charts).forEach(chart => {
            if (chart.resize) chart.resize();
        });
    }
});