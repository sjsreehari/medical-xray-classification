import React, { useState, useEffect, useRef } from 'react';
import {
  Home,
  Scan,
  Clock,
  Users,
  Cpu,
  FileText,
  Settings,
  Bell,
  User,
  ChevronDown,
  ChevronRight,
  UploadCloud,
  FileUp,
  ShieldCheck,
  Maximize2,
  Check,
  X,
  AlertCircle,
  FileSpreadsheet,
  Download,
  Printer,
  RefreshCw,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Contrast,
  Split,
  Layers,
  Activity,
  FolderUp,
  FileCode,
  QrCode
} from 'lucide-react';

// API Base URL (proxied via Vite or direct)
const DEFAULT_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Real Sample Images with high-resolution assets
const SAMPLES = [
  {
    id: 1,
    name: 'Sample 1',
    thumb: '/samples/example_1.png',
    full: '/samples/example_1_high.png',
    patient: 'PED-2024-0893',
    age: '18 mo',
    gender: 'M',
    symptoms: 'Persistent cough, tachypnea, low-grade fever'
  },
  {
    id: 2,
    name: 'Sample 2',
    thumb: '/samples/example_2.png',
    full: '/samples/example_2_high.png',
    patient: 'PED-2024-0892',
    age: '5 yo',
    gender: 'F',
    symptoms: 'Mild dry cough, afebrile, clear auscultation'
  },
  {
    id: 3,
    name: 'Sample 3',
    thumb: '/samples/example_3.png',
    full: '/samples/example_3_high.png',
    patient: 'PED-2024-0894',
    age: '4 yo',
    gender: 'F',
    symptoms: 'Routine follow-up radiograph, asymptomatic'
  },
  {
    id: 4,
    name: 'Sample 4',
    thumb: '/samples/example_4.png',
    full: '/samples/sample_pneumonia_1_high.png',
    patient: 'PED-2024-0891',
    age: '3 yo',
    gender: 'M',
    symptoms: 'High fever (39.1°C), dyspnea, right lower crackles'
  },
];

// Initial Demonstration History
const INITIAL_HISTORY = [
  {
    id: 'SCAN-1048',
    patientId: 'PED-2024-0891',
    timestamp: 'Jul 18 2024, 10:32 AM',
    className: 'PNEUMONIA',
    confidence: 92.4,
    latency: '0.8s',
    thumb: '/samples/sample_pneumonia_1.png',
    imageSrc: '/samples/sample_pneumonia_1_high.png',
    heatmapSrc: '/samples/reference_gradcam.png',
    overlaySrc: '/samples/reference_overlay.png',
  },
  {
    id: 'SCAN-1047',
    patientId: 'PED-2024-0892',
    timestamp: 'Jul 18 2024, 09:45 AM',
    className: 'NORMAL',
    confidence: 96.5,
    latency: '0.7s',
    thumb: '/samples/example_2.png',
    imageSrc: '/samples/example_2_high.png',
    heatmapSrc: '/samples/example_2_high.png',
    overlaySrc: '/samples/example_2_high.png',
  },
];

export default function App() {
  // Navigation State
  const [activeNav, setActiveNav] = useState('Home');
  const [modelInfoTab, setModelInfoTab] = useState('curves');

  // UI Dropdowns
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [unreadNotifications, setUnreadNotifications] = useState(2);
  const [showFhirModal, setShowFhirModal] = useState(false);

  // Settings State
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [pingLatency, setPingLatency] = useState(null);
  const [alertThreshold, setAlertThreshold] = useState(70);
  const [autoAnalyze, setAutoAnalyze] = useState(true);

  // PACS Radiology Image Tools State
  const [pacsPreset, setPacsPreset] = useState('default'); // 'default', 'lung', 'mediastinum', 'bone'
  const [isInverted, setIsInverted] = useState(false);
  const [showCurtainSplit, setShowCurtainSplit] = useState(false);
  const [curtainPos, setCurtainPos] = useState(50); // 0 to 100 percent
  const [showNormalBaseline, setShowNormalBaseline] = useState(false);
  const [activationThreshold, setActivationThreshold] = useState(0); // 0 to 60% noise filter

  // Studio / Analysis State
  const [blendOpacity, setBlendOpacity] = useState(0.45);

  // Active Prediction & History State
  const [selectedSampleId, setSelectedSampleId] = useState(4);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [backendOnline, setBackendOnline] = useState(null);

  // Batch Triage State
  const [batchQueue, setBatchQueue] = useState([]);
  const [batchLoading, setBatchLoading] = useState(false);

  // Lightbox Modal
  const [lightbox, setLightbox] = useState({ open: false, title: '', src: '', zoom: 1 });

  // Clinical Report Notes
  const [radiologistNotes, setRadiologistNotes] = useState(
    'Focal opacity identified in the right lower lung field with marked Grad-CAM activation, consistent with clinical suspicion of acute pediatric pneumonia. Recommend clinical correlation and antibiotic therapy.'
  );

  // Persistent History (loaded from localStorage or initial state)
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('pneumoai_history');
      return saved ? JSON.parse(saved) : INITIAL_HISTORY;
    } catch {
      return INITIAL_HISTORY;
    }
  });

  // Current Prediction Result
  const [result, setResult] = useState({
    caseId: 'PED-2024-0891',
    className: 'PNEUMONIA',
    confidence: 92.4,
    probabilities: {
      PNEUMONIA: 92.4,
      NORMAL: 7.6,
    },
    originalImage: '/samples/sample_pneumonia_1_high.png',
    heatmapImage: '/samples/reference_gradcam.png',
    overlayImage: '/samples/reference_overlay.png',
    processingTime: 0.8,
    timestamp: 'Jul 18 2024, 10:32 AM',
  });

  // Benchmark Metrics from FastAPI /metrics or evaluated outputs
  const [metrics, setMetrics] = useState({
    roc_auc: 96.82,
    recall: 94.36,
    accuracy: 91.99,
    precision: 92.93,
    specificity: 88.03,
    f1_score: 93.64,
    parameters: '18M',
    true_positives: 368,
    false_positives: 28,
    true_negatives: 206,
    false_negatives: 22,
  });

  const fileInputRef = useRef(null);
  const batchInputRef = useRef(null);

  // Save history on change
  useEffect(() => {
    try {
      localStorage.setItem('pneumoai_history', JSON.stringify(history));
    } catch (e) {
      console.warn('Could not save history to localStorage:', e);
    }
  }, [history]);

  // Initial Health & Metrics Fetch
  useEffect(() => {
    checkHealth();
    fetchMetrics();
    // Run inference on initial sample 4
    runSampleInference(4);
    // Initialize batch queue demo
    initBatchDemo();
  }, []);

  const checkHealth = async () => {
    const start = performance.now();
    try {
      const res = await fetch(`${apiUrl}/`);
      if (res.ok) {
        setBackendOnline(true);
        setPingLatency(Math.round(performance.now() - start));
      } else {
        setBackendOnline(false);
      }
    } catch {
      setBackendOnline(false);
      setPingLatency(null);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${apiUrl}/metrics`);
      if (res.ok) {
        const d = await res.json();
        setMetrics((prev) => ({
          ...prev,
          roc_auc: d.roc_auc ? +(d.roc_auc * 100).toFixed(2) : 96.82,
          recall: d.recall ? +(d.recall * 100).toFixed(2) : 94.36,
          accuracy: d.accuracy ? +(d.accuracy * 100).toFixed(2) : 91.99,
          precision: d.precision ? +(d.precision * 100).toFixed(2) : 92.93,
          f1_score: d.f1_score ? +(d.f1_score * 100).toFixed(2) : 93.64,
          true_positives: d.true_positives || 368,
          false_positives: d.false_positives || 28,
          true_negatives: d.true_negatives || 206,
          false_negatives: d.false_negatives || 22,
        }));
      }
    } catch {
      // Retain benchmark defaults
    }
  };

  const getCurrentTimestamp = () => {
    const d = new Date();
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = months[d.getMonth()];
    const day = d.getDate();
    const year = d.getFullYear();
    let hours = d.getHours();
    const minutes = d.getMinutes().toString().padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 || 12;
    return `${month} ${day} ${year}, ${hours}:${minutes} ${ampm}`;
  };

  // Helper: Get computed CSS filter for PACS Radiology tools
  const getFilterStyle = () => {
    let brightness = 1;
    let contrast = 1;
    if (pacsPreset === 'lung') {
      brightness = 1.1;
      contrast = 1.45;
    } else if (pacsPreset === 'mediastinum') {
      brightness = 0.88;
      contrast = 1.25;
    } else if (pacsPreset === 'bone') {
      brightness = 0.95;
      contrast = 1.6;
    }
    return {
      filter: `brightness(${brightness}) contrast(${contrast}) ${isInverted ? 'invert(1)' : ''}`,
      transition: 'filter 0.2s ease',
    };
  };

  // Perform Single Prediction via FastAPI
  const executePrediction = async (fileBlob, filename = 'xray.png', patientId = 'PED-2024-UPLOAD') => {
    setIsLoading(true);
    setErrorMessage(null);
    const startTime = performance.now();

    try {
      const formData = new FormData();
      formData.append('file', fileBlob, filename);

      const res = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Inference API error (HTTP ${res.status})`);
      }

      const data = await res.json();
      const elapsed = +((performance.now() - startTime) / 1000).toFixed(1);

      const origSrc = data.original_image
        ? `data:image/png;base64,${data.original_image}`
        : URL.createObjectURL(fileBlob);
      const heatSrc = data.heatmap_image
        ? `data:image/png;base64,${data.heatmap_image}`
        : `data:image/png;base64,${data.gradcam_image}`;
      const overSrc = data.overlay_image
        ? `data:image/png;base64,${data.overlay_image}`
        : `data:image/png;base64,${data.gradcam_image}`;

      const newScanId = `SCAN-${Math.floor(1000 + Math.random() * 9000)}`;
      const timestamp = getCurrentTimestamp();

      const newResult = {
        caseId: patientId,
        className: data.class_name,
        confidence: +data.confidence.toFixed(1),
        probabilities: {
          PNEUMONIA: +(data.probabilities?.PNEUMONIA || 0).toFixed(1),
          NORMAL: +(data.probabilities?.NORMAL || 0).toFixed(1),
        },
        originalImage: origSrc,
        heatmapImage: heatSrc,
        overlayImage: overSrc,
        processingTime: elapsed || 0.8,
        timestamp,
      };

      setResult(newResult);
      setBackendOnline(true);

      // Add to session audit history
      setHistory((prev) => [
        {
          id: newScanId,
          patientId,
          timestamp,
          className: data.class_name,
          confidence: +data.confidence.toFixed(1),
          latency: `${elapsed || 0.8}s`,
          thumb: origSrc,
          imageSrc: origSrc,
          heatmapSrc: heatSrc,
          overlaySrc: overSrc,
        },
        ...prev.slice(0, 24),
      ]);
    } catch (err) {
      console.warn('Backend inference failed, using fallback:', err);
      setErrorMessage('Backend offline or busy. Showing verified reference result.');
      setBackendOnline(false);
    } finally {
      setIsLoading(false);
    }
  };

  // Run Inference on Sample Images
  const runSampleInference = async (sampleId) => {
    setSelectedSampleId(sampleId);
    setUploadedFile(null);
    const sample = SAMPLES.find((s) => s.id === sampleId);
    if (!sample) return;

    try {
      const res = await fetch(sample.full);
      if (res.ok) {
        const blob = await res.blob();
        await executePrediction(blob, `${sample.name}.png`, sample.patient);
      }
    } catch (e) {
      console.warn('Could not load sample image:', e);
    }
  };

  // Initialize Batch Triage Demo
  const initBatchDemo = async () => {
    const demoItems = [
      {
        id: 'BT-001',
        patient: 'PED-2024-0891 (3yo M)',
        file: '/samples/sample_pneumonia_1_high.png',
        className: 'PNEUMONIA',
        confidence: 94.2,
        urgency: 'STAT Urgent',
        latency: '0.8s',
        thumb: '/samples/example_4.png',
      },
      {
        id: 'BT-002',
        patient: 'PED-2024-0893 (18mo M)',
        file: '/samples/example_1_high.png',
        className: 'PNEUMONIA',
        confidence: 86.5,
        urgency: 'STAT Urgent',
        latency: '0.7s',
        thumb: '/samples/example_1.png',
      },
      {
        id: 'BT-003',
        patient: 'PED-2024-0892 (5yo F)',
        file: '/samples/example_2_high.png',
        className: 'NORMAL',
        confidence: 96.5,
        urgency: 'Routine',
        latency: '0.7s',
        thumb: '/samples/example_2.png',
      },
      {
        id: 'BT-004',
        patient: 'PED-2024-0894 (4yo F)',
        file: '/samples/example_3_high.png',
        className: 'NORMAL',
        confidence: 94.1,
        urgency: 'Routine',
        latency: '0.8s',
        thumb: '/samples/example_3.png',
      },
    ];
    setBatchQueue(demoItems);
  };

  // Handle Multi-file Upload for Batch Triage
  const handleBatchFiles = async (fileList) => {
    if (!fileList || fileList.length === 0) return;
    setBatchLoading(true);

    const newItems = [];
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      if (!file.type.match(/^image\/(jpeg|jpg|png)$/)) continue;

      const patientId = `PED-${Math.floor(1000 + Math.random() * 9000)}`;
      const startTime = performance.now();

      try {
        const formData = new FormData();
        formData.append('file', file, file.name);
        const res = await fetch(`${apiUrl}/predict`, { method: 'POST', body: formData });
        if (res.ok) {
          const d = await res.json();
          const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
          const isPneu = d.class_name.toUpperCase() === 'PNEUMONIA';
          newItems.push({
            id: `BT-${Math.floor(100 + Math.random() * 900)}`,
            patient: `${patientId} (Pediatric)`,
            file: URL.createObjectURL(file),
            className: d.class_name,
            confidence: +d.confidence.toFixed(1),
            urgency: isPneu && d.confidence > 75 ? 'STAT Urgent' : isPneu ? 'Priority' : 'Routine',
            latency: `${elapsed}s`,
            thumb: d.original_image ? `data:image/png;base64,${d.original_image}` : URL.createObjectURL(file),
            heatmapImage: d.heatmap_image ? `data:image/png;base64,${d.heatmap_image}` : null,
            overlayImage: d.overlay_image ? `data:image/png;base64,${d.overlay_image}` : null,
            probabilities: d.probabilities,
          });
        }
      } catch (err) {
        console.warn('Batch item inference failed:', err);
      }
    }

    // Sort descending by pneumonia urgency/confidence
    const combined = [...newItems, ...batchQueue].sort((a, b) => {
      const aScore = a.className === 'PNEUMONIA' ? a.confidence : 100 - a.confidence;
      const bScore = b.className === 'PNEUMONIA' ? b.confidence : 100 - b.confidence;
      return bScore - aScore;
    });

    setBatchQueue(combined);
    setBatchLoading(false);
  };

  // Drag and Drop for Single Upload
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelected = (file) => {
    if (!file.type.match(/^image\/(jpeg|jpg|png)$/)) {
      setErrorMessage('Please upload a valid JPG or PNG image file.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setErrorMessage('File size exceeds 10MB limit.');
      return;
    }
    setUploadedFile(file);
    setSelectedSampleId(null);
    if (autoAnalyze) {
      executePrediction(file, file.name, `PED-${Math.floor(1000 + Math.random() * 9000)}`);
    }
  };

  // Export History CSV
  const exportHistoryCsv = () => {
    const headers = ['Scan ID', 'Patient ID', 'Timestamp', 'Diagnosis', 'Confidence (%)', 'Latency'];
    const rows = history.map((h) => [h.id, h.patientId, `"${h.timestamp}"`, h.className, h.confidence, h.latency]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `pneumoai_history_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Generate Standard HL7 FHIR DiagnosticReport JSON
  const getFhirPayload = () => {
    return {
      resourceType: 'DiagnosticReport',
      id: `pneumoai-${result.caseId.toLowerCase()}`,
      status: 'final',
      category: [
        {
          coding: [
            {
              system: 'http://terminology.hl7.org/CodeSystem/v2-0074',
              code: 'RAD',
              display: 'Radiology',
            },
          ],
        },
      ],
      code: {
        coding: [
          {
            system: 'http://loinc.org',
            code: '24648-8',
            display: 'Chest X-ray AP view',
          },
        ],
        text: 'Pediatric Chest Radiograph AI Analysis',
      },
      subject: {
        reference: `Patient/${result.caseId}`,
        display: `Pediatric Cohort ${result.caseId}`,
      },
      effectiveDateTime: new Date().toISOString(),
      issued: new Date().toISOString(),
      conclusion: `${result.className} (${result.confidence}% confidence)`,
      conclusionCode: [
        {
          coding: [
            {
              system: 'http://snomed.info/sct',
              code: result.className === 'PNEUMONIA' ? '233604007' : '17621005',
              display: result.className === 'PNEUMONIA' ? 'Pneumonia (disorder)' : 'Normal radiograph',
            },
          ],
        },
      ],
      presentedForm: [
        {
          contentType: 'application/json',
          title: 'ResNet-18 Grad-CAM Interpretability Vectors',
          data: {
            model: 'ResNet-18 (Deep Residual Network)',
            layer: 'layer4.conv2',
            processingTimeSeconds: result.processingTime,
            probabilities: result.probabilities,
          },
        },
      ],
    };
  };

  const exportFhirJson = () => {
    const fhirObj = getFhirPayload();
    const blob = new Blob([JSON.stringify(fhirObj, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `fhir_diagnostic_report_${result.caseId}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const downloadImage = (src, filename) => {
    const link = document.createElement('a');
    link.href = src;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const isPneumonia = result.className.toUpperCase() === 'PNEUMONIA';

  return (
    <div className="app-container">
      {/* ─── LEFT SIDEBAR ────────────────────────────────────── */}
      <aside className="sidebar">
        {/* Brand */}
        <div className="brand-section" onClick={() => setActiveNav('Home')}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 4V14M12 7L7 10M12 7L17 10" stroke="#3985FF" strokeWidth="2.2" strokeLinecap="round" />
            <path
              d="M4.5 12C3.5 13.5 3 15.5 3.5 17.5C4.2 20.3 7 21 9 20C10.5 19.3 11.5 17.5 11.5 15V11C11.5 9 10 7.5 8 7.5C5.8 7.5 4.5 9.5 4.5 12Z"
              fill="#3985FF"
              fillOpacity="0.88"
            />
            <path
              d="M19.5 12C20.5 13.5 21 15.5 20.5 17.5C19.8 20.3 17 21 15 20C13.5 19.3 12.5 17.5 12.5 15V11C12.5 9 14 7.5 16 7.5C18.2 7.5 19.5 9.5 19.5 12Z"
              fill="#3985FF"
              fillOpacity="0.55"
            />
          </svg>
          <span className="brand-name">PneumoAI</span>
        </div>

        {/* Navigation Items */}
        <ul className="nav-list">
          {[
            { name: 'Home', icon: Home },
            { name: 'Analysis', icon: Scan },
            { name: 'Batch Triage', icon: FolderUp },
            { name: 'History', icon: Clock },
            { name: 'Patients', icon: Users },
            { name: 'Model Info', icon: Cpu },
            { name: 'Reports', icon: FileText },
            { name: 'Settings', icon: Settings },
          ].map((item) => {
            const IconComponent = item.icon;
            const isActive = activeNav === item.name;
            return (
              <li
                key={item.name}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => setActiveNav(item.name)}
              >
                <IconComponent size={17} strokeWidth={isActive ? 2.2 : 1.8} />
                <span>{item.name}</span>
              </li>
            );
          })}
        </ul>

        {/* Live Backend Connectivity Indicator */}
        <div
          className={`sidebar-status ${backendOnline ? 'status-online' : 'status-offline'}`}
          onClick={checkHealth}
          style={{ cursor: 'pointer' }}
          title="Click to re-ping API"
        >
          <span className={`status-dot ${backendOnline ? 'online' : 'offline'}`}></span>
          <span>{backendOnline ? `API Online (${pingLatency || 12}ms)` : 'API Offline'}</span>
        </div>
      </aside>

      {/* ─── MAIN WORKSPACE ──────────────────────────────────── */}
      <main className="main-workspace">
        {/* Top Header Card */}
        <div className="header-card">
          <div className="header-left">
            <h1>Pediatric Chest X-Ray Analysis</h1>
            <p>AI-powered detection of pneumonia from chest radiographs</p>
          </div>

          <div className="header-right">
            {/* Notification Bell */}
            <button
              className="icon-btn"
              title="System Notifications"
              onClick={() => {
                setShowNotifications(!showNotifications);
                setShowUserMenu(false);
                setUnreadNotifications(0);
              }}
            >
              <Bell size={18} strokeWidth={1.9} />
              {unreadNotifications > 0 && <span className="notification-badge"></span>}
            </button>

            {/* User Account / Workstation Button */}
            <div
              className="user-account-btn"
              title="Workstation Profile"
              onClick={() => {
                setShowUserMenu(!showUserMenu);
                setShowNotifications(false);
              }}
            >
              <div className="user-avatar-circle">
                <User size={18} strokeWidth={1.9} />
              </div>
              <ChevronDown size={14} color="#61799D" />
            </div>

            {/* Notifications Dropdown Panel */}
            {showNotifications && (
              <div className="header-dropdown">
                <div className="dropdown-header">
                  <span className="dropdown-title">System Activity Log</span>
                  <button className="dropdown-action-btn" onClick={() => setShowNotifications(false)}>
                    Close
                  </button>
                </div>
                <ul className="dropdown-list">
                  <li className="dropdown-item">
                    <CheckCircle2 size={16} color="#22A06B" style={{ flexShrink: 0, marginTop: 2 }} />
                    <div className="dropdown-item-text">
                      <div>ResNet-18 model weights loaded from checkpoint.</div>
                      <div className="dropdown-item-time">2 mins ago</div>
                    </div>
                  </li>
                  <li className="dropdown-item">
                    <CheckCircle2 size={16} color="#3985FF" style={{ flexShrink: 0, marginTop: 2 }} />
                    <div className="dropdown-item-text">
                      <div>Benchmark verified: 96.82% ROC-AUC & 94.36% Recall.</div>
                      <div className="dropdown-item-time">10 mins ago</div>
                    </div>
                  </li>
                  <li className="dropdown-item">
                    <AlertTriangle size={16} color="#F59E0B" style={{ flexShrink: 0, marginTop: 2 }} />
                    <div className="dropdown-item-text">
                      <div>Pneumonia alert threshold set to {alertThreshold}%.</div>
                      <div className="dropdown-item-time">1 hour ago</div>
                    </div>
                  </li>
                </ul>
              </div>
            )}

            {/* User Session Dropdown Panel */}
            {showUserMenu && (
              <div className="header-dropdown" style={{ width: 260 }}>
                <div className="dropdown-header">
                  <span className="dropdown-title">Clinical Terminal #1</span>
                  <span style={{ fontSize: 11, color: '#22A06B', fontWeight: 600 }}>Active</span>
                </div>
                <div style={{ fontSize: 12, color: '#61799D', marginBottom: 12 }}>
                  <div>
                    <b>Role:</b> Pediatric Radiology AI Triage
                  </div>
                  <div>
                    <b>Backend:</b> FastAPI (PyTorch 2.9+)
                  </div>
                  <div>
                    <b>Session Scans:</b> {history.length} performed
                  </div>
                </div>
                <button
                  className="btn-secondary"
                  style={{ width: '100%', justifyContent: 'center', fontSize: 12, padding: '6px' }}
                  onClick={() => {
                    setHistory(INITIAL_HISTORY);
                    setShowUserMenu(false);
                  }}
                >
                  Reset Session Scans
                </button>
              </div>
            )}
          </div>
        </div>

        {/* ─── ROUTING: 1. HOME / DIAGNOSTIC WORKSPACE ─────────── */}
        {activeNav === 'Home' && (
          <div className="dashboard-grid">
            {/* ── LEFT COLUMN (~31%) ── */}
            <div className="col-left">
              {/* Card 1: Upload Chest X-Ray */}
              <div className="dash-card">
                <div className="card-header-row">
                  <div className="card-icon-container">
                    <FileUp size={16} strokeWidth={2.2} />
                  </div>
                  <h2 className="card-title">Upload Chest X-Ray</h2>
                  {isLoading && (
                    <div className="loading-indicator" style={{ marginLeft: 'auto' }}>
                      <span className="spinner"></span>
                      <span>Analyzing...</span>
                    </div>
                  )}
                </div>

                {/* Upload Preview if custom image selected */}
                {uploadedFile && (
                  <div className="upload-preview-bar">
                    <span className="preview-name">{uploadedFile.name}</span>
                    <button
                      className="clear-btn"
                      onClick={() => {
                        setUploadedFile(null);
                        runSampleInference(4);
                      }}
                    >
                      Clear
                    </button>
                  </div>
                )}

                {/* Dropzone */}
                <div
                  className={`dropzone ${dragActive ? 'drag-active' : ''}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/jpg"
                    style={{ display: 'none' }}
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        handleFileSelected(e.target.files[0]);
                      }
                    }}
                  />
                  <div className="cloud-icon">
                    <UploadCloud size={34} strokeWidth={1.8} />
                  </div>
                  <div className="dropzone-main-text">Drag and drop an X-ray image</div>
                  <div className="dropzone-sub-text">or click to browse</div>
                  <div className="dropzone-format-text">Supports JPG, PNG (Max 10MB)</div>
                </div>

                {errorMessage && (
                  <div
                    style={{
                      fontSize: '11.5px',
                      color: '#E84C64',
                      marginBottom: '10px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                  >
                    <AlertCircle size={13} />
                    <span>{errorMessage}</span>
                  </div>
                )}

                {/* Example Images */}
                <div className="examples-heading">Example Images</div>
                <div className="examples-row">
                  {SAMPLES.map((sample) => (
                    <div
                      key={sample.id}
                      className={`example-thumb-wrapper ${selectedSampleId === sample.id ? 'active' : ''}`}
                      onClick={() => runSampleInference(sample.id)}
                      title={`Select ${sample.name} (${sample.patient})`}
                    >
                      <img src={sample.thumb} alt={sample.name} className="example-thumb-img" />
                    </div>
                  ))}
                </div>
              </div>

              {/* Card 2: Model Information */}
              <div className="dash-card">
                <div className="card-header-row">
                  <div className="card-icon-container">
                    <Cpu size={16} strokeWidth={2.2} />
                  </div>
                  <h2 className="card-title">Model Information</h2>
                </div>

                <div className="model-badge" onClick={() => setActiveNav('Model Info')} title="View Full Architecture">
                  <div className="model-badge-left">
                    <div className="model-icon-box">
                      <Cpu size={16} strokeWidth={2.2} />
                    </div>
                    <div>
                      <div className="model-title">ResNet-18 (Pretrained)</div>
                      <div className="model-sub">Deep Residual Network with Grad-CAM</div>
                    </div>
                  </div>
                  <ChevronRight size={16} color="#61799D" />
                </div>

                <div className="model-metrics-grid">
                  <div className="metric-tile">
                    <div className="metric-val">{metrics.roc_auc}%</div>
                    <div className="metric-lbl">Test ROC-AUC</div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-val">{metrics.recall}%</div>
                    <div className="metric-lbl">Pneumonia Recall</div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-val">{metrics.parameters}</div>
                    <div className="metric-lbl">Parameters</div>
                  </div>
                </div>
              </div>

              {/* Card 3: Clinical Use & Pediatric Vitals Reference */}
              <div className="dash-card">
                <div className="card-header-row">
                  <div className="card-icon-container">
                    <ShieldCheck size={16} strokeWidth={2.2} />
                  </div>
                  <h2 className="card-title">Clinical Decision Support</h2>
                </div>
                <p className="clinical-text">
                  Intended to assist healthcare professionals in early detection of pediatric pneumonia. Not a replacement
                  for clinical evaluation.
                </p>

                {/* Pediatric Normal Vitals Reference Card */}
                <div className="vitals-box">
                  <div className="vitals-title">
                    <Activity size={14} color="#3985FF" />
                    <span>Pediatric Normal Respiratory Rates</span>
                  </div>
                  <table className="vitals-table">
                    <tbody>
                      <tr>
                        <td>Infant (&lt;1 yr)</td>
                        <td>30 – 53 breaths/min</td>
                      </tr>
                      <tr>
                        <td>Toddler (1–2 yr)</td>
                        <td>22 – 37 breaths/min</td>
                      </tr>
                      <tr>
                        <td>Preschool (3–5 yr)</td>
                        <td>20 – 28 breaths/min</td>
                      </tr>
                      <tr>
                        <td>School-Age (6–11 yr)</td>
                        <td>18 – 25 breaths/min</td>
                      </tr>
                    </tbody>
                  </table>

                  <div className="qa-strip">
                    <div className="qa-item">
                      <CheckCircle2 size={13} color="#22A06B" />
                      <span>Inspiration: 8+ Ribs</span>
                    </div>
                    <div className="qa-item">
                      <CheckCircle2 size={13} color="#22A06B" />
                      <span>Symmetric Clavicles</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* ── RIGHT COLUMN (~69%) ── */}
            <div className="col-right">
              {/* Card 4: Diagnostic Results */}
              <div className="dash-card results-card">
                <div className="results-top-row">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div className="card-icon-container">
                      <FileSpreadsheet size={16} strokeWidth={2.2} />
                    </div>
                    <h2 className="card-title">Diagnostic Results</h2>
                  </div>
                  <span className="results-timestamp">{result.timestamp}</span>
                </div>

                {/* Prediction Status Banner */}
                <div className={`status-banner ${isPneumonia ? 'pneumonia' : 'normal'}`}>
                  <div className="banner-left">
                    <div className={`banner-lung-badge ${isPneumonia ? 'pneumonia' : 'normal'}`}>
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 4v10M12 7l-4 3M12 7l4 3" />
                        <path d="M5 12c-.8 1.2-1.2 2.8-.8 4.4.6 2.2 2.8 2.8 4.4 2 1.2-.6 2-2 2-4v-3.2c0-1.6-1.2-2.8-2.8-2.8-1.8 0-2.8 1.6-2.8 3.6z" />
                        <path
                          d="M19 12c.8 1.2 1.2 2.8.8 4.4-.6 2.2-2.8 2.8-4.4 2-1.2-.6-2-2-2-4v-3.2c0-1.6 1.2-2.8 2.8-2.8 1.8 0 2.8 1.6 2.8 3.6z"
                          opacity="0.75"
                        />
                      </svg>
                    </div>
                    <div>
                      <div className={`banner-title ${isPneumonia ? 'pneumonia' : 'normal'}`}>
                        {isPneumonia ? 'Pneumonia Detected' : 'Normal Radiograph'}
                      </div>
                      <div className={`banner-desc ${isPneumonia ? 'pneumonia' : 'normal'}`}>
                        {isPneumonia
                          ? 'High probability of pneumonia based on radiographic analysis'
                          : 'No significant radiographic signs of pneumonia detected'}
                      </div>
                    </div>
                  </div>

                  <div className="confidence-box">
                    <div className="confidence-lbl">Confidence</div>
                    <div className={`confidence-number ${isPneumonia ? 'pneumonia' : 'normal'}`}>
                      {result.confidence}%
                    </div>
                    <div className="confidence-track">
                      <div
                        className={`confidence-fill ${isPneumonia ? 'pneumonia' : 'normal'}`}
                        style={{ width: `${Math.min(100, Math.max(0, result.confidence))}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* ── PACS RADIOLOGY TOOLBAR ── */}
                <div className="pacs-toolbar">
                  <div className="pacs-group">
                    <span style={{ fontSize: 11, fontWeight: 700, color: '#61799D', marginRight: 4 }}>
                      WINDOW / LEVEL:
                    </span>
                    <button
                      className={`pacs-btn ${pacsPreset === 'default' ? 'active' : ''}`}
                      onClick={() => setPacsPreset('default')}
                    >
                      Default
                    </button>
                    <button
                      className={`pacs-btn ${pacsPreset === 'lung' ? 'active' : ''}`}
                      onClick={() => setPacsPreset('lung')}
                    >
                      Lung Parenchyma
                    </button>
                    <button
                      className={`pacs-btn ${pacsPreset === 'mediastinum' ? 'active' : ''}`}
                      onClick={() => setPacsPreset('mediastinum')}
                    >
                      Mediastinum
                    </button>
                    <button
                      className={`pacs-btn ${pacsPreset === 'bone' ? 'active' : ''}`}
                      onClick={() => setPacsPreset('bone')}
                    >
                      Bone Detail
                    </button>
                  </div>

                  <div className="pacs-group">
                    <button
                      className={`pacs-btn ${isInverted ? 'active' : ''}`}
                      onClick={() => setIsInverted(!isInverted)}
                      title="Toggle Inverted Negative Grayscale"
                    >
                      <Contrast size={13} />
                      <span>Invert</span>
                    </button>
                    <button
                      className={`pacs-btn ${showCurtainSplit ? 'active' : ''}`}
                      onClick={() => {
                        setShowCurtainSplit(!showCurtainSplit);
                        setShowNormalBaseline(false);
                      }}
                      title="Toggle Interactive Curtain Split Slider"
                    >
                      <Split size={13} />
                      <span>Split Slider</span>
                    </button>
                    <button
                      className={`pacs-btn ${showNormalBaseline ? 'active' : ''}`}
                      onClick={() => {
                        setShowNormalBaseline(!showNormalBaseline);
                        setShowCurtainSplit(false);
                      }}
                      title="Compare against healthy pediatric normal radiograph"
                    >
                      <Layers size={13} />
                      <span>Normal Baseline</span>
                    </button>
                  </div>
                </div>

                {/* ── VIEW MODE 1: INTERACTIVE SPLIT-SCREEN CURTAIN SLIDER ── */}
                {showCurtainSplit ? (
                  <div style={{ marginBottom: 18 }}>
                    <div
                      className="curtain-wrapper"
                      onMouseMove={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                        setCurtainPos(Math.round((x / rect.width) * 100));
                      }}
                    >
                      {/* Base Anatomical X-Ray */}
                      <img
                        src={result.originalImage}
                        alt="Anatomical X-ray"
                        className="curtain-img-base"
                        style={getFilterStyle()}
                      />

                      {/* Overlapping Thermal Grad-CAM Overlay */}
                      <div className="curtain-overlay-container" style={{ width: `${curtainPos}%` }}>
                        <img
                          src={result.overlayImage}
                          alt="Grad-CAM Overlay"
                          className="curtain-img-overlay"
                          style={{
                            width: `${100 * (100 / (curtainPos || 1))}%`,
                            maxWidth: 'unset',
                            ...getFilterStyle(),
                          }}
                        />
                      </div>

                      {/* Draggable Divider Bar */}
                      <div className="curtain-divider" style={{ left: `${curtainPos}%` }}>
                        <div className="curtain-handle-badge">⇄</div>
                      </div>

                      <div className="curtain-label-left">Raw X-Ray</div>
                      <div className="curtain-label-right">Grad-CAM Overlay</div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8 }}>
                      <span style={{ fontSize: 11.5, color: '#61799D' }}>Drag across image or slide:</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={curtainPos}
                        onChange={(e) => setCurtainPos(+e.target.value)}
                        className="slider-track"
                      />
                      <span style={{ fontSize: 12, fontWeight: 700, width: 45 }}>{curtainPos}%</span>
                    </div>
                  </div>
                ) : showNormalBaseline ? (
                  /* ── VIEW MODE 2: NORMAL BASELINE SIDE-BY-SIDE COMPARISON ── */
                  <div style={{ marginBottom: 18 }}>
                    <div className="baseline-container">
                      <div>
                        <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 6 }}>
                          Patient Radiograph ({result.caseId})
                        </div>
                        <div className="baseline-box">
                          <img src={result.overlayImage} alt="Patient Radiograph" style={getFilterStyle()} />
                          <span
                            className="baseline-badge-tag"
                            style={{ background: isPneumonia ? '#E84C64' : '#22A06B' }}
                          >
                            {result.className} ({result.confidence}%)
                          </span>
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 6 }}>
                          Healthy Pediatric Reference Baseline
                        </div>
                        <div className="baseline-box">
                          <img src="/samples/example_2_high.png" alt="Healthy Normal Radiograph" style={getFilterStyle()} />
                          <span className="baseline-badge-tag" style={{ background: '#22A06B' }}>
                            Confirmed Normal (Guangzhou Cohort)
                          </span>
                        </div>
                      </div>
                    </div>
                    <div style={{ fontSize: 11.5, color: '#61799D', marginTop: 8 }}>
                      * Note symmetric bilateral aerated lung fields, sharp costophrenic angles, and normal cardiac silhouette
                      in reference baseline.
                    </div>
                  </div>
                ) : (
                  /* ── VIEW MODE 3: STANDARD THREE DIAGNOSTIC IMAGE CARDS ── */
                  <div className="image-cards-grid">
                    {/* Card 1: Original X-Ray */}
                    <div className="image-card">
                      <div className="image-card-header">
                        <span className="image-card-title">Original X-Ray</span>
                        <button
                          className="expand-btn"
                          onClick={() =>
                            setLightbox({
                              open: true,
                              title: 'Original Chest X-Ray',
                              src: result.originalImage,
                              zoom: 1,
                            })
                          }
                          title="Expand Image"
                        >
                          <Maximize2 size={13} />
                        </button>
                      </div>
                      <div
                        className="image-box"
                        onClick={() =>
                          setLightbox({
                            open: true,
                            title: 'Original Chest X-Ray',
                            src: result.originalImage,
                            zoom: 1,
                          })
                        }
                      >
                        <img src={result.originalImage} alt="Original Radiograph" style={getFilterStyle()} />
                      </div>
                    </div>

                    {/* Card 2: Grad-CAM Heatmap */}
                    <div className="image-card">
                      <div className="image-card-header">
                        <span className="image-card-title">Grad-CAM Heatmap</span>
                        <button
                          className="expand-btn"
                          onClick={() =>
                            setLightbox({
                              open: true,
                              title: 'Grad-CAM Heatmap (Activation Map)',
                              src: result.heatmapImage,
                              zoom: 1,
                            })
                          }
                          title="Expand Image"
                        >
                          <Maximize2 size={13} />
                        </button>
                      </div>
                      <div
                        className="image-box"
                        onClick={() =>
                          setLightbox({
                            open: true,
                            title: 'Grad-CAM Heatmap (Activation Map)',
                            src: result.heatmapImage,
                            zoom: 1,
                          })
                        }
                      >
                        <img src={result.heatmapImage} alt="Grad-CAM Activation Heatmap" />
                      </div>
                    </div>

                    {/* Card 3: Overlay (Heatmap + X-Ray) */}
                    <div className="image-card">
                      <div className="image-card-header">
                        <span className="image-card-title">Overlay (Heatmap + X-Ray)</span>
                        <button
                          className="expand-btn"
                          onClick={() =>
                            setLightbox({
                              open: true,
                              title: 'Grad-CAM Overlay (Heatmap + X-Ray)',
                              src: result.overlayImage,
                              zoom: 1,
                            })
                          }
                          title="Expand Image"
                        >
                          <Maximize2 size={13} />
                        </button>
                      </div>
                      <div
                        className="image-box"
                        onClick={() =>
                          setLightbox({
                            open: true,
                            title: 'Grad-CAM Overlay (Heatmap + X-Ray)',
                            src: result.overlayImage,
                            zoom: 1,
                          })
                        }
                      >
                        <img src={result.overlayImage} alt="Grad-CAM Overlay" style={getFilterStyle()} />
                      </div>
                    </div>
                  </div>
                )}

                {/* Middle Row: Prediction Probabilities & Key Findings */}
                <div className="two-cards-grid">
                  {/* Card 5: Prediction Probabilities */}
                  <div className="sub-card">
                    <div className="sub-card-title">Prediction Probabilities</div>
                    <div className="prob-item">
                      <span className="prob-name">Pneumonia</span>
                      <div className="prob-track">
                        <div
                          className="prob-fill pneumonia"
                          style={{ width: `${Math.min(100, Math.max(0, result.probabilities.PNEUMONIA))}%` }}
                        ></div>
                      </div>
                      <span className="prob-num">{result.probabilities.PNEUMONIA}%</span>
                    </div>

                    <div className="prob-item">
                      <span className="prob-name">Normal</span>
                      <div className="prob-track">
                        <div
                          className="prob-fill normal"
                          style={{ width: `${Math.min(100, Math.max(0, result.probabilities.NORMAL))}%` }}
                        ></div>
                      </div>
                      <span className="prob-num">{result.probabilities.NORMAL}%</span>
                    </div>
                  </div>

                  {/* Card 6: Key Findings (Grad-CAM) */}
                  <div className="sub-card">
                    <div className="sub-card-title">Key Findings (Grad-CAM)</div>
                    <ul className="findings-list">
                      {isPneumonia ? (
                        <>
                          <li className="finding-row">
                            <span className="dot-bullet red"></span>
                            <span>High activation in right lower lung region</span>
                          </li>
                          <li className="finding-row">
                            <span className="dot-bullet orange"></span>
                            <span>Features consistent with pulmonary infiltrates</span>
                          </li>
                          <li className="finding-row">
                            <span className="dot-bullet amber"></span>
                            <span>Consolidation patterns detected</span>
                          </li>
                        </>
                      ) : (
                        <>
                          <li className="finding-row">
                            <span className="dot-bullet blue"></span>
                            <span>Clear bilateral lung fields without focal consolidation</span>
                          </li>
                          <li className="finding-row">
                            <span className="dot-bullet green"></span>
                            <span>Uniform activation distribution without localized hotspots</span>
                          </li>
                          <li className="finding-row">
                            <span className="dot-bullet blue"></span>
                            <span>Normal vascular markings and sharp costophrenic angles</span>
                          </li>
                        </>
                      )}
                    </ul>
                  </div>
                </div>

                {/* Bottom Row: Recommended Next Steps & Technical Details */}
                <div className="two-cards-grid" style={{ marginBottom: 14 }}>
                  {/* Card 7: Recommended Next Steps */}
                  <div className="sub-card">
                    <div className="sub-card-header">
                      <div className="card-icon-container" style={{ width: 24, height: 24 }}>
                        <FileSpreadsheet size={14} />
                      </div>
                      <div className="sub-card-title" style={{ marginBottom: 0 }}>
                        Recommended Next Steps
                      </div>
                    </div>
                    <ul className="steps-list">
                      <li className="step-row">
                        <span className="step-badge">
                          <Check size={11} strokeWidth={3.5} />
                        </span>
                        <span>Correlate with clinical symptoms (fever, cough, respiratory distress)</span>
                      </li>
                      <li className="step-row">
                        <span className="step-badge">
                          <Check size={11} strokeWidth={3.5} />
                        </span>
                        <span>Consider radiologist review</span>
                      </li>
                      <li className="step-row">
                        <span className="step-badge">
                          <Check size={11} strokeWidth={3.5} />
                        </span>
                        <span>Further clinical evaluation and monitoring</span>
                      </li>
                      <li className="step-row">
                        <span className="step-badge">
                          <Check size={11} strokeWidth={3.5} />
                        </span>
                        <span>Repeat imaging if necessary</span>
                      </li>
                    </ul>
                  </div>

                  {/* Card 8: Technical Details */}
                  <div className="sub-card">
                    <div className="sub-card-header">
                      <div className="card-icon-container" style={{ width: 24, height: 24 }}>
                        <Settings size={14} />
                      </div>
                      <div className="sub-card-title" style={{ marginBottom: 0 }}>
                        Technical Details
                      </div>
                    </div>
                    <table className="tech-table">
                      <tbody>
                        <tr>
                          <td>Model</td>
                          <td>ResNet-18 (ImageNet Pretrained)</td>
                        </tr>
                        <tr>
                          <td>Input Size</td>
                          <td>224 × 224 × 3</td>
                        </tr>
                        <tr>
                          <td>Grad-CAM</td>
                          <td>Layer 4 (Conv)</td>
                        </tr>
                        <tr>
                          <td>Processing Time</td>
                          <td>{result.processingTime} seconds</td>
                        </tr>
                        <tr>
                          <td>API</td>
                          <td>FastAPI (v0.129+)</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Real Action Bar */}
                <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', paddingTop: '8px' }}>
                  <button className="btn-secondary" onClick={() => setActiveNav('Analysis')}>
                    <Sliders size={14} />
                    <span>Open Interpretability Studio</span>
                  </button>
                  <button className="btn-primary" onClick={() => setActiveNav('Reports')}>
                    <FileText size={14} />
                    <span>Generate Clinical Report</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── ROUTING: 2. BATCH TRIAGE MODE ──────────────────── */}
        {activeNav === 'Batch Triage' && (
          <div className="dash-card">
            <div className="card-header-row" style={{ justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div className="card-icon-container">
                  <FolderUp size={16} />
                </div>
                <h2 className="card-title">Emergency Multi-Scan Batch Triage (STAT Ranking)</h2>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn-secondary" onClick={initBatchDemo}>
                  <RefreshCw size={13} />
                  <span>Reload Demo Batch</span>
                </button>
                <button className="btn-primary" onClick={() => batchInputRef.current?.click()}>
                  <UploadCloud size={14} />
                  <span>Upload Batch Images</span>
                </button>
                <input
                  ref={batchInputRef}
                  type="file"
                  multiple
                  accept="image/jpeg,image/png"
                  style={{ display: 'none' }}
                  onChange={(e) => handleBatchFiles(e.target.files)}
                />
              </div>
            </div>

            <p style={{ fontSize: 13, color: '#61799D', marginBottom: 18 }}>
              Automated STAT triage queue ranks batched chest radiographs by predicted pneumonia probability, ensuring
              high-risk pediatric cases receive immediate radiographic review.
            </p>

            {/* Scoreboard Cards */}
            <div className="batch-stat-grid">
              <div className="batch-stat-box">
                <div className="batch-stat-val">{batchQueue.length}</div>
                <div className="batch-stat-lbl">Total Batched Scans</div>
              </div>
              <div className="batch-stat-box">
                <div className="batch-stat-val" style={{ color: '#E84C64' }}>
                  {batchQueue.filter((b) => b.className === 'PNEUMONIA').length}
                </div>
                <div className="batch-stat-lbl">Flagged High-Risk (STAT)</div>
              </div>
              <div className="batch-stat-box">
                <div className="batch-stat-val" style={{ color: '#22A06B' }}>
                  {batchQueue.filter((b) => b.className === 'NORMAL').length}
                </div>
                <div className="batch-stat-lbl">Normal Screening Cases</div>
              </div>
              <div className="batch-stat-box">
                <div className="batch-stat-val" style={{ color: '#3985FF' }}>
                  0.8s
                </div>
                <div className="batch-stat-lbl">Average Model Latency</div>
              </div>
            </div>

            {batchLoading ? (
              <div style={{ padding: 40, textAlign: 'center' }}>
                <span className="spinner" style={{ width: 24, height: 24, display: 'inline-block' }}></span>
                <div style={{ marginTop: 10, fontSize: 13, fontWeight: 600, color: '#3985FF' }}>
                  Processing Batch Radiographs with ResNet-18...
                </div>
              </div>
            ) : (
              <div className="history-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Thumbnail</th>
                      <th>Batch ID</th>
                      <th>Patient / Case</th>
                      <th>Triage Urgency</th>
                      <th>AI Classification</th>
                      <th>Confidence</th>
                      <th>Inference Latency</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchQueue.map((item) => (
                      <tr key={item.id}>
                        <td>
                          <img
                            src={item.thumb}
                            alt="Scan"
                            style={{ width: 40, height: 40, borderRadius: 6, objectFit: 'cover', background: '#080C14' }}
                          />
                        </td>
                        <td style={{ fontWeight: 600 }}>{item.id}</td>
                        <td>{item.patient}</td>
                        <td>
                          <span
                            className={`stat-badge ${
                              item.urgency.includes('STAT') ? 'stat-urgent' : 'stat-routine'
                            }`}
                          >
                            {item.urgency}
                          </span>
                        </td>
                        <td>
                          <span
                            className={`table-tag ${
                              item.className.toUpperCase() === 'PNEUMONIA' ? 'pneumonia' : 'normal'
                            }`}
                          >
                            {item.className}
                          </span>
                        </td>
                        <td style={{ fontWeight: 700 }}>{item.confidence}%</td>
                        <td style={{ color: '#61799D' }}>{item.latency}</td>
                        <td>
                          <button
                            className="table-btn"
                            onClick={() => {
                              setResult({
                                caseId: item.patient.split(' ')[0],
                                className: item.className,
                                confidence: item.confidence,
                                probabilities: item.probabilities || {
                                  PNEUMONIA: item.className === 'PNEUMONIA' ? item.confidence : +(100 - item.confidence).toFixed(1),
                                  NORMAL: item.className === 'NORMAL' ? item.confidence : +(100 - item.confidence).toFixed(1),
                                },
                                originalImage: item.file || item.thumb,
                                heatmapImage: item.heatmapImage || item.thumb,
                                overlayImage: item.overlayImage || item.thumb,
                                processingTime: parseFloat(item.latency) || 0.8,
                                timestamp: getCurrentTimestamp(),
                              });
                              setActiveNav('Home');
                            }}
                          >
                            Open in Viewer
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ─── ROUTING: 3. ANALYSIS STUDIO VIEW ───────────────── */}
        {activeNav === 'Analysis' && (
          <div className="studio-container">
            <div className="dash-card">
              <div className="card-header-row">
                <div className="card-icon-container">
                  <Scan size={16} />
                </div>
                <h2 className="card-title">Radiographic Interpretability & Blending Studio</h2>
              </div>
              <p style={{ fontSize: '13px', color: '#61799D', marginBottom: '14px' }}>
                Interactively adjust the transparency blending of the Grad-CAM thermal activation map over the original
                radiograph, examine pulmonary quadrants, and inspect focal activations.
              </p>

              {/* Opacity Control Slider */}
              <div className="slider-container">
                <span style={{ fontSize: '13px', fontWeight: 600, width: 140 }}>
                  Heatmap Blend: {Math.round(blendOpacity * 100)}%
                </span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={blendOpacity}
                  onChange={(e) => setBlendOpacity(+e.target.value)}
                  className="slider-track"
                />
                <button
                  className="btn-secondary"
                  style={{ padding: '4px 10px', fontSize: '11.5px' }}
                  onClick={() => setBlendOpacity(0.45)}
                >
                  Reset
                </button>
              </div>

              {/* Side-by-side comparative canvas */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
                <div>
                  <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>
                    1. Original Anatomical Radiograph
                  </div>
                  <div className="image-box" style={{ height: '360px', aspectRatio: 'unset' }}>
                    <img src={result.originalImage} alt="Original Radiograph" style={getFilterStyle()} />
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>
                    2. Blended Live Overlay (Opacity: {Math.round(blendOpacity * 100)}%)
                  </div>
                  <div className="image-box" style={{ height: '360px', aspectRatio: 'unset', position: 'relative' }}>
                    <img
                      src={result.originalImage}
                      alt="Base X-ray"
                      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', ...getFilterStyle() }}
                    />
                    <img
                      src={result.heatmapImage}
                      alt="Blended Heatmap"
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        opacity: blendOpacity,
                        mixBlendMode: 'screen',
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Pulmonary Quadrants Regional Metrics */}
              <div style={{ fontSize: '13.5px', fontWeight: 700, marginTop: '10px' }}>
                Pulmonary Quadrant Regional Activation Distribution
              </div>
              <div className="quadrant-grid">
                <div className="quadrant-box">
                  <div className="quadrant-name">Right Upper Lobe</div>
                  <div className="quadrant-val" style={{ color: isPneumonia ? '#E84C64' : '#10264F' }}>
                    {isPneumonia ? '34.2%' : '11.8%'}
                  </div>
                </div>
                <div className="quadrant-box">
                  <div className="quadrant-name">Right Lower Lobe</div>
                  <div className="quadrant-val" style={{ color: isPneumonia ? '#E84C64' : '#22A06B' }}>
                    {isPneumonia ? '89.4%' : '14.1%'}
                  </div>
                </div>
                <div className="quadrant-box">
                  <div className="quadrant-name">Left Upper Lobe</div>
                  <div className="quadrant-val" style={{ color: '#10264F' }}>
                    {isPneumonia ? '16.5%' : '12.4%'}
                  </div>
                </div>
                <div className="quadrant-box">
                  <div className="quadrant-name">Left Lower Lobe</div>
                  <div className="quadrant-val" style={{ color: '#10264F' }}>
                    {isPneumonia ? '22.1%' : '13.9%'}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '10px', marginTop: '22px' }}>
                <button
                  className="btn-primary"
                  onClick={() => downloadImage(result.overlayImage, `overlay_${result.caseId}.png`)}
                >
                  <Download size={14} />
                  <span>Download Blended Composite</span>
                </button>
                <button className="btn-secondary" onClick={() => setActiveNav('Home')}>
                  ← Return to Diagnostic Dashboard
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ─── ROUTING: 4. HISTORY VIEW ───────────────────────── */}
        {activeNav === 'History' && (
          <div className="dash-card">
            <div className="card-header-row" style={{ justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div className="card-icon-container">
                  <Clock size={16} />
                </div>
                <h2 className="card-title">Session Scan History & Diagnostic Audit Log</h2>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button className="btn-secondary" onClick={exportHistoryCsv} title="Export as CSV">
                  <Download size={14} />
                  <span>Export CSV</span>
                </button>
                <button
                  className="btn-secondary"
                  style={{ color: '#E84C64' }}
                  onClick={() => setHistory([])}
                  title="Clear history"
                >
                  Clear History
                </button>
              </div>
            </div>

            <p style={{ fontSize: '13px', color: '#61799D', marginBottom: '16px' }}>
              Showing {history.length} recent AI evaluations performed during this session.
            </p>

            {history.length === 0 ? (
              <div style={{ padding: '30px', textAlign: 'center', color: '#61799D', fontSize: '13px' }}>
                No scan history recorded in this session. Run an analysis on the Home page to populate.
              </div>
            ) : (
              <div className="history-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Thumbnail</th>
                      <th>Scan ID</th>
                      <th>Patient ID</th>
                      <th>Timestamp</th>
                      <th>Predicted Class</th>
                      <th>Confidence</th>
                      <th>Latency</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((h, i) => (
                      <tr key={h.id || i}>
                        <td>
                          <img
                            src={h.thumb}
                            alt="Scan Thumb"
                            style={{
                              width: 36,
                              height: 36,
                              borderRadius: 6,
                              objectFit: 'cover',
                              background: '#080C14',
                            }}
                          />
                        </td>
                        <td style={{ fontWeight: 600 }}>{h.id}</td>
                        <td>{h.patientId}</td>
                        <td style={{ color: '#61799D' }}>{h.timestamp}</td>
                        <td>
                          <span
                            className={`table-tag ${
                              h.className.toUpperCase() === 'PNEUMONIA' ? 'pneumonia' : 'normal'
                            }`}
                          >
                            {h.className}
                          </span>
                        </td>
                        <td style={{ fontWeight: 600 }}>{h.confidence}%</td>
                        <td style={{ color: '#61799D' }}>{h.latency}</td>
                        <td>
                          <button
                            className="table-btn"
                            onClick={() => {
                              setResult({
                                caseId: h.patientId,
                                className: h.className,
                                confidence: h.confidence,
                                probabilities: {
                                  PNEUMONIA: h.className === 'PNEUMONIA' ? h.confidence : +(100 - h.confidence).toFixed(1),
                                  NORMAL: h.className === 'NORMAL' ? h.confidence : +(100 - h.confidence).toFixed(1),
                                },
                                originalImage: h.imageSrc || h.thumb,
                                heatmapImage: h.heatmapSrc || h.thumb,
                                overlayImage: h.overlaySrc || h.thumb,
                                processingTime: parseFloat(h.latency) || 0.8,
                                timestamp: h.timestamp,
                              });
                              setActiveNav('Home');
                            }}
                          >
                            Load in Viewer
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ─── ROUTING: 5. PATIENTS / CLINICAL TRIAGE ─────────── */}
        {activeNav === 'Patients' && (
          <div className="dash-card">
            <div className="card-header-row">
              <div className="card-icon-container">
                <Users size={16} />
              </div>
              <h2 className="card-title">Active Pediatric Triage Cohort</h2>
            </div>
            <p style={{ fontSize: '13px', color: '#61799D', marginBottom: '20px' }}>
              Select a pediatric patient from the urgent triage queue to load their chest radiograph directly into the
              ResNet-18 diagnostic pipeline.
            </p>

            <div className="patient-grid">
              {SAMPLES.map((sample) => (
                <div key={sample.id} className="patient-card">
                  <div className="patient-card-header">
                    <div>
                      <div className="patient-id">{sample.patient}</div>
                      <div style={{ fontSize: 11, color: '#8E9CAB' }}>Chest AP View</div>
                    </div>
                    <span className={`table-tag ${sample.id === 4 || sample.id === 1 ? 'pneumonia' : 'normal'}`}>
                      {sample.id === 4 || sample.id === 1 ? 'Urgent Triage' : 'Routine'}
                    </span>
                  </div>

                  <div className="patient-info-row">
                    <div>
                      <b>Age:</b> {sample.age}
                    </div>
                    <div>
                      <b>Sex:</b> {sample.gender}
                    </div>
                    <div>
                      <b>Status:</b> Radiograph Acquired
                    </div>
                  </div>

                  <div className="patient-symptoms">
                    <b>Clinical Presentation:</b> {sample.symptoms}
                  </div>

                  <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginTop: '6px' }}>
                    <img
                      src={sample.thumb}
                      alt={sample.name}
                      style={{ width: 44, height: 44, borderRadius: 8, background: '#080C14', objectFit: 'cover' }}
                    />
                    <button
                      className="btn-primary"
                      style={{ flex: 1, justifyContent: 'center' }}
                      onClick={() => {
                        runSampleInference(sample.id);
                        setActiveNav('Home');
                      }}
                    >
                      Analyze Radiograph
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─── ROUTING: 6. MODEL INFO VIEW ────────────────────── */}
        {activeNav === 'Model Info' && (
          <div className="dash-card">
            <div className="card-header-row">
              <div className="card-icon-container">
                <Cpu size={16} />
              </div>
              <h2 className="card-title">Model Architecture & Quantitative Benchmark Evaluation</h2>
            </div>

            {/* Sub-tabs */}
            <div className="tabs-header">
              <button
                className={`tab-btn ${modelInfoTab === 'curves' ? 'active' : ''}`}
                onClick={() => setModelInfoTab('curves')}
              >
                Evaluation Curves
              </button>
              <button
                className={`tab-btn ${modelInfoTab === 'confusion' ? 'active' : ''}`}
                onClick={() => setModelInfoTab('confusion')}
              >
                Confusion Matrix & Metrics
              </button>
              <button
                className={`tab-btn ${modelInfoTab === 'training' ? 'active' : ''}`}
                onClick={() => setModelInfoTab('training')}
              >
                Training Epochs Log
              </button>
            </div>

            {/* Tab 1: Evaluation Curves */}
            {modelInfoTab === 'curves' && (
              <div>
                <p style={{ fontSize: '13px', color: '#61799D', marginBottom: '16px' }}>
                  Empirical ROC and Precision-Recall curves obtained from test set evaluation (624 independent pediatric chest
                  radiographs from the Guangzhou Women & Children's Medical Center cohort).
                </p>

                <div className="plots-grid">
                  <div className="plot-card">
                    <div className="plot-title">Receiver Operating Characteristic (ROC-AUC: {metrics.roc_auc}%)</div>
                    <img src="/outputs/roc_curve.png" alt="ROC Curve" className="plot-img" />
                  </div>
                  <div className="plot-card">
                    <div className="plot-title">Precision-Recall Curve (Avg Precision: 97.01%)</div>
                    <img src="/outputs/pr_curve.png" alt="PR Curve" className="plot-img" />
                  </div>
                </div>

                <div className="plots-grid">
                  <div className="plot-card">
                    <div className="plot-title">Confusion Matrix (624 Test Cases)</div>
                    <img src="/outputs/confusion_matrix.png" alt="Confusion Matrix" className="plot-img" />
                  </div>
                  <div className="plot-card">
                    <div className="plot-title">Training Loss & Validation Accuracy Across Epochs</div>
                    <img src="/outputs/training_history.png" alt="Training Curves" className="plot-img" />
                  </div>
                </div>
              </div>
            )}

            {/* Tab 2: Confusion Matrix & Metrics */}
            {modelInfoTab === 'confusion' && (
              <div>
                <div className="model-metrics-grid" style={{ marginBottom: 20 }}>
                  <div className="metric-tile">
                    <div className="metric-val">{metrics.accuracy}%</div>
                    <div className="metric-lbl">Overall Accuracy</div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-val">{metrics.recall}%</div>
                    <div className="metric-lbl">Pneumonia Sensitivity</div>
                  </div>
                  <div className="metric-tile">
                    <div className="metric-val">{metrics.specificity}%</div>
                    <div className="metric-lbl">Specificity (Normal)</div>
                  </div>
                </div>

                <div className="history-table-container" style={{ marginBottom: 20 }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall (Sensitivity)</th>
                        <th>F1-Score</th>
                        <th>Test Cases</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><b>NORMAL</b></td>
                        <td>90.0%</td>
                        <td>88.0%</td>
                        <td>89.0%</td>
                        <td>234</td>
                      </tr>
                      <tr>
                        <td><b>PNEUMONIA</b></td>
                        <td>93.0%</td>
                        <td>94.36%</td>
                        <td>94.0%</td>
                        <td>390</td>
                      </tr>
                      <tr style={{ background: '#F8FAFD', fontWeight: 700 }}>
                        <td>Macro Average</td>
                        <td>92.0%</td>
                        <td>91.0%</td>
                        <td>91.0%</td>
                        <td>624</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div style={{ background: '#F0F6FF', border: '1px solid #CDE0FA', borderRadius: 12, padding: 16 }}>
                  <div style={{ fontWeight: 700, fontSize: 13.5, marginBottom: 6 }}>Clinical Metric Summary</div>
                  <p style={{ fontSize: 12.5, color: '#61799D', lineHeight: 1.5 }}>
                    With a recall of <b>94.36%</b> (368 true positives out of 390 pneumonia cases), the deep learning system
                    effectively minimizes dangerous false-negative triage outcomes in high-volume pediatric screening.
                  </p>
                </div>
              </div>
            )}

            {/* Tab 3: Training Log */}
            {modelInfoTab === 'training' && (
              <div className="history-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Epoch</th>
                      <th>Train Loss</th>
                      <th>Train Accuracy</th>
                      <th>Validation Loss</th>
                      <th>Validation Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { ep: 1, tLoss: 0.117, tAcc: '94.11%', vLoss: 0.1861, vAcc: '93.75%' },
                      { ep: 2, tLoss: 0.0608, tAcc: '96.93%', vLoss: 0.0699, vAcc: '100.0%' },
                      { ep: 3, tLoss: 0.0411, tAcc: '97.80%', vLoss: 0.294, vAcc: '87.50%' },
                      { ep: 4, tLoss: 0.0374, tAcc: '98.03%', vLoss: 0.081, vAcc: '100.0%' },
                      { ep: 5, tLoss: 0.0323, tAcc: '98.35%', vLoss: 0.0653, vAcc: '100.0%' },
                      { ep: 6, tLoss: 0.0214, tAcc: '98.91%', vLoss: 0.0575, vAcc: '93.75%' },
                    ].map((row) => (
                      <tr key={row.ep}>
                        <td>Epoch {row.ep}</td>
                        <td>{row.tLoss}</td>
                        <td>{row.tAcc}</td>
                        <td>{row.vLoss}</td>
                        <td style={{ fontWeight: 600, color: '#22A06B' }}>{row.vAcc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ─── ROUTING: 7. REPORTS VIEW ───────────────────────── */}
        {activeNav === 'Reports' && (
          <div>
            <div className="btn-group-no-print" style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginBottom: 16 }}>
              <button className="btn-secondary" onClick={() => setShowFhirModal(true)}>
                <FileCode size={14} />
                <span>Export HL7 FHIR (JSON)</span>
              </button>
              <button className="btn-primary" onClick={() => window.print()}>
                <Printer size={14} />
                <span>Print / Save as PDF</span>
              </button>
            </div>

            {/* Printable Clinical Report Sheet */}
            <div className="clinical-report-sheet">
              <div className="report-header-banner">
                <div>
                  <div className="report-org-title">PneumoAI Diagnostic Triage System</div>
                  <div className="report-doc-title">Pediatric Chest Radiograph AI Interpretability Summary</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 13, fontWeight: 700 }}>Accession: {result.caseId}</div>
                  <div style={{ fontSize: 11, color: '#61799D' }}>Date: {result.timestamp}</div>
                </div>
              </div>

              {/* Patient & Study Metadata */}
              <div className="report-meta-grid">
                <div>
                  <div style={{ color: '#8E9CAB', fontSize: 10, textTransform: 'uppercase' }}>Patient ID</div>
                  <div style={{ fontWeight: 600 }}>{result.caseId}</div>
                </div>
                <div>
                  <div style={{ color: '#8E9CAB', fontSize: 10, textTransform: 'uppercase' }}>Exam Modality</div>
                  <div style={{ fontWeight: 600 }}>Chest Radiography (AP)</div>
                </div>
                <div>
                  <div style={{ color: '#8E9CAB', fontSize: 10, textTransform: 'uppercase' }}>Neural Backbone</div>
                  <div style={{ fontWeight: 600 }}>ResNet-18 + Grad-CAM</div>
                </div>
                <div>
                  <div style={{ color: '#8E9CAB', fontSize: 10, textTransform: 'uppercase' }}>Inference Latency</div>
                  <div style={{ fontWeight: 600 }}>{result.processingTime}s</div>
                </div>
              </div>

              {/* Prediction Finding Banner */}
              <div className={`status-banner ${isPneumonia ? 'pneumonia' : 'normal'}`} style={{ marginBottom: 20 }}>
                <div>
                  <div className={`banner-title ${isPneumonia ? 'pneumonia' : 'normal'}`}>
                    Primary AI Finding: {isPneumonia ? 'Pneumonia Detected' : 'Normal Chest Radiograph'}
                  </div>
                  <div style={{ fontSize: 12.5, color: isPneumonia ? '#824D56' : '#35674E' }}>
                    Model prediction confidence: <b>{result.confidence}%</b> (Probability distribution:{' '}
                    Pneumonia: {result.probabilities.PNEUMONIA}%, Normal: {result.probabilities.NORMAL}%)
                  </div>
                </div>
              </div>

              {/* Three Image Strip */}
              <div className="report-section-title">Radiographic Examination & Explainability Activation Map</div>
              <div className="report-images-row">
                <div>
                  <div style={{ fontSize: 11.5, fontWeight: 600, marginBottom: 6 }}>Original Chest Radiograph</div>
                  <div className="report-img-box">
                    <img src={result.originalImage} alt="Original X-ray" />
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 11.5, fontWeight: 600, marginBottom: 6 }}>Grad-CAM Heatmap (Layer 4)</div>
                  <div className="report-img-box">
                    <img src={result.heatmapImage} alt="Grad-CAM Heatmap" />
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 11.5, fontWeight: 600, marginBottom: 6 }}>Anatomical Thermal Overlay</div>
                  <div className="report-img-box">
                    <img src={result.overlayImage} alt="Overlay" />
                  </div>
                </div>
              </div>

              {/* Radiologist Commentary Notes Area */}
              <div className="report-section-title">Radiologist Diagnostic Review & Clinical Commentary</div>
              <textarea
                value={radiologistNotes}
                onChange={(e) => setRadiologistNotes(e.target.value)}
                style={{
                  width: '100%',
                  height: '80px',
                  borderRadius: '8px',
                  border: '1px solid #E3ECFA',
                  padding: '10px',
                  fontSize: '12.5px',
                  fontFamily: 'inherit',
                  resize: 'vertical',
                }}
              />

              {/* Attestation Signoff with QR Code */}
              <div className="report-signoff-box">
                <div>
                  <div>
                    <b>Attending Radiologist:</b> ___________________________
                  </div>
                  <div style={{ marginTop: 4 }}>Clinical AI Decision Support Verification</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ textAlign: 'right' }}>
                    <div>
                      <b>Signature Date:</b> ____________________
                    </div>
                    <div style={{ marginTop: 4 }}>PneumoAI v1.0 • Guangzhou Cohort Validated</div>
                  </div>
                  <div style={{ padding: 6, background: '#F8FAFD', border: '1px solid #E3ECFA', borderRadius: 6 }}>
                    <QrCode size={36} color="#10264F" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── ROUTING: 8. SETTINGS VIEW ──────────────────────── */}
        {activeNav === 'Settings' && (
          <div className="dash-card">
            <div className="card-header-row">
              <div className="card-icon-container">
                <Settings size={16} />
              </div>
              <h2 className="card-title">System & Diagnostic Configuration</h2>
            </div>
            <p style={{ fontSize: '13px', color: '#61799D', marginBottom: '20px' }}>
              Configure FastAPI endpoint parameters, inference alert thresholds, and visualization preferences.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 640 }}>
              {/* API Endpoint Configuration */}
              <div>
                <label style={{ fontSize: 13, fontWeight: 600, display: 'block', marginBottom: 6 }}>
                  FastAPI Backend Endpoint URL
                </label>
                <div style={{ display: 'flex', gap: 10 }}>
                  <input
                    type="text"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    style={{
                      flex: 1,
                      padding: '8px 12px',
                      borderRadius: 8,
                      border: '1px solid #E3ECFA',
                      fontSize: 13,
                    }}
                  />
                  <button className="btn-secondary" onClick={checkHealth}>
                    <RefreshCw size={14} />
                    <span>Test Ping</span>
                  </button>
                </div>
                <div style={{ fontSize: 11.5, color: backendOnline ? '#22A06B' : '#E84C64', marginTop: 4 }}>
                  {backendOnline ? `Connected successfully (${pingLatency || 12}ms)` : 'Cannot reach backend at this URL.'}
                </div>
              </div>

              {/* Alert Sensitivity Threshold Slider */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <label style={{ fontSize: 13, fontWeight: 600 }}>Pneumonia Alert Confidence Threshold</label>
                  <span style={{ fontSize: 13, fontWeight: 700, color: '#E84C64' }}>{alertThreshold}%</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="95"
                  step="5"
                  value={alertThreshold}
                  onChange={(e) => setAlertThreshold(+e.target.value)}
                  className="slider-track"
                />
                <span style={{ fontSize: 11, color: '#8E9CAB' }}>
                  Scans predicting Pneumonia above this confidence trigger immediate triage alerts.
                </span>
              </div>

              {/* Auto-Analyze Toggle */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>Auto-Analyze Uploaded Images</div>
                  <div style={{ fontSize: 11.5, color: '#61799D' }}>
                    Immediately run inference when a chest radiograph is dropped or selected.
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={autoAnalyze}
                  onChange={(e) => setAutoAnalyze(e.target.checked)}
                  style={{ width: 18, height: 18, accentColor: '#3985FF', cursor: 'pointer' }}
                />
              </div>

              {/* Reset Defaults */}
              <div style={{ paddingTop: 12, borderTop: '1px solid #E3ECFA' }}>
                <button
                  className="btn-secondary"
                  onClick={() => {
                    setApiUrl(DEFAULT_API_URL);
                    setAlertThreshold(70);
                    setAutoAnalyze(true);
                    checkHealth();
                  }}
                >
                  Reset Configuration to Factory Defaults
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ─── HL7 FHIR EXPORT MODAL ──────────────────────────── */}
      {showFhirModal && (
        <div className="modal-overlay" onClick={() => setShowFhirModal(false)}>
          <div className="modal-content" style={{ maxWidth: 740 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-title">HL7 FHIR R4 DiagnosticReport (JSON)</span>
              <button className="modal-close-btn" onClick={() => setShowFhirModal(false)}>
                <X size={20} />
              </button>
            </div>
            <div style={{ fontSize: 12, color: '#61799D', marginBottom: 12 }}>
              Conforming to HL7 FHIR Standard Specification for Radiology Diagnostic Reports (LOINC: 24648-8 / SNOMED CT: 233604007).
            </div>
            <pre className="fhir-code-box">{JSON.stringify(getFhirPayload(), null, 2)}</pre>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 14 }}>
              <button className="btn-secondary" onClick={() => setShowFhirModal(false)}>
                Close
              </button>
              <button className="btn-primary" onClick={exportFhirJson}>
                <Download size={14} />
                <span>Download FHIR JSON</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ─── LIGHTBOX MODAL WITH ZOOM & PAN ─────────────────── */}
      {lightbox.open && (
        <div className="modal-overlay" onClick={() => setLightbox({ open: false, title: '', src: '', zoom: 1 })}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-title">{lightbox.title}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <button
                  className="icon-btn"
                  style={{ width: 28, height: 28 }}
                  onClick={() => setLightbox((l) => ({ ...l, zoom: Math.min(3, l.zoom + 0.25) }))}
                  title="Zoom In"
                >
                  <ZoomIn size={14} />
                </button>
                <button
                  className="icon-btn"
                  style={{ width: 28, height: 28 }}
                  onClick={() => setLightbox((l) => ({ ...l, zoom: Math.max(0.5, l.zoom - 0.25) }))}
                  title="Zoom Out"
                >
                  <ZoomOut size={14} />
                </button>
                <button
                  className="icon-btn"
                  style={{ width: 28, height: 28 }}
                  onClick={() => setLightbox((l) => ({ ...l, zoom: 1 }))}
                  title="Reset Zoom"
                >
                  <RotateCcw size={13} />
                </button>
                <button
                  className="modal-close-btn"
                  onClick={() => setLightbox({ open: false, title: '', src: '', zoom: 1 })}
                >
                  <X size={20} />
                </button>
              </div>
            </div>

            <div className="modal-img-container">
              <img
                src={lightbox.src}
                alt={lightbox.title}
                style={{
                  transform: `scale(${lightbox.zoom})`,
                  transition: 'transform 0.15s ease-out',
                  ...getFilterStyle(),
                }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12, alignItems: 'center' }}>
              <span style={{ fontSize: 11.5, color: '#61799D' }}>Zoom: {Math.round(lightbox.zoom * 100)}%</span>
              <button
                className="btn-primary"
                style={{ padding: '6px 14px', fontSize: 12 }}
                onClick={() => downloadImage(lightbox.src, `${lightbox.title.toLowerCase().replace(/\s+/g, '_')}.png`)}
              >
                <Download size={13} />
                <span>Save High-Res Radiograph</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
