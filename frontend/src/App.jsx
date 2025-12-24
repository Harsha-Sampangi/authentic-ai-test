import { useState, useEffect } from 'react'

function App() {
  const [currentScreen, setCurrentScreen] = useState('landing')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  // Handle file selection with video support
  const handleFileSelect = (file) => {
    if (file) {
      const isImage = file.type.startsWith('image/')
      const isVideo = file.type.startsWith('video/')
      
      if (isImage || isVideo) {
        setSelectedFile(file)
        setPreviewUrl(URL.createObjectURL(file))
      } else {
        alert('Please upload an image or video file')
      }
    }
  }

  // Handle drag & drop
  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0])
    }
  }

  // Analyze image or video
  const handleAnalyze = async () => {
    setCurrentScreen('analyzing')
    
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      
      // Determine if image or video
      const isVideo = selectedFile.type.startsWith('video/')
      const endpoint = isVideo ? '/api/analyze-video' : '/api/analyze'
      
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error('Analysis failed')
      }
      
      const data = await response.json()
      
      console.log('Backend response:', data)
      
    setResult({
      score: data.authenticity_score,
      status: data.is_deepfake ? 'HIGHLY SUSPICIOUS' : 'LIKELY AUTHENTIC',
      filename: selectedFile.name,
      timestamp: new Date().toISOString(),
      alerts: data.alerts || [],
      report: data.report,
      type: isVideo ? 'video' : 'image',
      // Video-specific data
      frameCount: data.frame_count,
      suspiciousFrames: data.suspicious_frames,
      suspiciousPercentage: data.suspicious_percentage,
      timeline: data.timeline,
      suspiciousSegments: data.suspicious_segments,
      scoreRange: data.score_range
    })
    setResult({
  score: data.authenticity_score,
  status: data.is_deepfake ? 'HIGHLY SUSPICIOUS' : 'LIKELY AUTHENTIC',
  filename: selectedFile.name,
  timestamp: new Date().toISOString(),
  alerts: data.alerts || [],
  report: data.report,
  type: isVideo ? 'video' : 'image',
  frameCount: data.frame_count,
  suspiciousFrames: data.suspicious_frames,
  suspiciousPercentage: data.suspicious_percentage,
  timeline: data.timeline,
  suspiciousSegments: data.suspicious_segments,
  scoreRange: data.score_range,
  // NEW: Add these two lines
    forensicBreakdown: data.forensic_breakdown ? {
    sections: data.forensic_breakdown.sections || [],
    confidenceBreakdown: {
      visual_consistency: data.forensic_breakdown.confidence_breakdown?.visual_consistency || 0,
      temporal_analysis: data.forensic_breakdown.confidence_breakdown?.temporal_analysis || null,
      noise_compression: data.forensic_breakdown.confidence_breakdown?.noise_compression || 0,
      metadata_integrity: data.forensic_breakdown.confidence_breakdown?.metadata_integrity || 0
    },
    explanation: data.forensic_breakdown.explanation || ''
  } : null,
})
      
      setCurrentScreen('results')
    } catch (error) {
      console.error('Analysis failed:', error)
      alert('Analysis failed. Make sure backend is running on port 8000')
      setCurrentScreen('landing')
    }
  }

  // Reset to landing
  const resetToLanding = () => {
    setCurrentScreen('landing')
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
  }

  return (
    <div className="min-h-screen bg-[#0a0e17]">
      {/* Navigation */}
      <nav className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gray-700 rounded flex items-center justify-center text-xl">
              📁
            </div>
            <span className="text-xl font-bold">Authenti.AI</span>
          </div>
          <div className="flex items-center gap-6">
            <button 
              onClick={() => setCurrentScreen('landing')}
              className="text-sm text-gray-400 hover:text-white"
            >
              Dashboard
            </button>
            <button 
              onClick={() => setCurrentScreen('landing')}
              className="text-sm text-gray-400 hover:text-white"
            >
              Analysis
            </button>
            <button 
              onClick={() => setCurrentScreen('history')}
              className="text-sm text-gray-400 hover:text-white"
            >
              History
            </button>
            <button className="text-sm text-gray-400 hover:text-white">Settings</button>
            <button className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center">
              ?
            </button>
            <button className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center">
              👤
            </button>
          </div>
        </div>
      </nav>

      {/* Landing Screen */}
      {currentScreen === 'landing' && (
        <div className="max-w-5xl mx-auto px-6 py-20">
          <div 
            className="relative rounded-3xl overflow-hidden"
            style={{
              background: 'linear-gradient(135deg, #1a4d5c 0%, #2d7a8a 50%, #4fa8b8 100%)',
              minHeight: '500px'
            }}
          >
            <div className="absolute inset-0 bg-black/20" />
            
            <div className="relative z-10 flex flex-col items-center justify-center py-20 px-8 text-center">
              <h1 className="text-5xl font-bold mb-4 leading-tight">
                Verify Digital Reality with AI Truth Analysis
              </h1>
              <p className="text-lg text-white/90 mb-12 max-w-2xl">
                Upload media to detect deepfakes, manipulated evidence, and misinformation.
              </p>

              {/* Drag & Drop Area */}
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`relative w-full max-w-md mb-8 transition-all ${
                  dragActive ? 'scale-105' : ''
                }`}
              >
                <input
                  type="file"
                  accept="image/*,video/*"
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                  className="hidden"
                  id="file-input"
                />
                <label
                  htmlFor="file-input"
                  className="block cursor-pointer bg-cyan-400 hover:bg-cyan-300 text-gray-900 font-semibold py-5 px-8 rounded-xl transition-all shadow-lg"
                >
                  {selectedFile ? selectedFile.name : 'Drag & Drop Image or Video'}
                </label>
              </div>

              {/* Preview */}
              {selectedFile && (
                <div className="mb-8">
                  {selectedFile.type.startsWith('video/') ? (
                    <video 
                      src={previewUrl} 
                      controls 
                      className="max-h-64 mx-auto rounded-lg"
                    />
                  ) : (
                    <img 
                      src={previewUrl} 
                      alt="Preview" 
                      className="max-h-64 mx-auto rounded-lg"
                    />
                  )}
                </div>
              )}

              {selectedFile && (
                <button
                  onClick={handleAnalyze}
                  className="bg-white text-gray-900 font-semibold py-4 px-12 rounded-xl hover:bg-gray-100 transition-all shadow-lg mb-8"
                >
                  Analyze Now
                </button>
              )}

              <p className="text-sm text-white/70">
                Supported formats: JPG, PNG, MP4, MOV, AVI
              </p>
            </div>
          </div>

          {/* Feature Badges */}
          <div className="flex justify-center gap-4 mt-8">
            <div className="bg-gray-800/80 px-6 py-3 rounded-xl text-sm">
              Cryptographic verification
            </div>
            <div className="bg-gray-800/80 px-6 py-3 rounded-xl text-sm">
              Explainable AI
            </div>
            <div className="bg-gray-800/80 px-6 py-3 rounded-xl text-sm">
              Chain-of-Custody
            </div>
          </div>

          {/* Footer */}
          <div className="flex justify-center gap-12 mt-16 text-sm text-gray-500">
            <button className="hover:text-gray-300">About</button>
            <button className="hover:text-gray-300">Security</button>
            <button className="hover:text-gray-300">Legal</button>
            <button className="hover:text-gray-300">Contact</button>
          </div>
          <p className="text-center text-xs text-gray-600 mt-6">
            © 2024 Authenti.AI. All rights reserved.
          </p>
        </div>
      )}

      {/* Analyzing Screen */}
      {currentScreen === 'analyzing' && (
        <AnalyzingScreen isVideo={selectedFile?.type.startsWith('video/')} />
      )}

      {/* Results Screen */}
      {currentScreen === 'results' && result && (
        <ResultsScreen result={result} previewUrl={previewUrl} onReset={resetToLanding} />
      )}

      {/* History Screen */}
      {currentScreen === 'history' && (
        <HistoryScreen onViewReport={() => setCurrentScreen('results')} />
      )}
    </div>
  )
}

// Analyzing Screen Component
function AnalyzingScreen({ isVideo }) {
  const [progress, setProgress] = useState({
    artifact: 0,
    noise: 0,
    metadata: 0,
    deepfake: 0
  })

  useEffect(() => {
    const timers = []
    
    // Artifact Inspection
    timers.push(setTimeout(() => {
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev.artifact >= 100) {
            clearInterval(interval)
            return prev
          }
          return { ...prev, artifact: prev.artifact + 2 }
        })
      }, 30)
    }, 0))

    // Noise Analysis
    timers.push(setTimeout(() => {
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev.noise >= 100) {
            clearInterval(interval)
            return prev
          }
          return { ...prev, noise: prev.noise + 2.5 }
        })
      }, 30)
    }, 1500))

    // Metadata Verification
    timers.push(setTimeout(() => {
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev.metadata >= 100) {
            clearInterval(interval)
            return prev
          }
          return { ...prev, metadata: prev.metadata + 3 }
        })
      }, 30)
    }, 2500))

    // Deepfake Detection
    timers.push(setTimeout(() => {
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev.deepfake >= 100) {
            clearInterval(interval)
            return prev
          }
          return { ...prev, deepfake: prev.deepfake + 2 }
        })
      }, 30)
    }, 3500))

    return () => timers.forEach(t => clearTimeout(t))
  }, [])

  return (
    <div className="max-w-4xl mx-auto px-6 py-20">
      <h1 className="text-3xl font-bold text-center mb-4">
        {isVideo ? 'Analyzing video frames...' : 'Analyzing media...'} Running forensic intelligence...
      </h1>

      {/* Spinning Animation */}
      <div className="flex justify-center mb-16">
        <div className="relative w-96 h-96">
          <div className="absolute inset-0 rounded-full" style={{
            background: 'conic-gradient(from 0deg, #d4af37, #c5a028, #d4af37)',
            animation: 'spin 3s linear infinite'
          }} />
          <div className="absolute inset-8 rounded-full bg-[#0a0e17]" />
        </div>
      </div>

      {/* Progress Bars */}
      <div className="space-y-8">
        <ProgressBar label={isVideo ? "Frame Extraction" : "Artifact Inspection"} progress={progress.artifact} />
        <ProgressBar label="Noise Analysis" progress={progress.noise} />
        <ProgressBar label="Metadata Verification" progress={progress.metadata} />
        <ProgressBar label="Deepfake Detection" progress={progress.deepfake} />
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}

function ProgressBar({ label, progress }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-2">
        <span className="text-white">{label}</span>
        <span className="text-gray-400">{Math.round(progress)}%</span>
      </div>
      <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
        <div 
          className="h-full bg-gradient-to-r from-cyan-400 to-teal-400 transition-all duration-300"
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
      </div>
    </div>
  )
}

// Results Screen Component
function ResultsScreen({ result, previewUrl, onReset }) {
  const [animatedScore, setAnimatedScore] = useState(0)

  useEffect(() => {
    const duration = 2000
    const steps = 60
    const increment = result.score / steps
    let current = 0

    const timer = setInterval(() => {
      current++
      setAnimatedScore(prev => Math.min(prev + increment, result.score))
      if (current >= steps) clearInterval(timer)
    }, duration / steps)

    return () => clearInterval(timer)
  }, [result.score])

  const getScoreColor = (score) => {
    if (score >= 70) return 'text-green-400'
    if (score >= 40) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getScoreBg = (score) => {
    if (score >= 70) return 'bg-green-600'
    if (score >= 40) return 'bg-yellow-600'
    return 'bg-red-600'
  }

  const getBgGradient = (score) => {
    if (score >= 70) return 'from-green-950 to-green-900/20 border-green-500/30'
    if (score >= 40) return 'from-yellow-950 to-yellow-900/20 border-yellow-500/30'
    return 'from-red-950 to-red-900/20 border-red-500/30'
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Analysis Complete</h1>
        <p className="text-sm text-gray-400">
          File: {result.filename} • {new Date(result.timestamp).toLocaleString()}
          {result.type === 'video' && ` • ${result.frameCount} frames analyzed`}
        </p>
        <div className="flex gap-3 mt-4">
          <button className="bg-cyan-400 hover:bg-cyan-300 text-gray-900 font-semibold px-6 py-2 rounded-lg text-sm">
            Download Report
          </button>
          <button className="bg-gray-700 hover:bg-gray-600 text-white px-6 py-2 rounded-lg text-sm">
            Export JSON
          </button>
          <button className="bg-gray-700 hover:bg-gray-600 text-white px-6 py-2 rounded-lg text-sm">
            Export CSV
          </button>
          <button onClick={onReset} className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm">
            Share
          </button>
        </div>
      </div>

      <h2 className="text-2xl font-bold mb-6">Analysis Results</h2>

      {/* Main Results Grid */}
      <div className="grid md:grid-cols-[2fr_1fr] gap-6 mb-6">
        {/* Score Card */}
        <div className={`bg-gradient-to-br ${getBgGradient(result.score)} rounded-2xl p-8 border`}>
          <div className="flex items-start justify-between mb-8">
            <div>
              <div className={`text-6xl font-bold mb-2 ${getScoreColor(result.score)}`}>
                {Math.round(animatedScore)}%
              </div>
              <div className={`text-xl font-semibold mb-1 ${getScoreColor(result.score)}`}>
                {result.status}
              </div>
              <div className="text-sm text-gray-400 mt-1">Authenticity Score</div>
              
              {/* Video-specific stats */}
              {result.type === 'video' && result.scoreRange && (
                <div className="mt-4 text-xs text-gray-400 space-y-1">
                  <div>Range: {result.scoreRange.min}% - {result.scoreRange.max}%</div>
                  <div>Variation: ±{result.scoreRange.std}%</div>
                </div>
              )}
            </div>
            {previewUrl && (
              result.type === 'video' ? (
                <video src={previewUrl} className="w-32 h-32 object-cover rounded-lg" />
              ) : (
                <img src={previewUrl} alt="Preview" className="w-32 h-32 object-cover rounded-lg" />
              )
            )}
          </div>
          <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className={`h-full ${getScoreBg(result.score)}`}
              style={{ width: `${result.score}%` }}
            />
          </div>
        </div>

        {/* Issue Alerts */}
        <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4">Issue Alerts</h3>
          {result.alerts && result.alerts.length > 0 ? (
            <div className="space-y-3">
              {result.alerts.map((alert, idx) => (
                <div key={idx} className="flex items-center gap-3 bg-gray-800 p-3 rounded-lg">
                  <span className="text-2xl">{alert.icon}</span>
                  <div className="flex-1">
                    <div className={`text-xs font-semibold mb-1 ${
                      alert.severity === 'High' ? 'text-red-400' : 
                      alert.severity === 'Medium' ? 'text-yellow-400' : 'text-blue-400'
                    }`}>
                      {alert.severity}
                    </div>
                    <div className="text-sm">{alert.title}</div>
                  </div>
                  <button className="text-gray-400 hover:text-white">ℹ️</button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-400">No significant issues detected</p>
          )}
        </div>
      </div>

      {/* Video Timeline (only for videos) */}
      {result.type === 'video' && result.timeline && (
        <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700 mb-6">
          <h3 className="text-lg font-semibold mb-4">Frame-by-Frame Timeline</h3>
          
          <div className="mb-4">
            <p className="text-sm text-gray-400">
              Analyzed {result.frameCount} frames • {result.suspiciousFrames} suspicious ({result.suspiciousPercentage}%)
            </p>
          </div>
          
          {/* Timeline visualization */}
          <div className="flex gap-1 h-20 items-end bg-gray-900 rounded-lg p-2">
            {result.timeline.map((point, idx) => (
              <div
                key={idx}
                className={`flex-1 rounded-t transition-all cursor-pointer hover:opacity-80 ${
                  point.suspicious ? 'bg-red-500' : 'bg-green-500'
                }`}
                style={{ height: `${point.score}%` }}
                title={`${point.timestamp.toFixed(1)}s: ${point.score.toFixed(1)}% ${point.suspicious ? '(Suspicious)' : '(Authentic)'}`}
              />
            ))}
          </div>
          
          <div className="flex justify-between text-xs text-gray-500 mt-2">
            <span>0:00</span>
            <span>Timeline</span>
            <span>{result.timeline[result.timeline.length - 1]?.timestamp.toFixed(1)}s</span>
          </div>
          
          {/* Suspicious segments */}
          {result.suspiciousSegments && result.suspiciousSegments.length > 0 && (
            <div className="mt-6">
              <h4 className="text-sm font-semibold mb-3">⚠️ Suspicious Segments Detected</h4>
              <div className="space-y-2">
                {result.suspiciousSegments.map((segment, idx) => (
                  <div key={idx} className="bg-red-900/20 border border-red-500/30 rounded-lg p-3">
                    <div className="flex justify-between text-sm">
                      <span className="flex items-center gap-2">
                        <span className="text-red-400">⏱️</span>
                        {segment.start_time.toFixed(1)}s - {segment.end_time.toFixed(1)}s
                      </span>
                      <span className="text-red-400">
                        {segment.frame_count} frames • avg {segment.avg_score.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Media Viewer */}
      <div className="bg-gray-800/50 rounded-2xl border border-gray-700 overflow-hidden mb-6">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold">Media Viewer</h3>
        </div>
        <div className="bg-black/50 aspect-video flex items-center justify-center">
          {previewUrl ? (
            result.type === 'video' ? (
              <video src={previewUrl} controls className="max-h-full" />
            ) : (
              <img src={previewUrl} alt="Analyzed media" className="max-h-full" />
            )
          ) : (
            <div className="text-gray-500">Preview not available</div>
          )}
        </div>
      </div>

      {/* Credibility Report */}
      <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700 mb-6">
        <h3 className="text-lg font-semibold mb-4">Credibility Report</h3>
        <p className="text-sm text-gray-300 leading-relaxed mb-4">{result.report}</p>
        <div className={`${
          result.score < 50 ? 'bg-red-900/20 border-red-500/30' : 'bg-green-900/20 border-green-500/30'
        } border rounded-lg p-4`}>
          <p className={`text-sm font-semibold mb-2 ${
            result.score < 50 ? 'text-red-300' : 'text-green-300'
          }`}>
            {result.score < 50 ? 'Advisory Notes: Conduct further investigation if necessary.' : 'Content appears authentic with no critical issues.'}
          </p>
          <p className="text-xs text-gray-400">
            {result.score < 50 
              ? 'Its credibility is compromised, and it should not be used as a reliable source without verification.'
              : 'While the content appears authentic, always verify from multiple sources for critical decisions.'
            }
          </p>
        </div>
      </div>
      {/* Explainable Forensic Breakdown */}
{result.forensicBreakdown && (
  <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700 mb-6">
    <h3 className="text-lg font-semibold mb-6">Explainable Forensic Breakdown</h3>
    
    {/* Accordion Sections */}
    <div className="space-y-3 mb-6">
      {result.forensicBreakdown.sections.map((section, idx) => (
        <ForensicSection key={idx} section={section} />
      ))}
    </div>
    
    {/* Confidence Breakdown */}
    <div className="mb-6">
      <h4 className="text-sm font-semibold mb-4">Decision Confidence Breakdown</h4>
      <div className="space-y-3">
        <ConfidenceBar 
          label="Visual Consistency" 
          value={result.forensicBreakdown.confidenceBreakdown.visual_consistency} 
        />
        {result.forensicBreakdown.confidenceBreakdown.temporal_analysis && (
          <ConfidenceBar 
            label="Temporal Analysis" 
            value={result.forensicBreakdown.confidenceBreakdown.temporal_analysis} 
          />
        )}
        <ConfidenceBar 
          label="Noise & Compression" 
          value={result.forensicBreakdown.confidenceBreakdown.noise_compression} 
        />
        <ConfidenceBar 
          label="Metadata Integrity" 
          value={result.forensicBreakdown.confidenceBreakdown.metadata_integrity} 
        />
      </div>
    </div>
    
    {/* AI Explanation */}
    <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
      <p className="text-xs font-semibold text-blue-300 mb-2">🤖 AI Explanation</p>
      <p className="text-sm text-gray-300 leading-relaxed">
        {result.forensicBreakdown.explanation}
      </p>
    </div>
  </div>
)}

{/* Evidence Integrity Record */}
{result.evidenceIntegrity && (
  <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700 mb-6">
    <h3 className="text-lg font-semibold mb-6">Evidence Integrity Record</h3>
    
    <div className="space-y-4">
      {/* File Hash */}
      <div>
        <p className="text-xs text-gray-400 mb-1">File Hash (SHA-256)</p>
        <p className="text-sm font-mono text-gray-300 break-all">
          {result.evidenceIntegrity.fileHash}
        </p>
      </div>
      
      {/* Timestamps */}
      <div>
        <p className="text-xs text-gray-400 mb-1">Upload Timestamp</p>
        <p className="text-sm text-gray-300">{result.evidenceIntegrity.uploadTime.ist}</p>
        <p className="text-xs text-gray-500">{result.evidenceIntegrity.uploadTime.utc}</p>
      </div>
      
      {/* Analysis Engine */}
      <div>
        <p className="text-xs text-gray-400 mb-1">Analysis Engine</p>
        <p className="text-sm text-gray-300">{result.evidenceIntegrity.analysisEngine}</p>
      </div>
      
      {/* Integrity Status */}
      <div>
        <p className="text-xs text-gray-400 mb-1">File Integrity Status</p>
        <p className="text-sm text-green-400 flex items-center gap-2">
          <span>✔</span> {result.evidenceIntegrity.integrityStatus}
        </p>
      </div>
      
      {/* Re-analysis History */}
      <div>
        <p className="text-xs text-gray-400 mb-1">Re-analysis History</p>
        <p className="text-sm text-gray-300">
          {result.evidenceIntegrity.reanalysisCount} re-analysis attempts
        </p>
      </div>
      
      {/* Trust Level */}
      <div>
        <p className="text-xs text-gray-400 mb-1">Evidence Trust Level</p>
        <span className={`inline-block px-3 py-1 rounded text-sm font-semibold ${
          result.evidenceIntegrity.trustLevel === 'HIGH' ? 'bg-green-900/30 text-green-400' :
          result.evidenceIntegrity.trustLevel === 'MEDIUM' ? 'bg-yellow-900/30 text-yellow-400' :
          'bg-red-900/30 text-red-400'
        }`}>
          {result.evidenceIntegrity.trustLevel}
        </span>
      </div>
      
      {/* Audit Log */}
      <details className="cursor-pointer">
        <summary className="text-sm text-gray-400 hover:text-white">▼ View Audit Log</summary>
        <div className="mt-3 space-y-2 pl-4 border-l-2 border-gray-700">
          {result.evidenceIntegrity.auditLog.map((log, idx) => (
            <div key={idx} className="text-xs text-gray-400">
              • {log.event} — {log.time}
            </div>
          ))}
        </div>
      </details>
    </div>
  </div>
)}

      {/* Techniques Used */}
      <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Analysis Techniques Used</h3>
        <div className="flex flex-wrap gap-3">
          {result.type === 'video' ? (
            <>
              <span className="bg-gray-700 px-4 py-2 rounded-lg text-sm">Frame Extraction</span>
              <span className="bg-gray-700 px-4 py-2 rounded-lg text-sm">Temporal Analysis</span>
              <span className="bg-gray-700 px-4 py-2 rounded-lg text-sm">Face Detection</span>
              <span className="bg-gray-700 px-4 py-2 rounded-lg text-sm">Deepfake Detection</span>
              <span className="bg-gray-700 px-4 py-2 rounded-lg text-sm">Metadata Analysis</span>
            </>
          ) : (
            <>
              <span className="bg-gray-700 px-4 py-2 rounded-lg text-sm">ELA</span>
              <span className="bg-gray-700 px-4 py-2 rounded-lg text-sm">Noise Spectrum</span>
              <span className="bg-gray-700 px-4 py-2 rounded-lg text-sm">Lighting Consistency</span>
              <span className="bg-gray-700 px-4 py-2 rounded-lg text-sm">Metadata Analysis</span>
            </>
          )}
        </div>
      </div>

      {/* Back Button */}
      <button
        onClick={onReset}
        className="mt-8 bg-blue-600 hover:bg-blue-700 text-white font-semibold px-8 py-3 rounded-lg"
      >
        Analyze Another File
      </button>
    </div>
  )
}

// History Screen Component
function HistoryScreen({ onViewReport }) {
  const mockHistory = [
    { id: 1, name: 'Filename.mp4', type: 'Video', date: '2024-01-20 14:30', thumbnail: '🎥', score: 34 },
    { id: 2, name: 'Image.jpg', type: 'Image', date: '2024-01-19 10:15', thumbnail: '🖼️', score: 87 },
    { id: 3, name: 'Video2.mov', type: 'Video', date: '2024-01-18 16:45', thumbnail: '🎬', score: 62 }
  ]

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="grid md:grid-cols-[2fr_1fr] gap-8">
        {/* Left: History List */}
        <div>
          <h1 className="text-3xl font-bold mb-8">History</h1>
          
          {/* Search Bar */}
          <div className="relative mb-6">
            <input
              type="text"
              placeholder="Search by filename, date, status"
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 pl-10 text-sm"
            />
            <span className="absolute left-3 top-3.5 text-gray-400">🔍</span>
          </div>

          {/* Filters */}
          <div className="flex gap-3 mb-8">
            <button className="bg-gray-700 px-4 py-2 rounded-lg text-sm">Image ▼</button>
            <button className="bg-gray-700 px-4 py-2 rounded-lg text-sm">Video ▼</button>
            <button className="bg-gray-700 px-4 py-2 rounded-lg text-sm">All ▼</button>
          </div>

                   {/* Records */}
          <h2 className="text-xl font-semibold mb-4">Analysis Records</h2>
          <div className="space-y-4">
            {mockHistory.map(item => (
              <div key={item.id} className="bg-gray-800/50 border border-gray-700 rounded-xl p-4 flex items-center gap-4">
                <div className="w-24 h-24 bg-gray-700 rounded-lg flex items-center justify-center text-4xl">
                  {item.thumbnail}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold">{item.name}</h3>
                  <p className="text-sm text-gray-400">{item.type} • {item.date}</p>
                </div>
                <button 
                  onClick={onViewReport}
                  className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg text-sm flex items-center gap-2"
                >
                  View Report 👁️
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Quick Analytics */}
        <div>
          <h2 className="text-xl font-semibold mb-6">Quick Analytics</h2>
          
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 mb-4">
            <h3 className="text-sm text-gray-400 mb-2">Total Scans</h3>
            <div className="text-4xl font-bold">120</div>
          </div>

          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 mb-6">
            <h3 className="text-sm text-gray-400 mb-2">Suspect %</h3>
            <div className="text-4xl font-bold text-red-400">15%</div>
          </div>

          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-sm text-gray-400 mb-4">Last 30 Days</h3>
            <div className="h-32 flex items-end gap-1">
              {[40, 65, 52, 80, 45, 70, 90, 55, 75, 85, 60, 95, 70].map((h, i) => (
                <div 
                  key={i} 
                  className="flex-1 bg-teal-500 rounded-t"
                  style={{ height: `${h}%` }}
                />
              ))}
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-2">
              <span>Jan 1</span>
              <span>Jan 8</span>
              <span>Jan 15</span>
              <span>Jan 22</span>
              <span>Jan 29</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
// Forensic Section Accordion Component
function ForensicSection({ section }) {
  const [isOpen, setIsOpen] = useState(true)
  
  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 bg-gray-800 flex items-center justify-between hover:bg-gray-750 transition-colors"
      >
        <span className="text-sm font-semibold">{section.title}</span>
        <span className="text-gray-400">{isOpen ? '▼' : '▶'}</span>
      </button>
      
      {isOpen && (
        <div className="px-4 py-3 space-y-2">
          {section.findings.map((finding, idx) => (
            <div key={idx} className="flex items-start gap-2 text-sm">
              <span className={`mt-0.5 ${
                finding.status === 'pass' ? 'text-green-400' :
                finding.status === 'warning' ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {finding.status === 'pass' ? '✔' : 
                 finding.status === 'warning' ? '⚠' : '✖'}
              </span>
              <span className="text-gray-300">{finding.text}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Confidence Bar Component
function ConfidenceBar({ label, value }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-300 font-semibold">{Math.round(value)}%</span>
      </div>
      <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
        <div 
          className={`h-full transition-all ${
            value >= 80 ? 'bg-green-500' :
            value >= 60 ? 'bg-yellow-500' :
            'bg-red-500'
          }`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  )
}

export default App