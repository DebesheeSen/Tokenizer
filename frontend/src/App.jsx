import { useState, useEffect, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || ''

function App() {
  const [activeTab, setActiveTab] = useState('workspace')
  
  // Model management states
  const [models, setModels] = useState([])
  const [activeModel, setActiveModel] = useState('')
  const [isModelsLoading, setIsModelsLoading] = useState(true)
  const [uploadError, setUploadError] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef(null)

  // Tokenization states
  const [inputText, setInputText] = useState('Tokenization is the process of breaking text into smaller units called tokens.')
  const [tokenResult, setTokenResult] = useState(null)
  const [isTokenizing, setIsTokenizing] = useState(false)
  const [tokenizeError, setTokenizeError] = useState('')

  // Training states
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState('')
  const [vocabSize, setVocabSize] = useState(1000)
  const [trainModelName, setTrainModelName] = useState('custom_tokenizer.pkl')
  const [trainingState, setTrainingState] = useState({
    status: 'idle', // 'idle', 'training', 'completed', 'error'
    progress: 0,
    total: 0,
    message: '',
    error: ''
  })
  const trainIntervalRef = useRef(null)

  // Pastel colors for token chips
  const pastelColors = [
    'bg-pink-100/60 border-pink-200 text-pink-700',
    'bg-blue-100/60 border-blue-200 text-blue-700',
    'bg-emerald-100/60 border-emerald-200 text-emerald-700',
    'bg-purple-100/60 border-purple-200 text-purple-700',
    'bg-amber-100/60 border-amber-200 text-amber-700'
  ]

  // Fetch models list
  const fetchModels = async (selectModelName = null) => {
    setIsModelsLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/models`)
      if (res.ok) {
        const data = await res.json()
        setModels(data)
        if (data.length > 0) {
          // If a model is specified, select it; otherwise select the first model or keep current
          if (selectModelName) {
            setActiveModel(selectModelName)
          } else if (!activeModel || !data.some(m => m.name === activeModel)) {
            // Prefer oliver_twist.pkl or monte_cristo.pkl if available
            const defaultModel = data.find(m => m.name === 'oliver_twist.pkl') || data.find(m => m.name === 'monte_cristo.pkl') || data[0]
            setActiveModel(defaultModel.name)
          }
        }
      }
    } catch (err) {
      console.error('Failed to fetch models:', err)
    } finally {
      setIsModelsLoading(false)
    }
  }

  // Fetch datasets list
  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/datasets`)
      if (res.ok) {
        const data = await res.json()
        setDatasets(data)
        if (data.length > 0) {
          setSelectedDataset(data[0].name)
        }
      }
    } catch (err) {
      console.error('Failed to fetch datasets:', err)
    }
  }

  // Handle tokenize API call
  const handleTokenize = async (textToTokenize = inputText, modelName = activeModel) => {
    if (!textToTokenize || !modelName) return
    setIsTokenizing(true)
    setTokenizeError('')
    try {
      const res = await fetch(`${API_BASE}/api/tokenize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToTokenize, model: modelName })
      })
      if (res.ok) {
        const data = await res.json()
        setTokenResult(data)
      } else {
        const errData = await res.json()
        setTokenizeError(errData.detail || 'Failed to tokenize text')
      }
    } catch (err) {
      setTokenizeError('Connection error to backend')
      console.error(err)
    } finally {
      setIsTokenizing(false)
    }
  }

  // Handle drag and drop upload
  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    
    if (!file.name.endsWith('.pkl')) {
      setUploadError('Only .pkl files are supported.')
      return
    }

    setUploadError('')
    setIsUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData
      })
      if (res.ok) {
        const data = await res.json()
        fetchModels(data.filename)
      } else {
        const errData = await res.json()
        setUploadError(errData.detail || 'Upload failed.')
      }
    } catch (err) {
      setUploadError('Error uploading file.')
      console.error(err)
    } finally {
      setIsUploading(false)
    }
  }

  // Handle BPE tokenizer training
  const handleTrain = async (e) => {
    e.preventDefault()
    if (!selectedDataset || !vocabSize || !trainModelName) return

    setTrainingState({
      status: 'training',
      progress: 0,
      total: 0,
      message: 'Starting training request...',
      error: ''
    })

    try {
      const res = await fetch(`${API_BASE}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset: selectedDataset,
          vocab_size: parseInt(vocabSize),
          model_name: trainModelName
        })
      })
      
      if (res.ok) {
        // Start polling status
        pollTrainingStatus()
      } else {
        const errData = await res.json()
        setTrainingState({
          status: 'error',
          progress: 0,
          total: 0,
          message: 'Failed to start training',
          error: errData.detail || 'Unknown error'
        })
      }
    } catch (err) {
      setTrainingState({
        status: 'error',
        progress: 0,
        total: 0,
        message: 'Network error',
        error: err.message || 'Unknown network error'
      })
    }
  }

  // Polling helper
  const pollTrainingStatus = () => {
    if (trainIntervalRef.current) clearInterval(trainIntervalRef.current)

    trainIntervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/train/status`)
        if (res.ok) {
          const status = await res.json()
          setTrainingState(status)
          
          if (status.status === 'completed') {
            clearInterval(trainIntervalRef.current)
            // Refresh models list and set the newly trained model as active
            const modelName = trainModelName.endsWith('.pkl') ? trainModelName : `${trainModelName}.pkl`
            fetchModels(modelName)
            // Auto switch back to workspace
            setTimeout(() => {
              setActiveTab('workspace')
              // Reset status
              fetch(`${API_BASE}/api/train/reset`, { method: 'POST' })
            }, 1500)
          } else if (status.status === 'error') {
            clearInterval(trainIntervalRef.current)
          }
        }
      } catch (err) {
        console.error('Error polling status:', err)
      }
    }, 500)
  }

  // Clean up interval on unmount
  useEffect(() => {
    return () => {
      if (trainIntervalRef.current) clearInterval(trainIntervalRef.current)
    }
  }, [])

  // Initial load
  useEffect(() => {
    fetchModels()
    fetchDatasets()
  }, [])

  // Re-run tokenization when active model or input changes
  useEffect(() => {
    if (activeModel) {
      handleTokenize(inputText, activeModel)
    }
  }, [activeModel])

  // Debounced tokenize on typing
  useEffect(() => {
    const timer = setTimeout(() => {
      if (inputText && activeModel) {
        handleTokenize(inputText, activeModel)
      }
    }, 400) // 400ms debounce
    return () => clearTimeout(timer)
  }, [inputText])

  // Helper to visually format whitespaces for display in chip
  const renderTokenText = (text) => {
    if (text === ' ') return '·'
    if (text === '\n') return '↵'
    if (text === '\t') return '⇥'
    return text.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥')
  }

  return (
    <div className="min-h-screen flex flex-col bg-surface font-sans text-on-surface">
      {/* TopAppBar */}
      <header className="fixed top-0 w-full z-50 backdrop-blur-xl bg-surface/80 shadow-sm flex items-center justify-between px-8 h-16 border-b border-outline-variant/10">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-2xl" data-icon="analytics">analytics</span>
          <span className="font-headline-md text-xl font-semibold text-primary">Tokenizer</span>
        </div>
        <nav className="hidden md:flex gap-8 items-center">
          <button 
            onClick={() => setActiveTab('workspace')}
            className={`font-label-caps text-xs tracking-wider uppercase font-semibold transition-colors ${activeTab === 'workspace' ? 'text-primary' : 'text-outline hover:text-primary'}`}
          >
            Workspace
          </button>
          <button 
            onClick={() => setActiveTab('train')}
            className={`font-label-caps text-xs tracking-wider uppercase font-semibold transition-colors ${activeTab === 'train' ? 'text-primary' : 'text-outline hover:text-primary'}`}
          >
            Train Tokenizer
          </button>
          <button 
            onClick={() => setActiveTab('about')}
            className={`font-label-caps text-xs tracking-wider uppercase font-semibold transition-colors ${activeTab === 'about' ? 'text-primary' : 'text-outline hover:text-primary'}`}
          >
            How it Works
          </button>
        </nav>
        <div className="flex items-center gap-2">
          <button className="material-symbols-outlined text-outline p-2 hover:bg-primary-container/20 rounded-full transition-all active:scale-95" data-icon="account_circle" href="https://debeshee-sen-portfolio.vercel.app/">
            account_circle
          </button>
        </div>
      </header>

      {/* Main Canvas */}
      <main className="flex-grow pt-24 pb-20 px-8 max-w-7xl mx-auto w-full">
        {activeTab === 'workspace' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* Left Column: Model Selection & Configuration */}
            <div className="lg:col-span-4 space-y-6">
              
              {/* Model Configuration Card */}
              <section className="glass-card rounded-xl p-6">
                <h2 className="font-label-caps text-xs tracking-wider uppercase font-semibold text-on-surface-variant mb-4">
                  MODEL CONFIGURATION
                </h2>
                
                <div className="space-y-4">
                  {/* Drag & Drop File Upload */}
                  <div 
                    onClick={handleUploadClick}
                    className="border-2 border-dashed border-outline-variant/50 rounded-lg p-5 flex flex-col items-center justify-center gap-2 hover:bg-surface-container/50 transition-colors cursor-pointer group"
                  >
                    <input 
                      type="file" 
                      ref={fileInputRef} 
                      onChange={handleFileChange} 
                      className="hidden" 
                      accept=".pkl"
                    />
                    <span className="material-symbols-outlined text-outline group-hover:text-primary text-3xl transition-colors" data-icon="upload_file">
                      {isUploading ? 'sync' : 'upload_file'}
                    </span>
                    <p className="font-body-main text-outline text-sm text-center">
                      {isUploading ? 'Uploading & Parsing...' : 'Upload .pkl File'}
                    </p>
                  </div>

                  {uploadError && (
                    <div className="text-error text-xs p-2 rounded bg-error-container/20 border border-error/25">
                      {uploadError}
                    </div>
                  )}

                  {/* Recent Files / Available Models */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <p className="font-label-caps text-[10px] tracking-wider uppercase font-semibold text-outline">
                        AVAILABLE MODELS
                      </p>
                      <button 
                        onClick={() => fetchModels()} 
                        className="material-symbols-outlined text-outline hover:text-primary text-sm p-1 rounded transition-colors"
                        title="Reload model files"
                      >
                        refresh
                      </button>
                    </div>

                    {isModelsLoading ? (
                      <div className="text-xs text-outline py-4 text-center">Loading model files...</div>
                    ) : models.length === 0 ? (
                      <div className="text-xs text-outline py-6 text-center border border-dashed border-outline-variant/30 rounded-lg">
                        No tokenizer models available. Try uploading or training one!
                      </div>
                    ) : (
                      <div className="flex flex-col gap-2 max-h-48 overflow-y-auto pr-1">
                        {models.map((model) => {
                          const isActive = model.name === activeModel
                          const sizeKB = (model.size_bytes / 1024).toFixed(1)
                          return (
                            <button
                              key={model.name}
                              onClick={() => setActiveModel(model.name)}
                              className={`flex justify-between items-center px-4 py-2.5 rounded-lg border text-left text-xs font-medium transition-all ${
                                isActive 
                                  ? 'bg-primary-container/30 border-primary/50 text-primary shadow-sm'
                                  : 'bg-surface-container border-outline-variant/20 text-on-surface-variant hover:bg-surface-container-high'
                              }`}
                            >
                              <span className="truncate pr-2 font-mono">{model.name}</span>
                              <span className="text-[10px] text-outline flex-shrink-0">{sizeKB} KB</span>
                            </button>
                          )
                        })}
                      </div>
                    )}
                  </div>
                </div>
              </section>

              {/* Quick Info Card */}
              <section className="glass-card rounded-xl p-6 bg-gradient-to-br from-primary-container/10 to-transparent">
                <h3 className="text-sm font-semibold text-primary mb-2 flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-sm">info</span>
                  What is BPE?
                </h3>
                <p className="text-xs text-on-surface-variant leading-relaxed">
                  Byte Pair Encoding (BPE) is a subword tokenization algorithm. It dynamically merges frequent byte pairs to construct a vocabulary, striking an optimal balance between character-level and word-level encoding.
                </p>
                <div className="mt-3 pt-3 border-t border-outline-variant/10 flex justify-between items-center text-[11px] text-outline">
                  <span>Current Model:</span>
                  <span className="font-mono font-medium text-primary bg-primary-container/20 px-1.5 py-0.5 rounded truncate max-w-[180px]" title={activeModel}>
                    {activeModel || 'None Selected'}
                  </span>
                </div>
              </section>
            </div>

            {/* Right Column: Input and Results Area */}
            <div className="lg:col-span-8 flex flex-col gap-6">
              
              {/* Text Input Area */}
              <section className="glass-card rounded-xl p-6">
                <label className="font-label-caps text-xs tracking-wider uppercase font-semibold text-on-surface-variant block mb-2">
                  INPUT TEXT
                </label>
                <textarea 
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="w-full bg-transparent border border-outline-variant/30 rounded-lg p-3 focus:outline-none focus:border-primary/50 font-body-main text-on-surface placeholder:text-outline-variant resize-none h-32 text-sm leading-relaxed" 
                  id="inputText" 
                  placeholder="Type or paste text here to see how it is tokenized in real-time..."
                />
                
                <div className="flex justify-between items-center mt-3">
                  <span className="text-xs text-outline font-medium">
                    {inputText.length} characters • {new Blob([inputText]).size} bytes
                  </span>
                  <button 
                    onClick={() => handleTokenize()}
                    disabled={isTokenizing || !activeModel}
                    className={`px-5 py-2.5 rounded-lg font-label-caps text-xs uppercase tracking-wider font-semibold shadow-sm hover:shadow-md active:scale-95 transition-all flex items-center justify-center gap-2 ${
                      !activeModel
                        ? 'bg-surface-container text-outline cursor-not-allowed'
                        : 'bg-primary text-on-primary hover:bg-primary/95'
                    }`}
                    id="visualizeBtn"
                  >
                    <span className="material-symbols-outlined text-sm" data-icon="auto_awesome">
                      {isTokenizing ? 'sync' : 'auto_awesome'}
                    </span>
                    {isTokenizing ? 'Tokenizing...' : 'VISUALISE TOKENS'}
                  </button>
                </div>
              </section>

              {/* Tokenized Output Card */}
              <section className="glass-card rounded-xl p-6 flex-grow flex flex-col min-h-[350px]">
                <div className="flex items-center justify-between mb-4 border-b border-outline-variant/10 pb-4">
                  <h2 className="font-label-caps text-xs tracking-wider uppercase font-semibold text-on-surface-variant">
                    TOKENIZED OUTPUT
                  </h2>
                  
                  {tokenResult && (
                    <div className="flex gap-6">
                      <div className="text-right">
                        <p className="text-[10px] tracking-wider uppercase font-semibold text-outline">TOKENS</p>
                        <p className="font-mono text-primary font-bold text-lg">{tokenResult.token_count}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-[10px] tracking-wider uppercase font-semibold text-outline" title="Space Savings relative to raw UTF-8 bytes">SPACE SAVINGS</p>
                        <p className="font-mono text-tertiary font-bold text-lg">{(tokenResult.efficiency * 100).toFixed(1)}%</p>
                      </div>
                      <div className="text-right">
                        <p className="text-[10px] tracking-wider uppercase font-semibold text-outline" title="Average characters per token">COMPRESSION</p>
                        <p className="font-mono text-secondary font-bold text-lg">{tokenResult.compression_ratio.toFixed(2)}x</p>
                      </div>
                    </div>
                  )}
                </div>

                {tokenizeError && (
                  <div className="text-error text-xs p-3 rounded-lg bg-error-container/20 border border-error/25 mb-4">
                    {tokenizeError}
                  </div>
                )}

                {/* Token Display Chips Grid */}
                <div className="flex flex-wrap gap-2 content-start flex-grow p-1">
                  {!activeModel ? (
                    <div className="m-auto text-center py-10">
                      <span className="material-symbols-outlined text-outline text-4xl block mb-2">construction</span>
                      <p className="text-sm text-outline">Select or upload a tokenizer model to get started</p>
                    </div>
                  ) : !tokenResult && !isTokenizing ? (
                    <div className="m-auto text-center py-10">
                      <span className="material-symbols-outlined text-outline text-4xl block mb-2">chat</span>
                      <p className="text-sm text-outline">Enter some text to see visual tokens</p>
                    </div>
                  ) : isTokenizing && !tokenResult ? (
                    <div className="m-auto text-center py-10">
                      <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                      <p className="text-sm text-outline">Computing subword tokens...</p>
                    </div>
                  ) : (
                    tokenResult?.tokens.map((tok, idx) => {
                      const colorClass = pastelColors[idx % pastelColors.length]
                      
                      // Highlight spaces/newlines with a special visual representation
                      const hasLeadingSpace = tok.text.startsWith(' ')
                      const isWhitespaceOnly = /^\s+$/.test(tok.text)
                      
                      return (
                        <div 
                          key={idx}
                          className="flex flex-col rounded-lg border border-outline-variant/30 overflow-hidden shadow-sm token-chip group relative select-none"
                        >
                          {/* Top Box: Sentence text fragment */}
                          <div className="px-3 py-1.5 bg-white text-zinc-800 font-mono text-xs font-semibold border-b border-outline-variant/20 flex items-center justify-center min-h-[32px] break-all select-all whitespace-pre-wrap">
                            {renderTokenText(tok.text)}
                          </div>
                          
                          {/* Bottom Box: Token ID value */}
                          <div className={`px-3 py-1 font-mono text-[9px] font-bold flex items-center justify-center select-all ${colorClass}`}>
                            {tok.id}
                          </div>
                          
                          {/* Tooltip on hover */}
                          <div className="hidden group-hover:flex flex-col gap-1 absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-inverse-surface text-inverse-on-surface text-[10px] py-1.5 px-3.5 rounded-lg whitespace-nowrap z-30 shadow-xl pointer-events-none transition-all">
                            <div className="font-semibold text-primary-fixed border-b border-outline/30 pb-0.5 mb-0.5">Token ID: {tok.id}</div>
                            <div>Bytes: <span className="font-mono text-[9px] font-bold text-secondary-fixed">{tok.bytes.join(' ')}</span></div>
                            <div className="flex gap-2 justify-between">
                              <span>Raw characters:</span>
                              <span className="font-mono text-tertiary-fixed font-bold">{tok.text.length} chars</span>
                            </div>
                            {/* Decorative pointer arrow */}
                            <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-inverse-surface"></div>
                          </div>
                        </div>
                      )
                    })
                  )}
                </div>
              </section>
            </div>
          </div>
        )}

        {activeTab === 'train' && (
          <div className="max-w-2xl mx-auto">
            <section className="glass-card rounded-xl p-8">
              <div className="flex items-center gap-3 border-b border-outline-variant/10 pb-4 mb-6">
                <span className="material-symbols-outlined text-primary text-3xl">model_training</span>
                <div>
                  <h2 className="text-lg font-bold text-on-surface">Train BPE Tokenizer</h2>
                  <p className="text-xs text-outline">Train a subword tokenizer model from scratch on local datasets</p>
                </div>
              </div>

              {trainingState.status === 'training' ? (
                <div className="py-8 space-y-6 text-center">
                  <div className="relative w-20 h-20 mx-auto">
                    <div className="absolute inset-0 border-4 border-primary-container rounded-full"></div>
                    <div className="absolute inset-0 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
                    <span className="material-symbols-outlined text-primary text-3xl absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                      settings_suggest
                    </span>
                  </div>

                  <div className="space-y-2">
                    <p className="text-sm font-semibold text-primary animate-pulse">{trainingState.message}</p>
                    <p className="text-xs text-outline">Please do not close this window while training is in progress...</p>
                  </div>

                  {trainingState.total > 0 && (
                    <div className="max-w-md mx-auto space-y-1.5">
                      <div className="flex justify-between text-xs text-outline font-semibold">
                        <span>Progress</span>
                        <span>{((trainingState.progress / trainingState.total) * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-surface-container rounded-full h-3.5 p-0.5 overflow-hidden">
                        <div 
                          className="bg-primary h-full rounded-full transition-all duration-300 shadow-sm"
                          style={{ width: `${(trainingState.progress / trainingState.total) * 100}%` }}
                        ></div>
                      </div>
                      <div className="text-[10px] text-outline text-right font-mono">
                        Merge {trainingState.progress} / {trainingState.total}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <form onSubmit={handleTrain} className="space-y-6">
                  {trainingState.status === 'completed' && (
                    <div className="flex items-center gap-3 p-4 rounded-lg bg-tertiary-container/20 border border-tertiary/20 text-tertiary">
                      <span className="material-symbols-outlined text-lg">check_circle</span>
                      <div className="text-xs">
                        <p className="font-semibold">{trainingState.message}</p>
                        <p className="opacity-80">Loading model and switching workspace...</p>
                      </div>
                    </div>
                  )}

                  {trainingState.status === 'error' && (
                    <div className="p-4 rounded-lg bg-error-container/20 border border-error/25 text-error">
                      <div className="flex items-center gap-3 mb-1">
                        <span className="material-symbols-outlined text-lg">error</span>
                        <p className="font-semibold text-xs">Training Failed</p>
                      </div>
                      <p className="text-xs pl-8 font-mono">{trainingState.message}</p>
                    </div>
                  )}

                  {/* Dataset Selector */}
                  <div className="space-y-1.5">
                    <label className="block text-xs font-bold text-on-surface-variant uppercase tracking-wider">
                      Select Training Dataset
                    </label>
                    <select
                      value={selectedDataset}
                      onChange={(e) => setSelectedDataset(e.target.value)}
                      className="w-full bg-surface border border-outline-variant/30 rounded-lg p-2.5 text-sm focus:outline-none focus:border-primary/50"
                      required
                    >
                      {datasets.length === 0 ? (
                        <option value="">No datasets found in datasets/</option>
                      ) : (
                        datasets.map((d) => (
                          <option key={d.name} value={d.name}>
                            {d.name} ({(d.size_bytes / 1024 / 1024).toFixed(2)} MB)
                          </option>
                        ))
                      )}
                    </select>
                    <p className="text-[10px] text-outline">
                      Select a text file from the backend `/datasets` folder.
                    </p>
                  </div>

                  {/* Vocabulary size */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="block text-xs font-bold text-on-surface-variant uppercase tracking-wider">
                        Vocabulary Size
                      </label>
                      <input
                        type="number"
                        min="257"
                        max="30000"
                        value={vocabSize}
                        onChange={(e) => setVocabSize(e.target.value)}
                        className="w-full bg-surface border border-outline-variant/30 rounded-lg p-2.5 text-sm focus:outline-none focus:border-primary/50 font-mono"
                        required
                      />
                      <p className="text-[10px] text-outline">
                        Must be greater than 256. BPE vocabulary will include 256 byte tokens plus custom merges.
                      </p>
                    </div>

                    {/* Model name output */}
                    <div className="space-y-1.5">
                      <label className="block text-xs font-bold text-on-surface-variant uppercase tracking-wider">
                        Output Model Filename
                      </label>
                      <input
                        type="text"
                        value={trainModelName}
                        onChange={(e) => setTrainModelName(e.target.value)}
                        className="w-full bg-surface border border-outline-variant/30 rounded-lg p-2.5 text-sm focus:outline-none focus:border-primary/50 font-mono"
                        placeholder="my_tokenizer.pkl"
                        required
                      />
                      <p className="text-[10px] text-outline">
                        File will be saved in the backend `/models` folder with `.pkl` extension.
                      </p>
                    </div>
                  </div>

                  {/* Warning on large vocab / text size */}
                  <div className="p-3.5 rounded-lg bg-surface-container-high border border-outline-variant/30 flex gap-3 text-xs text-on-surface-variant">
                    <span className="material-symbols-outlined text-outline text-lg">warning</span>
                    <div className="leading-relaxed">
                      <span className="font-semibold text-outline-variant">Note:</span> BPE training scales with the length of the text. Large vocab sizes and dataset sizes might take up to a few minutes to complete on the backend.
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={datasets.length === 0}
                    className="w-full bg-primary text-on-primary py-3 rounded-lg font-label-caps text-xs uppercase tracking-wider font-semibold shadow hover:bg-primary/95 transition-all active:scale-[0.98] disabled:bg-surface-container disabled:text-outline disabled:cursor-not-allowed"
                  >
                    Start Training Tokenizer
                  </button>
                </form>
              )}
            </section>
          </div>
        )}

        {activeTab === 'about' && (
          <div className="max-w-3xl mx-auto space-y-6">
            <section className="glass-card rounded-xl p-8 space-y-4">
              <h2 className="text-xl font-bold text-primary flex items-center gap-2">
                <span className="material-symbols-outlined">school</span>
                Interactive Tokenizer Walkthrough
              </h2>
              
              <div className="space-y-4 text-sm text-on-surface-variant leading-relaxed">
                <p>
                  To process text, Modern LLMs (like GPT-4, Llama, Gemini) don't read words directly. Instead, they use a process called <strong>Tokenization</strong> to break text into integer representations.
                </p>
                
                <h3 className="text-sm font-bold text-on-surface mt-4 uppercase tracking-wider">How Byte Pair Encoding (BPE) Works:</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 py-2">
                  <div className="p-4 rounded-lg bg-surface-container border border-outline-variant/10 text-center">
                    <div className="font-mono text-primary font-bold text-lg mb-1">1. Bytes</div>
                    <p className="text-xs text-outline">
                      The text is encoded as raw UTF-8 bytes (integers 0–255). This ensures any unicode symbol can be processed.
                    </p>
                  </div>
                  <div className="p-4 rounded-lg bg-surface-container border border-outline-variant/10 text-center">
                    <div className="font-mono text-primary font-bold text-lg mb-1">2. Frequencies</div>
                    <p className="text-xs text-outline">
                      The tokenizer counts frequencies of all adjacent pairs (e.g., matching `h` followed by `e` to make `he`).
                    </p>
                  </div>
                  <div className="p-4 rounded-lg bg-surface-container border border-outline-variant/10 text-center">
                    <div className="font-mono text-primary font-bold text-lg mb-1">3. Merges</div>
                    <p className="text-xs text-outline">
                      The most frequent pair is merged into a brand new token ID (starting from 256). This loops until the target vocabulary size is reached.
                    </p>
                  </div>
                </div>

                <p className="mt-2">
                  When encoding a new sentence, the tokenizer reads the bytes and repeats the merges in the exact order they were learned during training.
                </p>
              </div>
            </section>

            <section className="glass-card rounded-xl p-8 space-y-4">
              <h3 className="text-sm font-bold text-primary uppercase tracking-wider">Metrics explained:</h3>
              <div className="space-y-3 text-xs text-on-surface-variant">
                <div>
                  <span className="font-semibold text-on-surface">Tokens Count:</span> The number of integer ID tokens the input text was compressed to. Fewer tokens mean the text takes less computational overhead to process in a Transformer model.
                </div>
                <div>
                  <span className="font-semibold text-on-surface">Space Savings (Efficiency):</span> The percentage reduction in space compared to raw UTF-8 bytes. Calculated as <code>1.0 - (Token Count / Byte Count)</code>. A higher value indicates efficient compression.
                </div>
                <div>
                  <span className="font-semibold text-on-surface">Compression Ratio:</span> The average number of text characters represented by a single token. Computed as <code>Character Count / Token Count</code>.
                </div>
              </div>
            </section>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full mt-auto flex flex-col md:flex-row justify-between items-center gap-4 px-8 py-6 border-t border-outline-variant/10 bg-transparent text-xs text-outline">
        <div className="flex items-center gap-1.5 font-medium">
          <span>DevelopedBy</span>
          <a className="text-primary hover:underline transition-opacity font-bold" href="https://debeshee-sen-portfolio.vercel.app/">
            @DebesheeSen
          </a>
        </div>
        <div className="flex items-center gap-5">
          {/* GitHub */}
          <a aria-label="GitHub" className="hover:text-primary transition-colors cursor-pointer" href="https://github.com/debesheesen" target="_blank" rel="noopener noreferrer">
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
          </a>
          {/* LinkedIn */}
          <a aria-label="LinkedIn" className="hover:text-primary transition-colors cursor-pointer" href="https://linkedin.com/in/debesheesen" target="_blank" rel="noopener noreferrer">
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.779-1.75-1.75s.784-1.75 1.75-1.75 1.75.779 1.75 1.75-.784 1.75-1.75 1.75zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
            </svg>
          </a>
          {/* Email */}
          <a aria-label="Email" className="hover:text-primary transition-colors cursor-pointer" href="mailto:debeshee.sen.ds@gmail.com">
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <path d="M12 12.713l-11.985-9.713h23.97l-11.985 9.713zm0 2.574l12-9.725v15.438h-24v-15.438l12 9.725z"/>
            </svg>
          </a>
          {/* Website (Portfolio) */}
          <a aria-label="Website" className="hover:text-primary transition-colors cursor-pointer" href="https://debeshee-sen-portfolio.vercel.app/" target="_blank" rel="noopener noreferrer">
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <path d="M12 0c-6.627 0-12 5.373-12 12s5.373 12 12 12 12-5.373 12-12-5.373-12-12-12zm-1.001 22.028c-3.161-.439-5.918-2.234-7.409-4.856.883-.518 1.954-.863 3.013-.863.666 0 1.341.141 1.968.441.77.369 1.353.978 1.921 1.637.166.192.336.388.507.575v3.066zm1.001-14c1.103 0 2 .897 2 2s-.897 2-2 2-2-.897-2-2 .897-2 2-2zm8.41 10.164c-.815-1.579-2.316-2.83-4.225-3.411-.318-.097-.549-.379-.571-.711-.059-.893-.362-1.745-.884-2.483.568-.787 1.488-1.282 2.502-1.348 1.696-.11 3.238.749 4.025 2.181.365.663.593 1.391.681 2.138-.344.607-.852 1.157-1.528 1.634zm-8.41-16.192c.677 0 1.332.127 1.942.361.649.248 1.258.629 1.808 1.134.409.375.823.754 1.252 1.127v1.737c0 .553-.447 1-1 1h-8.002c-.553 0-1-.447-1-1v-1.713c.435-.378.854-.763 1.265-1.142.548-.503 1.156-.882 1.803-1.128.604-.23 1.252-.356 1.932-.356zm8.127 6.012c-.546-.995-1.464-1.738-2.544-2.023-.198-.052-.401-.089-.607-.107v-1.077c1.378.694 2.551 1.764 3.393 3.093.076.121.149.244.218.369-.153.242-.307.495-.46.745zm-16.254 0c.069-.125.142-.248.218-.369.842-1.329 2.015-2.399 3.393-3.093v1.077c-.206.018-.409.055-.607.107-1.08.285-1.998 1.028-2.544 2.023-.153-.25-.307-.503-.46-.745z"/>
            </svg>
          </a>
        </div>
      </footer>
    </div>
  )
}

export default App
