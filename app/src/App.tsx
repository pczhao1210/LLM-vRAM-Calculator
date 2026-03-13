import { useEffect, useState } from 'react'
import {
  doesGpuSupportQuantization,
  defaultCustomModel,
  getGpuById,
  getModelById,
  getGpuQuantizationSupportLevel,
  getQuantizationById,
  gpuCatalog,
  modelCatalog,
  quantizationCatalog,
  resolveCompatibleQuantizationId,
} from './data/catalog'
import { detectBrowserLocale, messages } from './i18n'
import {
  estimateDeploymentMemory,
  recommendConfigurations,
  recommendModelsForGpu,
  simulateConfiguration,
} from './lib/calculator'
import type {
  CalculatorDirection,
  CloudGpuInstance,
  CloudProvider,
  ConfigSimulation,
  DeploymentMode,
  EstimateProfile,
  GpuProfile,
  ModelPreset,
  QuantizationSupportLevel,
} from './types'

const contextPresetOptions = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
const defaultModelSelectionId = 'qwen3.5-27b'
const defaultModelSelection = getModelById(defaultModelSelectionId)

function formatGb(value: number) {
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} GB`
}

function formatCompact(value: number) {
  if (value >= 1024 * 1024) {
    return `${Math.round(value / (1024 * 1024))}M`
  }

  if (value >= 1024) {
    return `${Math.round(value / 1024)}K`
  }

  return value.toString()
}

function formatParams(value: number) {
  if (value >= 100) {
    return `${value.toFixed(0)}B`
  }

  return `${value.toFixed(1)}B`
}

function resolveContextPreset(value: number) {
  return contextPresetOptions.includes(value) ? String(value) : 'custom'
}

function getSupportedContextPresetOptions(maxContext: number) {
  return contextPresetOptions.filter((value) => value <= maxContext)
}

function formatDeploymentTier(
  tier: 'workstation' | 'datacenter' | 'platform',
  labels: Record<'workstation' | 'datacenter' | 'platform', string>,
) {
  return labels[tier]
}

function formatRecommendationGrade(
  item: ConfigSimulation,
  labels: Record<'recommended' | 'tight' | 'testOnly', string>,
) {
  return labels[item.recommendationGrade]
}

function formatQuantizationSupportLevel(
  level: QuantizationSupportLevel,
  labels: Record<'supported' | 'conditional' | 'notRecommended', string>,
) {
  return labels[level]
}

function formatQuantizationOptionLabel(
  label: string,
  level: QuantizationSupportLevel,
  supportLabels: Record<'supported' | 'conditional' | 'notRecommended', string>,
) {
  return `${label} · ${formatQuantizationSupportLevel(level, supportLabels)}`
}

function formatUnitCount(units: number) {
  return `${units}x`
}

function formatCloudProvider(provider: CloudProvider, locale: 'zh' | 'en') {
  switch (provider) {
    case 'aws':
      return 'AWS'
    case 'azure':
      return 'Azure'
    case 'gcp':
      return locale === 'zh' ? 'GCP' : 'GCP'
    case 'oracle':
      return locale === 'zh' ? 'Oracle Cloud' : 'Oracle Cloud'
    default:
      return provider
  }
}

function formatCloudInstance(item: CloudGpuInstance, locale: 'zh' | 'en') {
  const suffix = locale === 'zh' ? `${item.memoryPerGpuGB} GB / GPU` : `${item.memoryPerGpuGB} GB per GPU`
  return `${formatCloudProvider(item.provider, locale)} · ${item.instanceType} · ${suffix}${item.notes ? ` · ${item.notes}` : ''}`
}

function renderCloudInstances(gpu: GpuProfile, locale: 'zh' | 'en') {
  if (!gpu.cloudInstances?.length) {
    return null
  }

  return (
    <div className="notes-stack">
      {gpu.cloudInstances.map((item) => (
        <p key={`${gpu.id}-${item.provider}-${item.instanceType}`} className="micro-note">
          {formatCloudInstance(item, locale)}
        </p>
      ))}
    </div>
  )
}

function compareModelName(left: ModelPreset, right: ModelPreset) {
  return left.name.localeCompare(right.name, undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

function buildModelGroups(searchTerm: string) {
  const normalized = searchTerm.trim().toLowerCase()
  const visibleModels = modelCatalog
    .filter((model) => {
      if (!normalized) {
        return true
      }

      return `${model.family} ${model.name}`.toLowerCase().includes(normalized)
    })
    .sort(compareModelName)

  const groups = visibleModels.reduce<Record<string, ModelPreset[]>>((result, model) => {
    if (!result[model.family]) {
      result[model.family] = []
    }

    result[model.family].push(model)
    return result
  }, {})

  return Object.fromEntries(
    Object.entries(groups).sort(([leftFamily], [rightFamily]) =>
      leftFamily.localeCompare(rightFamily, undefined, { numeric: true, sensitivity: 'base' }),
    ),
  )
}

function App() {
  const [copySuccess, setCopySuccess] = useState(false)
  const searchParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : new URLSearchParams()

  const [locale, setLocale] = useState<'zh' | 'en'>((searchParams.get('locale') as 'zh' | 'en') || detectBrowserLocale())
  const [direction, setDirection] = useState<CalculatorDirection>((searchParams.get('direction') as CalculatorDirection) || 'model-to-gpu')
  const [useCustomModel, setUseCustomModel] = useState(searchParams.get('custom') === 'true')
  const [selectedModelId, setSelectedModelId] = useState(searchParams.get('modelId') || defaultModelSelectionId)
  const [modelSearch, setModelSearch] = useState('')
  const [customModel, setCustomModel] = useState<ModelPreset>(defaultCustomModel)

  const initialModel = getModelById(searchParams.get('modelId') || defaultModelSelectionId) || defaultModelSelection
  const [quantizationId, setQuantizationId] = useState(searchParams.get('quantId') || initialModel.defaultQuantizationId)
  const [contextTokens, setContextTokens] = useState(Number(searchParams.get('ctx')) || initialModel.defaultContext)
  const [contextPreset, setContextPreset] = useState(resolveContextPreset(Number(searchParams.get('ctx')) || initialModel.defaultContext))
  const [concurrency, setConcurrency] = useState(Number(searchParams.get('conc')) || 4)
  const [deploymentMode, setDeploymentMode] = useState<DeploymentMode>((searchParams.get('mode') as DeploymentMode) || 'inference')
  const [estimateProfile, setEstimateProfile] = useState<EstimateProfile>((searchParams.get('profile') as EstimateProfile) || 'balanced')
  const [trainSequenceLength, setTrainSequenceLength] = useState(Number(searchParams.get('trainSeq')) || 2048)
  const [microBatchSize, setMicroBatchSize] = useState(Number(searchParams.get('mbs')) || 2)
  const [loraRank, setLoraRank] = useState(Number(searchParams.get('rank')) || 16)
  const [checkpointing, setCheckpointing] = useState(searchParams.get('ckpt') !== 'false')
  const [selectedGpuId, setSelectedGpuId] = useState(searchParams.get('gpuId') || 'h100-80gb')
  const [gpuUnits, setGpuUnits] = useState(Number(searchParams.get('gpuUnits')) || 2)
  const [recommendationTierFilter, setRecommendationTierFilter] = useState<'all' | 'workstation' | 'datacenter'>('all')
  const [recommendationUnitFilter, setRecommendationUnitFilter] = useState('all')
  const [recommendationGpuSearch, setRecommendationGpuSearch] = useState('')
  const [openRecommendationGroups, setOpenRecommendationGroups] = useState<Record<string, boolean>>({})
  const [openRecommendationRows, setOpenRecommendationRows] = useState<Record<string, boolean>>({})

  const text = messages[locale]
  const activeModel = useCustomModel ? customModel : getModelById(selectedModelId)
  const quantization = getQuantizationById(quantizationId)
  const selectedGpu = getGpuById(selectedGpuId)
  const selectedUnits = selectedGpu.category === 'platform' ? 1 : gpuUnits
  const modelGroups = buildModelGroups(modelSearch)
  const visibleModelCount = Object.values(modelGroups).reduce((count, models) => count + models.length, 0)
  const supportedContextPresetOptions = getSupportedContextPresetOptions(activeModel.maxContext)
  const selectedGpuQuantizationEntries = quantizationCatalog.map((item) => ({
    ...item,
    supportLevel: getGpuQuantizationSupportLevel(selectedGpu, item.id),
  }))
  const supportedGpuQuantizationLabels = selectedGpuQuantizationEntries
    .filter((item) => item.supportLevel === 'supported')
    .map((item) => item.shortLabel)
    .join(' / ')
  const conditionalGpuQuantizationLabels = selectedGpuQuantizationEntries
    .filter((item) => item.supportLevel === 'conditional')
    .map((item) => item.shortLabel)
    .join(' / ')
  const notRecommendedGpuQuantizationLabels = selectedGpuQuantizationEntries
    .filter((item) => item.supportLevel === 'notRecommended')
    .map((item) => item.shortLabel)
    .join(' / ')
  const quantizationRequiresSpecialGpuSupport = (quantization.requiredGpuFeatures?.length ?? 0) > 0
  const quantizationDescription =
    text.quantizationDescriptions[quantization.id as keyof typeof text.quantizationDescriptions] ?? quantization.description
  const currentSelectedGpuQuantizationSupport = getGpuQuantizationSupportLevel(selectedGpu, quantizationId)

  const sizingInput = {
    contextTokens,
    concurrency,
    deploymentMode,
    estimateProfile,
    trainSequenceLength,
    microBatchSize,
    loraRank,
    checkpointing,
  }

  const breakdown = estimateDeploymentMemory(activeModel, quantizationId, sizingInput, locale)
  const recommendations = recommendConfigurations(breakdown, sizingInput, locale)
  const removedRecommendationGpuNames = recommendations
    .filter((item) => item.gpu.category !== 'platform' && !doesGpuSupportQuantization(item.gpu, quantizationId))
    .map((item) => item.gpu.name)
    .filter((name, index, list) => list.indexOf(name) === index)
    .join(' / ')
  const supportLevelRank: Record<QuantizationSupportLevel, number> = {
    supported: 0,
    conditional: 1,
    notRecommended: 2,
  }
  const conventionalRecommendations = recommendations
    .filter((item) => item.gpu.category !== 'platform' && doesGpuSupportQuantization(item.gpu, quantizationId))
    .sort(
      (left, right) =>
        supportLevelRank[getGpuQuantizationSupportLevel(left.gpu, quantizationId)] -
        supportLevelRank[getGpuQuantizationSupportLevel(right.gpu, quantizationId)],
    )
  const selectedGpuSimulation = simulateConfiguration(breakdown, selectedGpu, selectedUnits, sizingInput, locale)
  const reverseResults = recommendModelsForGpu(selectedGpu, selectedUnits, quantizationId, sizingInput, locale)
  const bestReverseResults = reverseResults.slice(0, 8)
  const filteredRecommendations = conventionalRecommendations.filter((item) => {
    if (recommendationTierFilter !== 'all' && item.gpu.deploymentTier !== recommendationTierFilter) {
      return false
    }

    if (recommendationUnitFilter !== 'all' && String(item.units) !== recommendationUnitFilter) {
      return false
    }

    if (
      recommendationGpuSearch.trim() &&
      !item.gpu.name.toLowerCase().includes(recommendationGpuSearch.trim().toLowerCase())
    ) {
      return false
    }

    return true
  })
  const recommendationGroups = filteredRecommendations.reduce<Record<string, ConfigSimulation[]>>((groups, item) => {
    const key = String(item.units)
    if (!groups[key]) {
      groups[key] = []
    }

    groups[key].push(item)
    return groups
  }, {})
  const groupedRecommendationEntries = Object.entries(recommendationGroups).sort(
    ([leftUnits], [rightUnits]) => Number(leftUnits) - Number(rightUnits),
  )

  function toggleRecommendationGroup(groupKey: string) {
    setOpenRecommendationGroups((current) => ({
      ...current,
      [groupKey]: !current[groupKey],
    }))
  }

  function toggleRecommendationRow(rowKey: string) {
    setOpenRecommendationRows((current) => ({
      ...current,
      [rowKey]: !current[rowKey],
    }))
  }

  function handleCopyReport() {
    const report = `
**[${locale === 'zh' ? 'LLM 显存估算报告 / LLM VRAM Estimation Report' : 'LLM VRAM Estimation Report'}]**
- **${locale === 'zh' ? '模型 / Model' : 'Model'}**: ${activeModel.name} (${formatParams(activeModel.parameterCountB)} / ${activeModel.layers} layers)
- **${locale === 'zh' ? '量化 / Quantization' : 'Quantization'}**: ${quantization.label}
- **${locale === 'zh' ? '上下文 / Context Length' : 'Context Length'}**: ${formatCompact(contextTokens)}
- **${locale === 'zh' ? '并发数 / Concurrency' : 'Concurrency'}**: ${concurrency}
- **${locale === 'zh' ? '预估总显存 / Estimated Total VRAM' : 'Estimated Total VRAM'}**: ${formatGb(breakdown.totalMemoryGB)}
  - ${locale === 'zh' ? '模型权重 / Weights' : 'Weights'}: ${formatGb(breakdown.weightMemoryGB)}
  - ${locale === 'zh' ? 'KV缓存 / KV Cache' : 'KV Cache'}: ${formatGb(breakdown.kvCacheMemoryGB)}
  - ${locale === 'zh' ? '激活与运行时 / Runtime' : 'Runtime'}: ${formatGb(breakdown.runtimeMemoryGB)}
  - ${locale === 'zh' ? '训练开销 / Tuning Extra' : 'Tuning Extra'}: ${formatGb(breakdown.tuningMemoryGB)}
- **${locale === 'zh' ? '链接 / Link' : 'Link'}**: ${window.location.href}
`.trim()

    navigator.clipboard.writeText(report).then(() => {
      setCopySuccess(true)
      setTimeout(() => setCopySuccess(false), 2000)
    })
  }

  useEffect(() => {
    const params = new URLSearchParams()
    if (direction !== 'model-to-gpu') params.set('direction', direction)
    if (useCustomModel) params.set('custom', 'true')
    if (selectedModelId !== defaultModelSelectionId) params.set('modelId', selectedModelId)
    params.set('quantId', quantizationId)
    params.set('ctx', contextTokens.toString())
    if (concurrency !== 4) params.set('conc', concurrency.toString())
    if (deploymentMode !== 'inference') params.set('mode', deploymentMode)
    if (estimateProfile !== 'balanced') params.set('profile', estimateProfile)
    if (locale !== 'zh') params.set('locale', locale)
    if (selectedGpuId !== 'h100-80gb') params.set('gpuId', selectedGpuId)
    if (gpuUnits !== 2) params.set('gpuUnits', gpuUnits.toString())

    const newUrl = `${window.location.pathname}${params.toString() ? '?' + params.toString() : ''}`
    window.history.replaceState({}, '', newUrl)
  }, [
    direction, useCustomModel, selectedModelId, quantizationId, contextTokens,
    concurrency, deploymentMode, estimateProfile, locale, selectedGpuId, gpuUnits
  ])

  useEffect(() => {
    document.title = text.pageTitle
    document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en'
  }, [locale, text.pageTitle])

  useEffect(() => {
    if (direction !== 'model-to-gpu') {
      return
    }

    if (contextTokens <= activeModel.maxContext) {
      return
    }

    const clampedContext = activeModel.maxContext
    setContextTokens(clampedContext)
    setContextPreset(resolveContextPreset(clampedContext))
  }, [activeModel.maxContext, contextTokens, direction])

  useEffect(() => {
    if (direction !== 'gpu-to-model') {
      return
    }

    const nextQuantizationId = resolveCompatibleQuantizationId(selectedGpu, [
      quantizationId,
      'int4-awq',
      'int8',
      'bf16',
      'gguf-q8',
      'gguf-q6',
      'gguf-q4km',
    ])

    if (nextQuantizationId !== quantizationId) {
      setQuantizationId(nextQuantizationId)
    }
  }, [direction, quantizationId, selectedGpu])

  return (
    <div className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <header className="hero panel">
        <div className="hero-copy">
          <p className="eyebrow">{text.eyebrow}</p>
          <h1>{text.heroTitle}</h1>
          <p className="hero-text">{text.heroDescription}</p>

          <div className="hero-metrics">
            <div>
              <span>{text.labels.modelProfiles}</span>
              <strong>{modelCatalog.length}</strong>
            </div>
            <div>
              <span>{text.labels.gpuProfiles}</span>
              <strong>{gpuCatalog.length}</strong>
            </div>
            <div>
              <span>{text.labels.estimateProfile}</span>
              <strong>{text.profileLabels[estimateProfile]}</strong>
            </div>
          </div>
        </div>

        <div className="hero-aside">
          <label className="language-switch">
            <span>{text.language}</span>
            <select value={locale} onChange={(event) => setLocale(event.target.value as 'zh' | 'en')}>
              <option value="zh">{text.languageOptions.zh}</option>
              <option value="en">{text.languageOptions.en}</option>
            </select>
          </label>

          <div className="stat-card accent-card">
            <span>{text.labels.currentWorkload}</span>
            <strong>{text.deploymentLabels[deploymentMode]}</strong>
            <p>{text.currentWorkloadSummary(formatCompact(contextTokens), concurrency, quantization.shortLabel)}</p>
          </div>
          <div className="stat-card">
            <span>{text.labels.estimatedTotalVram}</span>
            <strong>{formatGb(breakdown.totalMemoryGB)}</strong>
            <p>{text.currentWorkloadDescription}</p>
          </div>
          
          <button 
            type="button"
            onClick={handleCopyReport} 
            style={{ 
              width: '100%', 
              marginTop: '16px', 
              padding: '12px', 
              background: copySuccess ? 'var(--green-500, #10b981)' : 'var(--accent-color, #3b82f6)', 
              color: '#fff', 
              border: 'none', 
              borderRadius: '8px', 
              cursor: 'pointer', 
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              transition: 'background-color 0.2s'
            }}
          >
            {copySuccess ? (locale === 'zh' ? '✅ 已复制链接与报告' : '✅ Copied URL & Report') : (locale === 'zh' ? '📋 分享 / 复制报告' : '📋 Auto-Share & Copy Report')}
          </button>
        </div>
      </header>

      <main className="workspace-grid">
        <section className="input-column">
          <div className="panel section-panel">
            <div className="section-head">
              <h2>{text.labels.modeSwitch}</h2>
              <p>
                {locale === 'zh'
                  ? '同一套估算引擎，同时服务正向部署与反向选型。'
                  : 'One estimation engine for both forward deployment sizing and reverse hardware planning.'}
              </p>
            </div>
            <div className="segmented">
              <button
                className={direction === 'model-to-gpu' ? 'active' : ''}
                onClick={() => setDirection('model-to-gpu')}
                type="button"
              >
                {text.labels.modelToGpu}
              </button>
              <button
                className={direction === 'gpu-to-model' ? 'active' : ''}
                onClick={() => setDirection('gpu-to-model')}
                type="button"
              >
                {text.labels.gpuToModel}
              </button>
            </div>
          </div>

          {direction === 'model-to-gpu' ? (
            <>
              <div className="panel section-panel">
                <div className="section-head">
                  <h2>{text.labels.modelProfile}</h2>
                  <p>{text.labels.modelProfileDesc}</p>
                </div>

                <label className="checkbox-field inline-toggle">
                  <span>{text.labels.useCustomModel}</span>
                  <input
                    type="checkbox"
                    checked={useCustomModel}
                    onChange={(event) => setUseCustomModel(event.target.checked)}
                  />
                </label>

                {!useCustomModel ? (
                  <>
                    <label>
                      {text.labels.searchModelProfile}
                      <input
                        type="text"
                        placeholder={text.labels.searchModelPlaceholder}
                        value={modelSearch}
                        onChange={(event) => setModelSearch(event.target.value)}
                      />
                    </label>

                    <label>
                      {text.labels.selectBuiltInProfile}
                      <select
                        value={selectedModelId}
                        onChange={(event) => {
                          const nextModel = getModelById(event.target.value)
                          setSelectedModelId(event.target.value)
                          setContextTokens(nextModel.defaultContext)
                          setContextPreset(resolveContextPreset(nextModel.defaultContext))
                          setQuantizationId(nextModel.defaultQuantizationId)
                        }}
                      >
                        {Object.entries(modelGroups).map(([family, models]) => (
                          <optgroup key={family} label={`${family} (${models.length})`}>
                            {models.map((model) => (
                              <option key={model.id} value={model.id}>
                                {model.name}
                              </option>
                            ))}
                          </optgroup>
                        ))}
                      </select>
                    </label>

                    <div className="micro-note">{text.visibleProfiles(visibleModelCount)}</div>
                  </>
                ) : (
                  <div className="sub-panel">
                    <div className="field-grid two-up">
                      <label>
                        {locale === 'zh' ? '模型名称' : 'Model Name'}
                        <input
                          type="text"
                          value={customModel.name}
                          onChange={(event) =>
                            setCustomModel((current) => ({ ...current, name: event.target.value || 'Custom Model' }))
                          }
                        />
                      </label>
                      <label>
                        {locale === 'zh' ? '参数量 B' : 'Parameter Count B'}
                        <input
                          type="number"
                          min={1}
                          step={0.5}
                          value={customModel.parameterCountB || ''}
                          onChange={(event) =>
                            setCustomModel((current) => ({
                              ...current,
                              parameterCountB: event.target.value === '' ? 0 : Number(event.target.value),
                            }))
                          }
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 1) {
                              setCustomModel((current) => ({ ...current, parameterCountB: 1 }))
                            }
                          }}
                        />
                      </label>
                    </div>

                    <div className="field-grid three-up">
                      <label>
                        {locale === 'zh' ? '层数' : 'Layers'}
                        <input
                          type="number"
                          min={1}
                          value={customModel.layers || ''}
                          onChange={(event) =>
                            setCustomModel((current) => ({
                              ...current,
                              layers: event.target.value === '' ? 0 : Number(event.target.value),
                            }))
                          }
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 1) {
                              setCustomModel((current) => ({ ...current, layers: 1 }))
                            }
                          }}
                        />
                      </label>
                      <label>
                        Hidden size
                        <input
                          type="number"
                          min={128}
                          step={128}
                          value={customModel.hiddenSize || ''}
                          onChange={(event) =>
                            setCustomModel((current) => ({
                              ...current,
                              hiddenSize: event.target.value === '' ? 0 : Number(event.target.value),
                            }))
                          }
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 128) {
                              setCustomModel((current) => ({ ...current, hiddenSize: 128 }))
                            }
                          }}
                        />
                      </label>
                      <label>
                        KV heads
                        <input
                          type="number"
                          min={1}
                          value={customModel.kvHeads || ''}
                          onChange={(event) =>
                            setCustomModel((current) => ({
                              ...current,
                              kvHeads: event.target.value === '' ? 0 : Number(event.target.value),
                            }))
                          }
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 1) {
                              setCustomModel((current) => ({ ...current, kvHeads: 1 }))
                            }
                          }}
                        />
                      </label>
                    </div>

                    <div className="field-grid two-up">
                      <label>
                        Head dim
                        <input
                          type="number"
                          min={32}
                          step={32}
                          value={customModel.headDim || ''}
                          onChange={(event) =>
                            setCustomModel((current) => ({
                              ...current,
                              headDim: event.target.value === '' ? 0 : Number(event.target.value),
                            }))
                          }
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 32) {
                              setCustomModel((current) => ({ ...current, headDim: 32 }))
                            }
                          }}
                        />
                      </label>
                      <label>
                        Max context
                        <input
                          type="number"
                          min={1024}
                          step={1024}
                          value={customModel.maxContext || ''}
                          onChange={(event) =>
                            setCustomModel((current) => ({
                              ...current,
                              maxContext: event.target.value === '' ? 0 : Number(event.target.value),
                            }))
                          }
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 1024) {
                              setCustomModel((current) => ({ ...current, maxContext: 1024 }))
                            }
                          }}
                        />
                      </label>
                    </div>
                  </div>
                )}

                <div className="catalog-card">
                  <strong>{activeModel.name}</strong>
                  <span>
                    {formatParams(activeModel.parameterCountB)} / {activeModel.layers} layers / {activeModel.hiddenSize} hidden
                  </span>
                  <p>{activeModel.notes}</p>
                </div>
              </div>

              <div className="panel section-panel">
                <div className="section-head">
                  <h2>{text.labels.deploymentParams}</h2>
                  <p>{text.labels.deploymentParamsModelDesc}</p>
                </div>

                <label>
                  {text.labels.quantizationConfig}
                  <select value={quantizationId} onChange={(event) => setQuantizationId(event.target.value)}>
                    {quantizationCatalog.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>

                {quantizationRequiresSpecialGpuSupport ? (
                  <>
                    <div className="micro-note">{text.quantizationCompatibilityForRecommendations(quantization.shortLabel)}</div>

                    {removedRecommendationGpuNames ? (
                      <div className="compatibility-alert compatibility-alert-warning">
                        <strong>{locale === 'zh' ? '兼容性提示' : 'Compatibility Notice'}</strong>
                        <p>{text.removedGpuByQuantization(quantization.shortLabel, removedRecommendationGpuNames)}</p>
                      </div>
                    ) : null}
                  </>
                ) : null}

                <div className="field-grid">
                  <label>
                    <div style={{ marginBottom: '8px' }}>{text.labels.contextPreset}</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {supportedContextPresetOptions.map((value) => (
                        <button
                          key={value}
                          type="button"
                          className={contextPreset === String(value) ? 'active' : ''}
                          onClick={() => {
                            setContextPreset(String(value))
                            setContextTokens(value)
                          }}
                          style={{
                            padding: '6px 12px',
                            borderRadius: '16px',
                            border: '1px solid var(--border-light, #e5e7eb)',
                            background: contextPreset === String(value) ? 'var(--blue-500, #3b82f6)' : 'var(--bg-panel, #ffffff)',
                            color: contextPreset === String(value) ? '#fff' : 'var(--text-main, #111827)',
                            cursor: 'pointer',
                            fontSize: '13px'
                          }}
                        >
                          {formatCompact(value)}
                        </button>
                      ))}
                      <button
                        type="button"
                        className={contextPreset === 'custom' ? 'active' : ''}
                        onClick={() => setContextPreset('custom')}
                        style={{
                          padding: '6px 12px',
                          borderRadius: '16px',
                          border: '1px solid var(--border-light, #e5e7eb)',
                          background: contextPreset === 'custom' ? 'var(--blue-500, #3b82f6)' : 'var(--bg-panel, #ffffff)',
                          color: contextPreset === 'custom' ? '#fff' : 'var(--text-main, #111827)',
                          cursor: 'pointer',
                          fontSize: '13px'
                        }}
                      >
                        {text.labels.contextCustomOption}
                      </button>
                    </div>
                  </label>
                  
                  {contextPreset === 'custom' && (
                    <label style={{ marginTop: '16px', display: 'block' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <span>{text.labels.customContextTokens}</span>
                        <strong>{contextTokens}</strong>
                      </div>
                      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                        <input
                          type="range"
                          min={1024}
                          max={activeModel.maxContext}
                          step={1024}
                          value={contextTokens || 1024}
                          onChange={(event) =>
                            setContextTokens(Math.min(Number(event.target.value), activeModel.maxContext))
                          }
                          style={{ flex: 1 }}
                        />
                        <input
                          type="number"
                          min={1024}
                          max={activeModel.maxContext}
                          step={1024}
                          value={contextTokens || ''}
                          onChange={(event) =>
                            setContextTokens(event.target.value === '' ? 0 : Math.min(Number(event.target.value), activeModel.maxContext))
                          }
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 1024) setContextTokens(1024)
                          }}
                          style={{ width: '100px' }}
                        />
                      </div>
                    </label>
                  )}
                </div>

                <div className="field-grid">
                  <label>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span>{text.labels.concurrency}</span>
                      <strong>{concurrency || 1}</strong>
                    </div>
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                      <input
                        type="range"
                        min={1}
                        max={128}
                        value={concurrency || 1}
                        onChange={(event) => setConcurrency(Number(event.target.value))}
                        style={{ flex: 1 }}
                      />
                      <input
                        type="number"
                        min={1}
                        max={1024}
                        value={concurrency || ''}
                        onChange={(event) => setConcurrency(event.target.value === '' ? 0 : Number(event.target.value))}
                        onBlur={(event) => {
                          if (!event.target.value || Number(event.target.value) < 1) setConcurrency(1)
                        }}
                        style={{ width: '80px' }}
                      />
                    </div>
                  </label>
                </div>

                <div className="option-strip">
                  {(['aggressive', 'balanced', 'conservative'] as EstimateProfile[]).map((item) => (
                    <button
                      key={item}
                      className={estimateProfile === item ? 'active' : ''}
                      onClick={() => setEstimateProfile(item)}
                      type="button"
                    >
                      {text.profileLabels[item]}
                    </button>
                  ))}
                </div>

                <div className="option-strip">
                  {(['inference', 'lora', 'qlora'] as DeploymentMode[]).map((item) => (
                    <button
                      key={item}
                      className={deploymentMode === item ? 'active' : ''}
                      onClick={() => setDeploymentMode(item)}
                      type="button"
                    >
                      {text.deploymentLabels[item]}
                    </button>
                  ))}
                </div>

                {deploymentMode !== 'inference' ? (
                  <div className="sub-panel">
                    <div className="field-grid two-up">
                      <label>
                        {text.labels.trainSequenceLength}
                        <input
                          type="number"
                          min={512}
                          step={512}
                          value={trainSequenceLength || ''}
                          onChange={(event) => setTrainSequenceLength(event.target.value === '' ? 0 : Number(event.target.value))}
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 512) setTrainSequenceLength(512)
                          }}
                        />
                      </label>
                      <label>
                        {text.labels.microBatchSize}
                        <input
                          type="number"
                          min={1}
                          max={64}
                          value={microBatchSize || ''}
                          onChange={(event) => setMicroBatchSize(event.target.value === '' ? 0 : Number(event.target.value))}
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 1) setMicroBatchSize(1)
                          }}
                        />
                      </label>
                    </div>

                    <div className="field-grid two-up">
                      <label>
                        {text.labels.loraRank}
                        <input
                          type="number"
                          min={4}
                          step={4}
                          max={256}
                          value={loraRank || ''}
                          onChange={(event) => setLoraRank(event.target.value === '' ? 0 : Number(event.target.value))}
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 4) setLoraRank(4)
                          }}
                        />
                      </label>
                      <label className="checkbox-field">
                        <span>{text.labels.gradientCheckpointing}</span>
                        <input
                          type="checkbox"
                          checked={checkpointing}
                          onChange={(event) => setCheckpointing(event.target.checked)}
                        />
                      </label>
                    </div>
                  </div>
                ) : null}

                <div className="micro-note">{quantizationDescription}</div>
              </div>
            </>
          ) : (
            <>
              <div className="panel section-panel">
                <div className="section-head">
                  <h2>{text.labels.gpuConfig}</h2>
                  <p>{text.labels.gpuConfigDesc}</p>
                </div>

                <label>
                  {text.labels.gpuPlatform}
                  <select value={selectedGpuId} onChange={(event) => setSelectedGpuId(event.target.value)}>
                    {gpuCatalog.map((gpu) => (
                      <option key={gpu.id} value={gpu.id}>
                        {gpu.name}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  {text.labels.quantity}
                  <select
                    value={selectedUnits}
                    onChange={(event) => setGpuUnits(Number(event.target.value) || 1)}
                    disabled={selectedGpu.category === 'platform'}
                  >
                    {Array.from({ length: selectedGpu.maxUnits }, (_, index) => index + 1).map((unit) => (
                      <option key={unit} value={unit}>
                        {formatUnitCount(unit)}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="catalog-card">
                  <strong>{selectedGpu.name}</strong>
                  <span>
                    {selectedGpu.category === 'platform'
                      ? text.labels.platformShape
                      : `${selectedGpu.memoryPerUnitGB} GB / ${locale === 'zh' ? '单元' : 'unit'}`}
                  </span>
                  <p>{selectedGpu.summary}</p>
                  <p className="micro-note">{selectedGpu.notes}</p>
                  {renderCloudInstances(selectedGpu, locale)}
                </div>
              </div>

              <div className="panel section-panel">
                <div className="section-head">
                  <h2>{text.labels.deploymentParams}</h2>
                  <p>{text.labels.deploymentParamsGpuDesc}</p>
                </div>

                <label>
                  {text.labels.quantizationConfig}
                  <select value={quantizationId} onChange={(event) => setQuantizationId(event.target.value)}>
                    {selectedGpuQuantizationEntries.map((item) => (
                      <option key={item.id} value={item.id} disabled={item.supportLevel === 'notRecommended'}>
                        {formatQuantizationOptionLabel(item.label, item.supportLevel, text.quantizationSupportLevels)}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="micro-note">
                  {text.quantizationFilteredByGpu(
                    selectedGpu.name,
                    supportedGpuQuantizationLabels,
                    conditionalGpuQuantizationLabels,
                    notRecommendedGpuQuantizationLabels,
                  )}
                </div>

                <div className="micro-note">
                  {text.quantizationSupportSummary(
                    quantization.shortLabel,
                    formatQuantizationSupportLevel(currentSelectedGpuQuantizationSupport, text.quantizationSupportLevels),
                  )}
                </div>

                <div className="field-grid">
                  <label>
                    <div style={{ marginBottom: '8px' }}>{text.labels.contextPreset}</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {contextPresetOptions.map((value) => (
                        <button
                          key={value}
                          type="button"
                          className={contextPreset === String(value) ? 'active' : ''}
                          onClick={() => {
                            setContextPreset(String(value))
                            setContextTokens(value)
                          }}
                          style={{
                            padding: '6px 12px',
                            borderRadius: '16px',
                            border: '1px solid var(--border-light, #e5e7eb)',
                            background: contextPreset === String(value) ? 'var(--blue-500, #3b82f6)' : 'var(--bg-panel, #ffffff)',
                            color: contextPreset === String(value) ? '#fff' : 'var(--text-main, #111827)',
                            cursor: 'pointer',
                            fontSize: '13px'
                          }}
                        >
                          {formatCompact(value)}
                        </button>
                      ))}
                      <button
                        type="button"
                        className={contextPreset === 'custom' ? 'active' : ''}
                        onClick={() => setContextPreset('custom')}
                        style={{
                          padding: '6px 12px',
                          borderRadius: '16px',
                          border: '1px solid var(--border-light, #e5e7eb)',
                          background: contextPreset === 'custom' ? 'var(--blue-500, #3b82f6)' : 'var(--bg-panel, #ffffff)',
                          color: contextPreset === 'custom' ? '#fff' : 'var(--text-main, #111827)',
                          cursor: 'pointer',
                          fontSize: '13px'
                        }}
                      >
                        {text.labels.contextCustomOption}
                      </button>
                    </div>
                  </label>

                  {contextPreset === 'custom' && (
                    <label style={{ marginTop: '16px', display: 'block' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <span>{text.labels.customContextTokens}</span>
                        <strong>{contextTokens}</strong>
                      </div>
                      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                        <input
                          type="range"
                          min={1024}
                          max={1048576}
                          step={1024}
                          value={contextTokens || 1024}
                          onChange={(event) => setContextTokens(Number(event.target.value))}
                          style={{ flex: 1 }}
                        />
                        <input
                          type="number"
                          min={1024}
                          step={1024}
                          value={contextTokens || ''}
                          onChange={(event) => setContextTokens(event.target.value === '' ? 0 : Number(event.target.value))}
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 1024) setContextTokens(1024)
                          }}
                          style={{ width: '100px' }}
                        />
                      </div>
                    </label>
                  )}
                </div>

                <div className="field-grid">
                  <label>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span>{text.labels.concurrency}</span>
                      <strong>{concurrency || 1}</strong>
                    </div>
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                      <input
                        type="range"
                        min={1}
                        max={128}
                        value={concurrency || 1}
                        onChange={(event) => setConcurrency(Number(event.target.value))}
                        style={{ flex: 1 }}
                      />
                      <input
                        type="number"
                        min={1}
                        max={1024}
                        value={concurrency || ''}
                        onChange={(event) => setConcurrency(event.target.value === '' ? 0 : Number(event.target.value))}
                        onBlur={(event) => {
                          if (!event.target.value || Number(event.target.value) < 1) setConcurrency(1)
                        }}
                        style={{ width: '80px' }}
                      />
                    </div>
                  </label>
                </div>

                <div className="option-strip">
                  {(['aggressive', 'balanced', 'conservative'] as EstimateProfile[]).map((item) => (
                    <button
                      key={item}
                      className={estimateProfile === item ? 'active' : ''}
                      onClick={() => setEstimateProfile(item)}
                      type="button"
                    >
                      {text.profileLabels[item]}
                    </button>
                  ))}
                </div>

                <div className="option-strip">
                  {(['inference', 'lora', 'qlora'] as DeploymentMode[]).map((item) => (
                    <button
                      key={item}
                      className={deploymentMode === item ? 'active' : ''}
                      onClick={() => setDeploymentMode(item)}
                      type="button"
                    >
                      {text.deploymentLabels[item]}
                    </button>
                  ))}
                </div>

                {deploymentMode !== 'inference' ? (
                  <div className="sub-panel">
                    <div className="field-grid two-up">
                      <label>
                        {text.labels.trainSequenceLength}
                        <input
                          type="number"
                          min={512}
                          step={512}
                          value={trainSequenceLength || ''}
                          onChange={(event) => setTrainSequenceLength(event.target.value === '' ? 0 : Number(event.target.value))}
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 512) setTrainSequenceLength(512)
                          }}
                        />
                      </label>
                      <label>
                        {text.labels.microBatchSize}
                        <input
                          type="number"
                          min={1}
                          max={64}
                          value={microBatchSize || ''}
                          onChange={(event) => setMicroBatchSize(event.target.value === '' ? 0 : Number(event.target.value))}
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 1) setMicroBatchSize(1)
                          }}
                        />
                      </label>
                    </div>

                    <div className="field-grid two-up">
                      <label>
                        {text.labels.loraRank}
                        <input
                          type="number"
                          min={4}
                          step={4}
                          max={256}
                          value={loraRank || ''}
                          onChange={(event) => setLoraRank(event.target.value === '' ? 0 : Number(event.target.value))}
                          onBlur={(event) => {
                            if (!event.target.value || Number(event.target.value) < 4) setLoraRank(4)
                          }}
                        />
                      </label>
                      <label className="checkbox-field">
                        <span>{text.labels.gradientCheckpointing}</span>
                        <input
                          type="checkbox"
                          checked={checkpointing}
                          onChange={(event) => setCheckpointing(event.target.checked)}
                        />
                      </label>
                    </div>
                  </div>
                ) : null}

                <div className="micro-note">{quantizationDescription}</div>
              </div>
            </>
          )}
        </section>

        <section className="output-column">
          {direction === 'model-to-gpu' ? (
            <div className="panel section-panel panel-order-early">
              <div className="section-head">
                <h2>{text.labels.recommendations}</h2>
                <p>{text.labels.recommendationsDesc}</p>
              </div>

              <div className="recommendation-toolbar">
                <label>
                  {text.labels.deploymentTier}
                  <select
                    value={recommendationTierFilter}
                    onChange={(event) =>
                      setRecommendationTierFilter(event.target.value as 'all' | 'workstation' | 'datacenter')
                    }
                  >
                    <option value="all">{text.labels.all}</option>
                    <option value="workstation">{text.labels.workstation}</option>
                    <option value="datacenter">{text.labels.datacenter}</option>
                  </select>
                </label>
                <label>
                  {text.labels.clusterCount}
                  <select
                    value={recommendationUnitFilter}
                    onChange={(event) => setRecommendationUnitFilter(event.target.value)}
                  >
                    <option value="all">{text.labels.all}</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="4">4</option>
                    <option value="8">8</option>
                  </select>
                </label>
                <label>
                  {text.labels.gpuModelFilter}
                  <input
                    type="text"
                    placeholder={text.labels.gpuModelFilterPlaceholder}
                    value={recommendationGpuSearch}
                    onChange={(event) => setRecommendationGpuSearch(event.target.value)}
                  />
                </label>
              </div>

              {groupedRecommendationEntries.length ? (
                <div className="recommendation-stack">
                  {groupedRecommendationEntries.map(([units, items]) => {
                    const groupOpen = openRecommendationGroups[units] ?? false

                    return (
                      <section key={units} className="recommendation-group">
                        <button
                          type="button"
                          className="recommendation-group-toggle"
                          onClick={() => toggleRecommendationGroup(units)}
                          aria-expanded={groupOpen}
                        >
                          <div>
                            <span>{text.labels.clusterGroup}</span>
                            <div className="recommendation-group-title-row">
                              <h3>{formatUnitCount(Number(units))}</h3>
                              <span className="unit-count-badge muted-badge">{items[0].gpu.unitLabel}</span>
                            </div>
                          </div>

                          <div className="recommendation-group-actions">
                            <strong>{text.candidateCount(items.length)}</strong>
                            <span className="accordion-indicator">{groupOpen ? '−' : '+'}</span>
                          </div>
                        </button>

                        {groupOpen ? (
                          <div className="recommendation-list">
                            {items.map((item) => {
                              const rowKey = `${units}-${item.gpu.id}`
                              const rowOpen = openRecommendationRows[rowKey] ?? false
                              const quantizationSupportLevel = getGpuQuantizationSupportLevel(item.gpu, quantizationId)

                              return (
                                <article key={rowKey} className="recommendation-row">
                                  <button
                                    type="button"
                                    className="recommendation-row-summary"
                                    onClick={() => toggleRecommendationRow(rowKey)}
                                    aria-expanded={rowOpen}
                                  >
                                    <div className="recommendation-row-top">
                                      <div>
                                        <div className="hardware-title">
                                          <strong>{item.gpu.name}</strong>
                                          <span className="unit-count-badge">{formatUnitCount(item.units)}</span>
                                        </div>
                                        <div className="recommendation-chips">
                                          <span className="recommendation-chip">
                                            {formatDeploymentTier(item.gpu.deploymentTier, text.deploymentTierLabels)}
                                          </span>
                                          <span className="recommendation-chip">{item.interconnectLabel}</span>
                                          <span className="recommendation-chip">
                                            {text.quantizationSupportSummary(
                                              quantization.shortLabel,
                                              formatQuantizationSupportLevel(
                                                quantizationSupportLevel,
                                                text.quantizationSupportLevels,
                                              ),
                                            )}
                                          </span>
                                        </div>
                                      </div>

                                      <div className="recommendation-row-status">
                                        <span className={`recommendation-badge is-${item.recommendationGrade}`}>
                                          {formatRecommendationGrade(item, text.recommendationGrades)}
                                        </span>
                                        <span className="accordion-indicator">{rowOpen ? '−' : '+'}</span>
                                      </div>
                                    </div>
                                  </button>

                                  {rowOpen ? (
                                    <div className="recommendation-row-details">
                                      <div className="recommendation-data five-up">
                                        <div>
                                          <span>{text.labels.unitCount}</span>
                                          <strong>{formatUnitCount(item.units)}</strong>
                                        </div>
                                        <div>
                                          <span>{text.labels.singleUnitLoad}</span>
                                          <strong>{formatGb(item.requiredPerUnitGB)}</strong>
                                        </div>
                                        <div>
                                          <span>{text.labels.singleUnitAvailable}</span>
                                          <strong>{formatGb(item.availablePerUnitGB)}</strong>
                                        </div>
                                        <div>
                                          <span>{text.labels.interconnectPenalty}</span>
                                          <strong>{formatGb(item.interconnectPenaltyGB)}</strong>
                                        </div>
                                        <div>
                                          <span>{text.labels.headroom}</span>
                                          <strong>{formatGb(item.headroomGB)}</strong>
                                        </div>
                                      </div>

                                      <div className="interconnect-callout">
                                        <div className="interconnect-head">
                                          <div>
                                            <span>{text.labels.interconnect}</span>
                                            <strong>{item.interconnectLabel}</strong>
                                          </div>
                                          <span className={`recommendation-chip risk-${item.networkRisk}`}>
                                            {text.labels.interconnectFocus}
                                          </span>
                                        </div>
                                        <p>{item.interconnectNote}</p>
                                      </div>

                                      <p>{item.gpu.summary}</p>
                                      <p className="micro-note">{item.rationale}</p>
                                      {renderCloudInstances(item.gpu, locale)}
                                    </div>
                                  ) : null}
                                </article>
                              )
                            })}
                          </div>
                        ) : null}
                      </section>
                    )
                  })}
                </div>
              ) : (
                <div className="empty-state">
                  <strong>{text.labels.noRecommendationTitle}</strong>
                  <p>{text.labels.noRecommendationDesc}</p>
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="panel section-panel">
                <div className="section-head">
                  <h2>{text.labels.currentHardware}</h2>
                  <p>{text.labels.currentHardwareDesc}</p>
                </div>

                <div className="metric-grid single-row">
                  <article className="metric-card">
                    <span>{text.labels.selectedHardware}</span>
                    {selectedGpu.category === 'platform' ? (
                      <strong>{selectedGpu.name}</strong>
                    ) : (
                      <div className="hardware-title hardware-title-selected">
                        <strong>{selectedGpu.name}</strong>
                        <span className="unit-count-badge unit-count-badge-selected">{formatUnitCount(selectedUnits)}</span>
                      </div>
                    )}
                    <p>{selectedGpu.summary}</p>
                  </article>
                  <article className="metric-card">
                    <span>{text.labels.singleUnitCapacity}</span>
                    <strong>{formatGb(selectedGpu.memoryPerUnitGB * selectedGpu.usableMemoryFactor)}</strong>
                    <p>{selectedGpu.notes}</p>
                  </article>
                  <article className="metric-card">
                    <span>{text.labels.sampleModelLoad}</span>
                    <strong>{formatGb(selectedGpuSimulation.requiredPerUnitGB)}</strong>
                    <p>
                      {locale === 'zh'
                        ? '下方列表会给出当前配置下真正能装下的模型。'
                        : 'The list below shows the models that actually fit under the current configuration.'}
                    </p>
                  </article>
                </div>
              </div>

              <div className="panel section-panel">
                <div className="section-head">
                  <h2>{text.labels.runnableModels}</h2>
                  <p>{text.labels.runnableModelsDesc}</p>
                </div>

                <div className="recommendation-grid">
                  {bestReverseResults.map((item) => (
                    <article key={item.model.id} className="recommendation-card">
                      <div className="recommendation-head">
                        <div>
                          <span>{item.model.family}</span>
                          <strong>{item.model.name}</strong>
                        </div>
                        <em>{formatParams(item.model.parameterCountB)}</em>
                      </div>

                      <div className="recommendation-metrics">
                        <div>
                          <span>{text.labels.totalRequirement}</span>
                          <strong>{formatGb(item.breakdown.totalMemoryGB)}</strong>
                        </div>
                        <div>
                          <span>{text.labels.singleUnitLoad}</span>
                          <strong>{formatGb(item.config.requiredPerUnitGB)}</strong>
                        </div>
                        <div>
                          <span>{text.labels.headroom}</span>
                          <strong>{formatGb(item.config.headroomGB)}</strong>
                        </div>
                      </div>

                      <p>{item.model.notes}</p>
                      <p className="micro-note">{item.config.rationale}</p>
                    </article>
                  ))}
                </div>

                {!bestReverseResults.length ? (
                  <div className="empty-state">
                    <strong>{text.labels.noRunnableModelTitle}</strong>
                    <p>{text.labels.noRunnableModelDesc}</p>
                  </div>
                ) : null}
              </div>
            </>
          )}

          <div className={`panel section-panel ${direction === 'model-to-gpu' ? 'panel-order-late' : ''}`}>
            <div className="section-head">
              <h2>{text.labels.memoryBreakdown}</h2>
              <p>{text.labels.memoryBreakdownDesc}</p>
            </div>

            <div className="metric-grid">
              <article className="metric-card">
                <span>{text.labels.totalRequirement}</span>
                <strong>{formatGb(breakdown.totalMemoryGB)}</strong>
                <p>{useCustomModel ? text.labels.fromCustomModel : text.labels.fromCatalog}</p>
              </article>
              <article className="metric-card">
                <span>{text.labels.weightMemory}</span>
                <strong>{formatGb(breakdown.weightMemoryGB)}</strong>
                <p>
                  {quantization.shortLabel} {text.labels.weightDescription}
                </p>
              </article>
              <article className="metric-card">
                <span>{text.labels.kvCache}</span>
                <strong>{formatGb(breakdown.kvCacheMemoryGB)}</strong>
                <p>{text.contextConcurrencySummary(formatCompact(contextTokens), concurrency)}</p>
              </article>
              <article className="metric-card">
                <span>{text.labels.tuningExtra}</span>
                <strong>{formatGb(breakdown.tuningMemoryGB)}</strong>
                <p>{text.trainingExtraSummary(text.deploymentLabels[deploymentMode])}</p>
              </article>
            </div>

            <div className="breakdown-bars">
              {[
                { label: text.breakdownLabels.weights, value: breakdown.weightMemoryGB },
                { label: text.breakdownLabels.kvCache, value: breakdown.kvCacheMemoryGB },
                { label: text.breakdownLabels.runtime, value: breakdown.runtimeMemoryGB },
                { label: text.breakdownLabels.tuning, value: breakdown.tuningMemoryGB },
              ].map((item) => (
                <div key={item.label} className="bar-row">
                  <div className="bar-label">
                    <span>{item.label}</span>
                    <strong>{formatGb(item.value)}</strong>
                  </div>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{ width: `${Math.max(6, (item.value / breakdown.totalMemoryGB) * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {breakdown.notes.length ? (
              <div className="notes-stack">
                {breakdown.notes.map((note) => (
                  <p key={note}>{note}</p>
                ))}
              </div>
            ) : null}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
