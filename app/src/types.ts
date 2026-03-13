export type CalculatorDirection = 'model-to-gpu' | 'gpu-to-model'

export type DeploymentMode = 'inference' | 'lora' | 'qlora'

export type EstimateProfile = 'aggressive' | 'balanced' | 'conservative'

export type GpuCategory = 'single' | 'superchip' | 'platform'

export type GpuFeature = 'fp8'

export type CloudProvider = 'aws' | 'azure' | 'gcp' | 'oracle'

export type QuantizationSupportLevel = 'supported' | 'conditional' | 'notRecommended'

export interface CloudGpuInstance {
  provider: CloudProvider
  instanceType: string
  memoryPerGpuGB: number
  notes?: string
}

export interface QuantizationPreset {
  id: string
  label: string
  shortLabel: string
  bytesPerParameter: number
  kvBytesPerElement: number
  weightOverhead: number
  description: string
  requiredGpuFeatures?: GpuFeature[]
}

export interface ModelPreset {
  id: string
  name: string
  family: string
  parameterCountB: number
  activeParameterCountB?: number
  layers: number
  hiddenSize: number
  kvHeads: number
  headDim: number
  defaultContext: number
  maxContext: number
  defaultQuantizationId: string
  notes: string
}

export interface GpuProfile {
  id: string
  name: string
  category: GpuCategory
  deploymentTier: 'workstation' | 'datacenter' | 'platform'
  memoryPerUnitGB: number
  gpuCoresPerUnit: number
  maxUnits: number
  usableMemoryFactor: number
  interconnectPenaltyGB: number
  generationRank: number
  unitLabel: string
  summary: string
  notes: string
  supportedGpuFeatures?: GpuFeature[]
  cloudInstances?: CloudGpuInstance[]
}

export interface SizingInput {
  contextTokens: number
  concurrency: number
  deploymentMode: DeploymentMode
  estimateProfile: EstimateProfile
  trainSequenceLength: number
  microBatchSize: number
  loraRank: number
  checkpointing: boolean
}

export interface DeploymentBreakdown {
  weightMemoryGB: number
  kvCacheMemoryGB: number
  runtimeMemoryGB: number
  tuningMemoryGB: number
  adapterMemoryGB: number
  activationMemoryGB: number
  totalMemoryGB: number
  notes: string[]
}

export interface ConfigSimulation {
  gpu: GpuProfile
  units: number
  gpuCores: number
  availablePerUnitGB: number
  requiredPerUnitGB: number
  totalAvailableGB: number
  totalRequiredGB: number
  headroomGB: number
  headroomRatio: number
  fits: boolean
  recommendationGrade: 'recommended' | 'tight' | 'testOnly'
  interconnectType: 'single' | 'pcie' | 'nvlink' | 'nvswitch' | 'superchip' | 'ib'
  interconnectLabel: string
  interconnectNote: string
  interconnectPenaltyGB: number
  networkRisk: 'low' | 'medium' | 'high'
  rationale: string
}

export interface ReverseSizingResult {
  model: ModelPreset
  breakdown: DeploymentBreakdown
  config: ConfigSimulation
}