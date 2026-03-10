import { getQuantizationById, gpuCatalog, modelCatalog } from '../data/catalog'
import type { Locale } from '../i18n'
import type {
  ConfigSimulation,
  DeploymentBreakdown,
  EstimateProfile,
  GpuProfile,
  ModelPreset,
  ReverseSizingResult,
  SizingInput,
} from '../types'

const profileFactors: Record<
  EstimateProfile,
  {
    runtimeBaseGB: number
    runtimeFactor: number
    kvRuntimeFactor: number
    concurrencyFactor: number
  }
> = {
  aggressive: {
    runtimeBaseGB: 0.75,
    runtimeFactor: 0.03,
    kvRuntimeFactor: 0.015,
    concurrencyFactor: 0.25,
  },
  balanced: {
    runtimeBaseGB: 1.5,
    runtimeFactor: 0.06,
    kvRuntimeFactor: 0.03,
    concurrencyFactor: 0.5,
  },
  conservative: {
    runtimeBaseGB: 2.5,
    runtimeFactor: 0.1,
    kvRuntimeFactor: 0.05,
    concurrencyFactor: 0.8,
  },
}

function resolveInterconnectProfile(gpu: GpuProfile, units: number, locale: Locale) {
  if (units <= 1) {
    return {
      interconnectType: 'single' as const,
      interconnectLabel: locale === 'zh' ? '单卡本地' : 'Single-device local',
      interconnectNote:
        locale === 'zh'
          ? '单卡无需跨卡同步，重点只在显存余量与本地吞吐。'
          : 'Single-device deployment avoids cross-device sync. Focus is on local memory headroom and throughput.',
      networkRisk: 'low' as const,
      riskPenalty: 0,
    }
  }

  if (gpu.category === 'superchip') {
    const mediumRisk = units >= 4
    return {
      interconnectType: 'superchip' as const,
      interconnectLabel: locale === 'zh' ? 'NVLink-C2C / NVLink Switch' : 'NVLink-C2C / NVLink Switch',
      interconnectNote:
        locale === 'zh'
          ? 'Superchip 内部互联很强，但 2x 以上扩展时仍要关注 NVLink 域边界与 IB 网络规划。'
          : 'Interconnect inside a superchip is strong, but scaling beyond one or two superchips still depends on NVLink domains and IB planning.',
      networkRisk: mediumRisk ? ('medium' as const) : ('low' as const),
      riskPenalty: mediumRisk ? 0.04 : 0.02,
    }
  }

  if (gpu.deploymentTier === 'datacenter') {
    if (units <= 2) {
      return {
        interconnectType: 'nvlink' as const,
        interconnectLabel: locale === 'zh' ? 'NVLink / NVSwitch' : 'NVLink / NVSwitch',
        interconnectNote:
          locale === 'zh'
            ? '双卡优先放在同一台机器的 NVLink 或 NVSwitch 域内；如果跨节点拆分，请额外预留 IB 同步开销。'
            : 'For 2-GPU deployments, keep the pair inside the same NVLink or NVSwitch domain when possible. If you split across nodes, reserve extra IB sync overhead.',
        networkRisk: 'low' as const,
        riskPenalty: 0.02,
      }
    }

    if (units <= 4) {
      return {
        interconnectType: 'nvswitch' as const,
        interconnectLabel: locale === 'zh' ? 'NVLink / NVSwitch 优先' : 'NVLink / NVSwitch preferred',
        interconnectNote:
          locale === 'zh'
            ? '4 卡以内建议优先同一 NVSwitch 域；如果跨节点扩展，IB 网络会开始影响 all-reduce 和 KV 同步效率。'
            : 'Up to 4 GPUs should ideally stay inside one NVSwitch domain. Once you span nodes, IB starts to affect all-reduce and KV sync efficiency.',
        networkRisk: 'medium' as const,
        riskPenalty: 0.05,
      }
    }

    return {
      interconnectType: 'ib' as const,
      interconnectLabel: locale === 'zh' ? 'NVSwitch + IB 重点规划' : 'NVSwitch + IB critical path',
      interconnectNote:
        locale === 'zh'
          ? '8 卡级部署应优先保证同机 NVSwitch；一旦跨节点，IB 网络会显著影响张量并行、all-reduce 和尾延迟。'
          : 'At 8-GPU scale, keep workloads inside one NVSwitch box when possible. Once you cross nodes, IB becomes a major factor for tensor parallelism, all-reduce, and tail latency.',
      networkRisk: 'high' as const,
      riskPenalty: 0.09,
    }
  }

  if (units === 2) {
    return {
      interconnectType: 'pcie' as const,
      interconnectLabel: locale === 'zh' ? 'PCIe / Workstation P2P' : 'PCIe / Workstation P2P',
      interconnectNote:
        locale === 'zh'
          ? '工作站双卡通常以 PCIe P2P 为主，适合验证和中低并发；生产高并发时要谨慎评估跨卡开销。'
          : 'Dual-GPU workstations usually rely on PCIe P2P. This is fine for validation and moderate concurrency, but high-concurrency production needs careful cross-device evaluation.',
      networkRisk: 'medium' as const,
      riskPenalty: 0.05,
    }
  }

  return {
    interconnectType: 'pcie' as const,
    interconnectLabel: locale === 'zh' ? 'PCIe / Host Fabric' : 'PCIe / Host Fabric',
    interconnectNote:
      locale === 'zh'
        ? '工作站多卡缺少 NVSwitch，跨卡和跨 NUMA 通信更容易吃掉余量，更适合作为测试或中低并发方案。'
        : 'Multi-GPU workstations lack NVSwitch, so cross-card and cross-NUMA traffic consumes more headroom. Treat these layouts as test-oriented or moderate-concurrency options.',
    networkRisk: 'high' as const,
    riskPenalty: 0.1,
  }
}

function resolveRecommendationGrade(
  headroomRatio: number,
  fits: boolean,
  riskPenalty: number,
  interconnectPenaltyRatio: number,
) {
  if (!fits) {
    return 'testOnly' as const
  }

  const effectiveHeadroom = headroomRatio - riskPenalty - interconnectPenaltyRatio * 0.4

  if (effectiveHeadroom >= 0.22) {
    return 'recommended' as const
  }

  if (effectiveHeadroom >= 0.08) {
    return 'tight' as const
  }

  return 'testOnly' as const
}

function estimateTuningMemory(model: ModelPreset, sizing: SizingInput) {
  if (sizing.deploymentMode === 'inference') {
    return {
      tuningMemoryGB: 0,
      adapterMemoryGB: 0,
      activationMemoryGB: 0,
    }
  }

  const rankFactor = sizing.loraRank / 16
  const targetCoverage = sizing.deploymentMode === 'lora' ? 1 : 0.85
  const trainableParamsB = model.parameterCountB * 0.0028 * rankFactor * targetCoverage
  const adapterStateBytes = sizing.deploymentMode === 'lora' ? 12 : 10
  const adapterMemoryGB = trainableParamsB * adapterStateBytes

  const activationMultiplier = sizing.checkpointing
    ? sizing.deploymentMode === 'qlora'
      ? 5.5
      : 6.5
    : sizing.deploymentMode === 'qlora'
      ? 8
      : 10

  const activationMemoryGB =
    (model.layers *
      model.hiddenSize *
      sizing.trainSequenceLength *
      sizing.microBatchSize *
      activationMultiplier *
      2) /
    1_000_000_000

  const optimizerWorkspaceGB =
    0.8 +
    sizing.microBatchSize * 0.35 +
    (sizing.deploymentMode === 'qlora' ? 0.45 : 0.9)

  return {
    tuningMemoryGB: adapterMemoryGB + activationMemoryGB + optimizerWorkspaceGB,
    adapterMemoryGB,
    activationMemoryGB,
  }
}

export function estimateDeploymentMemory(
  model: ModelPreset,
  quantizationId: string,
  sizing: SizingInput,
  locale: Locale = 'zh',
): DeploymentBreakdown {
  const quantization = getQuantizationById(quantizationId)
  const profile = profileFactors[sizing.estimateProfile]
  const notes: string[] = []

  const weightMemoryGB =
    model.parameterCountB *
    quantization.bytesPerParameter *
    (1 + quantization.weightOverhead)

  const kvPerTokenGB =
    (2 * model.layers * model.kvHeads * model.headDim * quantization.kvBytesPerElement) /
    1_000_000_000

  const kvCacheMemoryGB = kvPerTokenGB * sizing.contextTokens * sizing.concurrency

  const runtimeMemoryGB =
    profile.runtimeBaseGB +
    weightMemoryGB * profile.runtimeFactor +
    kvCacheMemoryGB * profile.kvRuntimeFactor +
    Math.log2(sizing.concurrency + 1) * profile.concurrencyFactor

  const tuning = estimateTuningMemory(model, sizing)

  if (model.activeParameterCountB) {
    notes.push(
      locale === 'zh'
        ? `MoE 提示：推理时通常需要装载约 ${model.parameterCountB}B 总参数，但每 token 活跃参数约 ${model.activeParameterCountB}B。`
        : `MoE note: inference usually loads about ${model.parameterCountB}B total parameters, while active parameters per token are about ${model.activeParameterCountB}B.`,
    )
  }

  if (sizing.contextTokens >= 32768) {
    notes.push(
      locale === 'zh'
        ? '长上下文会让 KV Cache 成为主要显存消耗之一。'
        : 'Long context makes KV cache one of the main memory consumers.',
    )
  }

  if (sizing.concurrency >= 8) {
    notes.push(
      locale === 'zh'
        ? '并发较高时，建议优先关注 KV Cache 与碎片余量，而不只是权重是否装得下。'
        : 'At higher concurrency, pay attention to KV cache growth and fragmentation headroom, not only whether weights fit.',
    )
  }

  if (sizing.deploymentMode !== 'inference') {
    notes.push(
      locale === 'zh'
        ? '微调估算已包含 LoRA/QLoRA adapter、激活和训练态工作区的粗略开销。'
        : 'Tuning estimates already include rough overhead for LoRA or QLoRA adapters, activations, and training workspace.',
    )
  }

  return {
    weightMemoryGB,
    kvCacheMemoryGB,
    runtimeMemoryGB,
    tuningMemoryGB: tuning.tuningMemoryGB,
    adapterMemoryGB: tuning.adapterMemoryGB,
    activationMemoryGB: tuning.activationMemoryGB,
    totalMemoryGB:
      weightMemoryGB + kvCacheMemoryGB + runtimeMemoryGB + tuning.tuningMemoryGB,
    notes,
  }
}

export function simulateConfiguration(
  breakdown: DeploymentBreakdown,
  gpu: GpuProfile,
  requestedUnits: number,
  sizing: SizingInput,
  locale: Locale = 'zh',
): ConfigSimulation {
  const units = gpu.category === 'platform' ? 1 : Math.min(Math.max(1, requestedUnits), gpu.maxUnits)
  const availablePerUnitGB = gpu.memoryPerUnitGB * gpu.usableMemoryFactor
  const interconnectPenaltyGB = Math.max(0, units - 1) * gpu.interconnectPenaltyGB
  const interconnectProfile = resolveInterconnectProfile(gpu, units, locale)

  const weightShardDivisor = gpu.category === 'platform' ? 1 : units
  const kvShardDivisor = gpu.category === 'platform' ? 1 : Math.max(1, units)
  const runtimeShardDivisor = gpu.category === 'platform' ? 1 : Math.max(1, Math.min(units, 2))
  const tuningShardDivisor =
    gpu.category === 'platform' || sizing.deploymentMode === 'inference'
      ? 1
      : Math.max(1, Math.min(units, 2))

  const requiredPerUnitGB =
    breakdown.weightMemoryGB / weightShardDivisor +
    breakdown.kvCacheMemoryGB / kvShardDivisor +
    breakdown.runtimeMemoryGB / runtimeShardDivisor +
    breakdown.tuningMemoryGB / tuningShardDivisor +
    interconnectPenaltyGB

  const totalAvailableGB = availablePerUnitGB * units
  const totalRequiredGB = requiredPerUnitGB * units
  const headroomGB = availablePerUnitGB - requiredPerUnitGB
  const headroomRatio = headroomGB / availablePerUnitGB
  const fits = headroomGB >= 0
  const interconnectPenaltyRatio = interconnectPenaltyGB / Math.max(availablePerUnitGB, 1)
  const recommendationGrade = resolveRecommendationGrade(
    headroomRatio,
    fits,
    interconnectProfile.riskPenalty,
    interconnectPenaltyRatio,
  )

  let rationale = locale === 'zh' ? '勉强可行，建议保留更多余量。' : 'This fits, but keeping more headroom would be safer.'
  if (recommendationGrade === 'recommended') {
    rationale =
      locale === 'zh'
        ? '余量与互联风险都较稳，适合优先进入生产验证。'
        : 'Both headroom and interconnect risk are in a good range for production validation.'
  } else if (recommendationGrade === 'tight') {
    rationale =
      locale === 'zh'
        ? '显存能装下，但多卡同步或上下文放大后会比较紧。'
        : 'The model fits in memory, but multi-GPU sync or larger contexts will make the layout tight.'
  } else if (fits) {
    rationale =
      locale === 'zh'
        ? '可以跑通，但更适合作为测试或压测起点。'
        : 'This should run, but it is better treated as a testing or benchmarking baseline.'
  } else if (!fits) {
    rationale =
      locale === 'zh'
        ? '单卡负载超出可用显存，需要增加卡数或降低模型规模。'
        : 'Per-device load exceeds usable memory. Add more GPUs or lower the model size.'
  }

  return {
    gpu,
    units,
    gpuCores: gpu.gpuCoresPerUnit * units,
    availablePerUnitGB,
    requiredPerUnitGB,
    totalAvailableGB,
    totalRequiredGB,
    headroomGB,
    headroomRatio,
    fits,
    recommendationGrade,
    interconnectType: interconnectProfile.interconnectType,
    interconnectLabel: interconnectProfile.interconnectLabel,
    interconnectNote: interconnectProfile.interconnectNote,
    interconnectPenaltyGB,
    networkRisk: interconnectProfile.networkRisk,
    rationale,
  }
}

export function recommendConfigurations(
  breakdown: DeploymentBreakdown,
  sizing: SizingInput,
  locale: Locale = 'zh',
): ConfigSimulation[] {
  const candidates = gpuCatalog
    .map((gpu) => {
      for (let units = 1; units <= gpu.maxUnits; units += 1) {
        const simulation = simulateConfiguration(breakdown, gpu, units, sizing, locale)
        if (simulation.fits) {
          return simulation
        }
      }

      return simulateConfiguration(breakdown, gpu, gpu.maxUnits, sizing, locale)
    })
    .filter(Boolean)

  return candidates.sort((left, right) => {
    if (left.fits !== right.fits) {
      return left.fits ? -1 : 1
    }

    const gradeRank = {
      recommended: 0,
      tight: 1,
      testOnly: 2,
    }

    if (gradeRank[left.recommendationGrade] !== gradeRank[right.recommendationGrade]) {
      return gradeRank[left.recommendationGrade] - gradeRank[right.recommendationGrade]
    }

    const riskRank = {
      low: 0,
      medium: 1,
      high: 2,
    }

    if (riskRank[left.networkRisk] !== riskRank[right.networkRisk]) {
      return riskRank[left.networkRisk] - riskRank[right.networkRisk]
    }

    if (left.gpu.category !== right.gpu.category) {
      if (left.gpu.category === 'platform') {
        return 1
      }

      if (right.gpu.category === 'platform') {
        return -1
      }
    }

    if (left.gpuCores !== right.gpuCores) {
      return left.gpuCores - right.gpuCores
    }

    if (left.interconnectPenaltyGB !== right.interconnectPenaltyGB) {
      return left.interconnectPenaltyGB - right.interconnectPenaltyGB
    }

    if (left.gpu.generationRank !== right.gpu.generationRank) {
      return right.gpu.generationRank - left.gpu.generationRank
    }

    return right.headroomRatio - left.headroomRatio
  })
}

export function recommendModelsForGpu(
  gpu: GpuProfile,
  units: number,
  quantizationId: string,
  sizing: SizingInput,
  locale: Locale = 'zh',
): ReverseSizingResult[] {
  return modelCatalog
    .map((model) => {
      const breakdown = estimateDeploymentMemory(model, quantizationId, sizing, locale)
      const config = simulateConfiguration(breakdown, gpu, units, sizing, locale)

      return {
        model,
        breakdown,
        config,
      }
    })
    .filter((item) => item.config.fits)
    .sort((left, right) => {
      if (left.model.parameterCountB !== right.model.parameterCountB) {
        return right.model.parameterCountB - left.model.parameterCountB
      }

      return left.config.headroomRatio - right.config.headroomRatio
    })
}