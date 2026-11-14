// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '../adapter/device';

/** Internal profiler bucket used by core factories to report cache efficiency. */
export const CPU_HOTSPOT_PROFILER_MODULE = 'cpu-hotspot-profiler';

/** Cache counters shared by PipelineFactory, BindGroupFactory, and sampler caches. */
export type FactoryProfilerCounters = {
  pipelineCacheHits?: number;
  pipelineCacheMisses?: number;
  pipelineCachePendingHits?: number;
  pipelineCacheEvictions?: number;
  pipelineCacheInvalidations?: number;
  pipelineCompilationErrors?: number;
  bindGroupCacheHits?: number;
  bindGroupCacheMisses?: number;
  bindGroupCacheInvalidations?: number;
  samplerCacheHits?: number;
  samplerCacheMisses?: number;
  samplerCacheEvictions?: number;
  samplerCacheInvalidations?: number;
};

/** Increments one optional counter without adding profiling overhead to normal runs. */
export function incrementFactoryCounter(
  device: Device,
  counter: keyof FactoryProfilerCounters
): void {
  const profiler = device.userData[CPU_HOTSPOT_PROFILER_MODULE] as
    | (FactoryProfilerCounters & {enabled?: boolean})
    | undefined;
  if (profiler?.enabled) {
    profiler[counter] = (profiler[counter] || 0) + 1;
  }
}
