// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  Bindings,
  BindingsByGroup,
  ComputeShaderLayout,
  ShaderLayout
} from '../adapter/types/shader-layout';
import type {Device} from '../adapter/device';
import type {ComputePipeline} from '../adapter/resources/compute-pipeline';
import type {RenderPipeline} from '../adapter/resources/render-pipeline';
import {Buffer} from '../adapter/resources/buffer';
import {ExternalTexture} from '../adapter/resources/external-texture';
import {Sampler} from '../adapter/resources/sampler';
import {Texture} from '../adapter/resources/texture';
import {TextureView} from '../adapter/resources/texture-view';
import {normalizeBindingsByGroup} from '../adapter-utils/bind-groups';
import {incrementFactoryCounter} from '../utils/factory-profiler';

type AnyPipeline = RenderPipeline | ComputePipeline;
type AnyShaderLayout = ShaderLayout | ComputeShaderLayout;
type BindGroupCacheKeys = Partial<Record<number, object>>;
type BindGroupMap = Partial<Record<number, unknown>>;
type LayoutCache = Partial<Record<number, object>>;
type LayoutBindGroupCacheEntry = {
  bindGroup: unknown | null;
  bindings: Bindings;
  generations: number[];
};
type LayoutBindGroupCache = {
  bindGroupsBySource: WeakMap<object, LayoutBindGroupCacheEntry>;
  emptyBindGroup?: unknown;
};

export class BindGroupFactory {
  readonly device: Device;

  private _layoutCacheByPipeline: WeakMap<AnyPipeline, LayoutCache> = new WeakMap();
  private _bindGroupCacheByLayout: WeakMap<object, LayoutBindGroupCache> = new WeakMap();

  constructor(device: Device) {
    this.device = device;
  }

  /** Drops all bind-group and layout caches after device loss or resource invalidation. */
  invalidate(): void {
    this._layoutCacheByPipeline = new WeakMap();
    this._bindGroupCacheByLayout = new WeakMap();
    incrementFactoryCounter(this.device, 'bindGroupCacheInvalidations');
  }

  getBindGroups(
    pipeline: AnyPipeline,
    bindings?: Bindings | BindingsByGroup,
    bindGroupCacheKeys?: BindGroupCacheKeys
  ): BindGroupMap {
    if (this.device.type !== 'webgpu' || pipeline.shaderLayout.bindings.length === 0) {
      return {};
    }

    const bindingsByGroup = normalizeBindingsByGroup(pipeline.shaderLayout, bindings);
    const resolvedBindGroups: BindGroupMap = {};

    for (const group of getBindGroupIndicesUpToMax(pipeline.shaderLayout.bindings)) {
      const groupBindings = bindingsByGroup[group];
      const bindGroupLayout = this._getBindGroupLayout(pipeline, group);
      const bindGroupLabel = getBindGroupLabel(pipeline, pipeline.shaderLayout, group);

      if (!groupBindings || Object.keys(groupBindings).length === 0) {
        if (!hasBindingsInGroup(pipeline.shaderLayout.bindings, group)) {
          resolvedBindGroups[group] = this._getEmptyBindGroup(
            bindGroupLayout,
            pipeline.shaderLayout,
            group,
            bindGroupLabel
          );
        }
        continue;
      }

      const bindGroupCacheKey = bindGroupCacheKeys?.[group];
      if (bindGroupCacheKey) {
        const layoutCache = this._getLayoutBindGroupCache(bindGroupLayout);
        const cachedEntry = layoutCache.bindGroupsBySource.get(bindGroupCacheKey);
        if (cachedEntry && this._isBindGroupCacheEntryUsable(cachedEntry, groupBindings)) {
          cachedEntry.bindGroup = cachedEntry.bindGroup || null;
          resolvedBindGroups[group] = cachedEntry.bindGroup;
          incrementFactoryCounter(this.device, 'bindGroupCacheHits');
          continue;
        }
        if (cachedEntry) {
          incrementFactoryCounter(this.device, 'bindGroupCacheInvalidations');
        }

        const bindGroup = this.device._createBindGroupWebGPU(
          bindGroupLayout,
          pipeline.shaderLayout,
          groupBindings,
          group,
          bindGroupLabel
        );
        layoutCache.bindGroupsBySource.set(bindGroupCacheKey, {
          bindGroup,
          bindings: {...groupBindings},
          generations: getBindingGenerations(groupBindings)
        });
        incrementFactoryCounter(this.device, 'bindGroupCacheMisses');
        resolvedBindGroups[group] = bindGroup;
      } else {
        resolvedBindGroups[group] = this.device._createBindGroupWebGPU(
          bindGroupLayout,
          pipeline.shaderLayout,
          groupBindings,
          group,
          bindGroupLabel
        );
      }
    }

    return resolvedBindGroups;
  }

  private _getBindGroupLayout(pipeline: AnyPipeline, group: number): object {
    let layoutCache = this._layoutCacheByPipeline.get(pipeline);
    if (!layoutCache) {
      layoutCache = {};
      this._layoutCacheByPipeline.set(pipeline, layoutCache);
    }

    layoutCache[group] ||= this.device._createBindGroupLayoutWebGPU(pipeline, group) as object;
    return layoutCache[group];
  }

  private _getEmptyBindGroup(
    bindGroupLayout: object,
    shaderLayout: AnyShaderLayout,
    group: number,
    label: string
  ): unknown {
    const layoutCache = this._getLayoutBindGroupCache(bindGroupLayout);
    layoutCache.emptyBindGroup ||=
      this.device._createBindGroupWebGPU(bindGroupLayout, shaderLayout, {}, group, label) || null;
    return layoutCache.emptyBindGroup;
  }

  private _isBindGroupCacheEntryUsable(
    cachedEntry: LayoutBindGroupCacheEntry,
    bindings: Bindings
  ): boolean {
    if (!isBindingsSnapshotEqual(cachedEntry.bindings, bindings)) {
      return false;
    }
    if (!areBindingsUsable(bindings)) {
      return false;
    }
    const generations = getBindingGenerations(bindings);
    return (
      generations.length === cachedEntry.generations.length &&
      generations.every((generation, index) => generation === cachedEntry.generations[index])
    );
  }

  private _getLayoutBindGroupCache(bindGroupLayout: object): LayoutBindGroupCache {
    let layoutCache = this._bindGroupCacheByLayout.get(bindGroupLayout);
    if (!layoutCache) {
      layoutCache = {bindGroupsBySource: new WeakMap()};
      this._bindGroupCacheByLayout.set(bindGroupLayout, layoutCache);
    }
    return layoutCache;
  }
}

export function _getDefaultBindGroupFactory(device: Device): BindGroupFactory {
  device._factories.bindGroupFactory ||= new BindGroupFactory(device);
  return device._factories.bindGroupFactory as BindGroupFactory;
}

function getBindGroupIndicesUpToMax(bindings: AnyShaderLayout['bindings']): number[] {
  const maxGroup = bindings.reduce(
    (highestGroup, binding) => Math.max(highestGroup, binding.group),
    -1
  );
  return Array.from({length: maxGroup + 1}, (_, group) => group);
}

function hasBindingsInGroup(bindings: AnyShaderLayout['bindings'], group: number): boolean {
  return bindings.some(binding => binding.group === group);
}

function getBindGroupLabel(
  pipeline: AnyPipeline,
  shaderLayout: AnyShaderLayout,
  group: number
): string {
  const bindingNames = shaderLayout.bindings
    .filter(binding => binding.group === group)
    .sort((left, right) => left.location - right.location)
    .map(binding => binding.name);
  const bindingSuffix = bindingNames.length > 0 ? bindingNames.join(',') : 'empty';
  return `${pipeline.id}/group${group}[${bindingSuffix}]`;
}

function isBindingsSnapshotEqual(left: Bindings, right: Bindings): boolean {
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  if (leftKeys.length !== rightKeys.length) {
    return false;
  }
  return leftKeys.every(key => key in right && left[key] === right[key]);
}

function areBindingsUsable(bindings: Bindings): boolean {
  return Object.values(bindings).every(binding => isBindingUsable(binding));
}

function isBindingUsable(binding: Bindings[string]): boolean {
  if (binding instanceof Texture) {
    return !binding.destroyed && !binding.view.destroyed && !binding.sampler.destroyed;
  }
  if (binding instanceof TextureView) {
    return !binding.destroyed && !binding.texture.destroyed;
  }
  if (binding instanceof ExternalTexture) {
    return !binding.destroyed && !binding.sampler.destroyed;
  }
  if (binding instanceof Sampler || binding instanceof Buffer) {
    return !binding.destroyed;
  }
  if (binding && typeof binding === 'object' && 'buffer' in binding) {
    return !binding.buffer.destroyed;
  }
  return true;
}

function getBindingGenerations(bindings: Bindings): number[] {
  return Object.values(bindings).map(binding => getBindingGeneration(binding));
}

function getBindingGeneration(binding: Bindings[string]): number {
  if (binding && typeof binding === 'object') {
    if ('generation' in binding && typeof binding.generation === 'number') {
      return binding.generation;
    }
    if ('buffer' in binding && binding.buffer instanceof Object) {
      const buffer = binding.buffer as {generation?: unknown};
      if (typeof buffer.generation === 'number') {
        return buffer.generation;
      }
    }
  }
  return 0;
}
