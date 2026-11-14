// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '../adapter/device';
import {Sampler, type SamplerProps} from '../adapter/resources/sampler';
import {incrementFactoryCounter} from '../utils/factory-profiler';

type SamplerCacheEntry<HandleT> = {
  handle: HandleT;
  references: number;
  lastUsed: number;
};

/** Device-local native sampler-handle cache shared by public wrappers and textures. */
export class SamplerCache<HandleT> {
  readonly device: Device;

  private readonly _destroyHandle: (handle: HandleT) => void;
  private _entries = new Map<string, SamplerCacheEntry<HandleT>>();
  private _useCounter = 0;
  private _generation = 0;

  constructor(device: Device, destroyHandle: (handle: HandleT) => void) {
    this.device = device;
    this._destroyHandle = destroyHandle;
  }

  get generation(): number {
    return this._generation;
  }

  get size(): number {
    return this._entries.size;
  }

  /** Returns a retained native handle for normalized sampler props. */
  retain(
    props: SamplerProps,
    createHandle: () => HandleT
  ): {key: string; handle: HandleT; hit: boolean} {
    const key = getSamplerCacheKey(this.device, props);
    let entry = this._entries.get(key);
    if (entry) {
      entry.references++;
      entry.lastUsed = this._nextUse();
      incrementFactoryCounter(this.device, 'samplerCacheHits');
      return {key, handle: entry.handle, hit: true};
    }

    const handle = createHandle();
    entry = {handle, references: 1, lastUsed: this._nextUse()};
    this._entries.set(key, entry);
    incrementFactoryCounter(this.device, 'samplerCacheMisses');
    this._enforceLimit();
    return {key, handle, hit: false};
  }

  /** Releases one wrapper reference, allowing an unused native handle to be evicted. */
  release(key: string | undefined): void {
    if (!key) {
      return;
    }
    const entry = this._entries.get(key);
    if (!entry) {
      return;
    }
    entry.references = Math.max(0, entry.references - 1);
    entry.lastUsed = this._nextUse();
    this._enforceLimit();
  }

  /** Destroys every cached native handle after device loss or context destruction. */
  invalidate(): void {
    this._generation++;
    for (const entry of this._entries.values()) {
      this._destroyHandle(entry.handle);
    }
    this._entries.clear();
    incrementFactoryCounter(this.device, 'samplerCacheInvalidations');
  }

  private _enforceLimit(): void {
    const cacheLimit = this.device.props._samplerCacheLimit;
    if (cacheLimit === undefined || cacheLimit === null || !Number.isFinite(cacheLimit)) {
      return;
    }
    const normalizedLimit = Math.max(0, Math.floor(cacheLimit));
    while (this._entries.size > normalizedLimit) {
      const unusedEntry = Array.from(this._entries.entries())
        .filter(([, entry]) => entry.references === 0)
        .sort((left, right) => left[1].lastUsed - right[1].lastUsed)[0];
      if (!unusedEntry) {
        return;
      }
      this._entries.delete(unusedEntry[0]);
      this._destroyHandle(unusedEntry[1].handle);
      incrementFactoryCounter(this.device, 'samplerCacheEvictions');
    }
  }

  private _nextUse(): number {
    this._useCounter++;
    return this._useCounter;
  }
}

/** Stable sampler cache key. Identity props are excluded; capabilities are included. */
export function getSamplerCacheKey(device: Device, props: SamplerProps): string {
  const {
    id: _id,
    handle: _handle,
    userData: _userData,
    _isHandleBorrowed: _isHandleBorrowed,
    ...normalizedProps
  } = {...Sampler.defaultProps, ...props};

  return JSON.stringify({
    deviceType: device.type,
    featureLevel: device.info?.featureLevel ?? null,
    anisotropicFiltering: device.features.has('texture-filterable-anisotropic-webgl'),
    props: {
      type: normalizedProps.type,
      addressModeU: normalizedProps.addressModeU,
      addressModeV: normalizedProps.addressModeV,
      addressModeW: normalizedProps.addressModeW,
      magFilter: normalizedProps.magFilter,
      minFilter: normalizedProps.minFilter,
      mipmapFilter: normalizedProps.mipmapFilter,
      lodMinClamp: normalizedProps.lodMinClamp,
      lodMaxClamp: normalizedProps.lodMaxClamp,
      compare: normalizedProps.compare,
      maxAnisotropy: normalizedProps.maxAnisotropy
    }
  });
}
