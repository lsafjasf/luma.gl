// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Device} from '../adapter/device';
import {ComputePipeline, type ComputePipelineProps} from '../adapter/resources/compute-pipeline';
import {RenderPipeline, type RenderPipelineProps} from '../adapter/resources/render-pipeline';
import {DeviceResource} from '../adapter/resources/device-resource';
import type {SharedRenderPipeline} from '../adapter/resources/shared-render-pipeline';
import {log} from '../utils/log';
import {uid} from '../utils/uid';
import {incrementFactoryCounter} from '../utils/factory-profiler';
import type {CoreModuleState} from './core-module-state';

export type PipelineFactoryProps = RenderPipelineProps;

type CacheItem<ResourceT extends DeviceResource<any>> = {
  resource: ResourceT;
  useCount: number;
  lastUsed: number;
};
type PendingCacheItem<ResourceT extends DeviceResource<any>> = {
  promise: Promise<ResourceT>;
  useCount: number;
  lastUsed: number;
  generation: number;
};

const asyncCompilationByDevice = new WeakMap<Device, Promise<unknown>[]>();

/**
 * Efficiently creates / caches pipelines
 */
export class PipelineFactory {
  static defaultProps: Required<PipelineFactoryProps> = {...RenderPipeline.defaultProps};

  /** Get the singleton default pipeline factory for the specified device */
  static getDefaultPipelineFactory(device: Device): PipelineFactory {
    const moduleData = device.getModuleData<CoreModuleState>('@luma.gl/core');
    moduleData.defaultPipelineFactory ||= new PipelineFactory(device);
    return moduleData.defaultPipelineFactory;
  }

  /**
   * Opens a synchronous construction scope that collects asynchronous pipeline work.
   *
   * Orchestrators use this while constructing several independent pipeline owners. The scope must
   * be closed before yielding back to the event loop.
   *
   * @internal
   */
  static beginAsyncCompilation(device: Device): Promise<unknown>[] {
    if (asyncCompilationByDevice.has(device)) {
      throw new Error('An asynchronous pipeline compilation scope is already active');
    }
    const compilation: Promise<unknown>[] = [];
    asyncCompilationByDevice.set(device, compilation);
    return compilation;
  }

  /** @internal Returns the construction scope currently collecting pipelines for this device. */
  static getAsyncCompilation(device: Device): Promise<unknown>[] | undefined {
    return asyncCompilationByDevice.get(device);
  }

  /** @internal Closes a construction scope before its promises are awaited. */
  static endAsyncCompilation(device: Device, compilation: Promise<unknown>[]): void {
    if (asyncCompilationByDevice.get(device) !== compilation) {
      throw new Error('The asynchronous pipeline compilation scope is not active');
    }
    asyncCompilationByDevice.delete(device);
  }

  readonly device: Device;

  private _hashCounter: number = 0;
  private _useCounter: number = 0;
  private _cacheGeneration: number = 0;
  private _capabilityHash: number | null = null;
  private readonly _hashes: Record<string, number> = {};
  private readonly _renderPipelineCache: Record<string, CacheItem<RenderPipeline>> = {};
  private readonly _computePipelineCache: Record<string, CacheItem<ComputePipeline>> = {};
  private readonly _pendingRenderPipelineCache: Record<string, PendingCacheItem<RenderPipeline>> =
    {};
  private readonly _pendingComputePipelineCache: Record<string, PendingCacheItem<ComputePipeline>> =
    {};
  private readonly _sharedRenderPipelineCache: Record<string, CacheItem<SharedRenderPipeline>> = {};

  get [Symbol.toStringTag](): string {
    return 'PipelineFactory';
  }

  toString(): string {
    return `PipelineFactory(${this.device.id})`;
  }

  constructor(device: Device) {
    this.device = device;
    const moduleData = device.getModuleData<CoreModuleState>('@luma.gl/core');
    moduleData.pipelineFactories ||= new Set();
    moduleData.pipelineFactories.add(new WeakRef(this));
  }

  /** Monotonic cache generation, advanced when device capabilities or handles become invalid. */
  get cacheGeneration(): number {
    return this._cacheGeneration;
  }

  /** Current retained cache sizes for diagnostics and tests. */
  get cacheSize(): {
    renderPipelines: number;
    computePipelines: number;
    pendingRenderPipelines: number;
    pendingComputePipelines: number;
    sharedRenderPipelines: number;
  } {
    return {
      renderPipelines: Object.keys(this._renderPipelineCache).length,
      computePipelines: Object.keys(this._computePipelineCache).length,
      pendingRenderPipelines: Object.keys(this._pendingRenderPipelineCache).length,
      pendingComputePipelines: Object.keys(this._pendingComputePipelineCache).length,
      sharedRenderPipelines: Object.keys(this._sharedRenderPipelineCache).length
    };
  }

  /**
   * Drops every cached pipeline, pending compilation, and shared WebGL program.
   *
   * Device loss, context restoration, and explicit cache budget changes must call this
   * before creating resources again so invalid backend handles cannot be reused.
   */
  invalidate(reason: string = 'cache invalidated'): void {
    this._cacheGeneration++;
    this._invalidateCache(this._renderPipelineCache);
    this._invalidateCache(this._computePipelineCache);
    for (const key of Object.keys(this._pendingRenderPipelineCache)) {
      delete this._pendingRenderPipelineCache[key];
    }
    for (const key of Object.keys(this._pendingComputePipelineCache)) {
      delete this._pendingComputePipelineCache[key];
    }
    for (const sharedPipeline of Object.values(this._sharedRenderPipelineCache)) {
      sharedPipeline.resource.destroy();
    }
    for (const key of Object.keys(this._sharedRenderPipelineCache)) {
      delete this._sharedRenderPipelineCache[key];
    }
    incrementFactoryCounter(this.device, 'pipelineCacheInvalidations');
    if (this.device.props.debugFactories) {
      log.log(1, `${this}: invalidated caches (${reason})`)();
    }
  }

  /**
   * WebGL has two cache layers with different priorities:
   * - `_sharedRenderPipelineCache` owns `WEBGLSharedRenderPipeline` / `WebGLProgram` reuse.
   * - `_renderPipelineCache` owns `RenderPipeline` wrapper reuse.
   *
   * Shared WebGL program reuse is the hard requirement. Wrapper reuse is beneficial,
   * but wrapper cache misses are acceptable if that keeps the cache logic simple and
   * prevents incorrect cache hits.
   *
   * In particular, wrapper hash logic must never force program creation or linked-program
   * introspection just to decide whether a shared WebGL program can be reused.
   */
  /** Return a RenderPipeline matching supplied props. Reuses an equivalent pipeline if already created. */
  createRenderPipeline(props: RenderPipelineProps): RenderPipeline {
    if (!this.device.props._cachePipelines) {
      return this.device.createRenderPipeline(props);
    }

    const allProps: Required<RenderPipelineProps> = {...RenderPipeline.defaultProps, ...props};

    const cache = this._renderPipelineCache;
    const hash = this._hashRenderPipeline(allProps);

    let cacheItem: CacheItem<RenderPipeline> | undefined = cache[hash];
    if (cacheItem && !this._isCacheItemUsable(cacheItem.resource)) {
      this._removeCacheItem(cache, hash, true);
      cacheItem = undefined;
    }

    let pipeline: RenderPipeline;
    if (!cacheItem) {
      let sharedRenderPipeline: SharedRenderPipeline | undefined;
      try {
        sharedRenderPipeline =
          this.device.type === 'webgl' && this.device.props._sharePipelines
            ? this.createSharedRenderPipeline(allProps)
            : undefined;
        pipeline = this.device.createRenderPipeline({
          ...allProps,
          id: allProps.id ? `${allProps.id}-cached` : uid('unnamed-cached'),
          _sharedRenderPipeline: sharedRenderPipeline
        });
      } catch (error) {
        if (sharedRenderPipeline) {
          this._releaseSharedRenderPipelineResource(sharedRenderPipeline);
        }
        incrementFactoryCounter(this.device, 'pipelineCompilationErrors');
        throw error;
      }
      pipeline.hash = hash;
      cacheItem = {resource: pipeline, useCount: 1, lastUsed: this._nextUse()};
      cache[hash] = cacheItem;
      incrementFactoryCounter(this.device, 'pipelineCacheMisses');
      this._enforceCacheLimit(cache);
      if (this.device.props.debugFactories) {
        log.log(3, `${this}: ${pipeline} created, count=${cacheItem.useCount}`)();
      }
    } else {
      cacheItem.useCount++;
      cacheItem.lastUsed = this._nextUse();
      incrementFactoryCounter(this.device, 'pipelineCacheHits');
      if (this.device.props.debugFactories) {
        log.log(
          3,
          `${this}: ${cacheItem.resource} reused, count=${cacheItem.useCount}, (id=${props.id})`
        )();
      }
    }

    return cacheItem.resource;
  }

  /**
   * Asynchronously returns a render pipeline matching the supplied props.
   *
   * Concurrent requests for an equivalent pipeline share one pending backend compilation.
   */
  async createRenderPipelineAsync(props: RenderPipelineProps): Promise<RenderPipeline> {
    if (!this.device.props._cachePipelines) {
      return await this.device.createRenderPipelineAsync(props);
    }

    const allProps: Required<RenderPipelineProps> = {...RenderPipeline.defaultProps, ...props};
    const hash = this._hashRenderPipeline(allProps);
    return await this._createPipelineAsync(
      this._renderPipelineCache,
      this._pendingRenderPipelineCache,
      hash,
      () => this._createRenderPipelineAsyncInternal(allProps)
    );
  }

  /** Return a ComputePipeline matching supplied props. Reuses an equivalent pipeline if already created. */
  createComputePipeline(props: ComputePipelineProps): ComputePipeline {
    if (!this.device.props._cachePipelines) {
      return this.device.createComputePipeline(props);
    }

    const allProps: Required<ComputePipelineProps> = {...ComputePipeline.defaultProps, ...props};

    const cache = this._computePipelineCache;
    const hash = this._hashComputePipeline(allProps);

    let cacheItem: CacheItem<ComputePipeline> | undefined = cache[hash];
    if (cacheItem && !this._isCacheItemUsable(cacheItem.resource)) {
      this._removeCacheItem(cache, hash, true);
      cacheItem = undefined;
    }

    if (!cacheItem) {
      let pipeline: ComputePipeline;
      try {
        pipeline = this.device.createComputePipeline({
          ...allProps,
          id: allProps.id ? `${allProps.id}-cached` : undefined
        });
      } catch (error) {
        incrementFactoryCounter(this.device, 'pipelineCompilationErrors');
        throw error;
      }
      pipeline.hash = hash;
      cacheItem = {resource: pipeline, useCount: 1, lastUsed: this._nextUse()};
      cache[hash] = cacheItem;
      incrementFactoryCounter(this.device, 'pipelineCacheMisses');
      this._enforceCacheLimit(cache);
      if (this.device.props.debugFactories) {
        log.log(3, `${this}: ${pipeline} created, count=${cacheItem.useCount}`)();
      }
    } else {
      cacheItem.useCount++;
      cacheItem.lastUsed = this._nextUse();
      incrementFactoryCounter(this.device, 'pipelineCacheHits');
      if (this.device.props.debugFactories) {
        log.log(
          3,
          `${this}: ${cacheItem.resource} reused, count=${cacheItem.useCount}, (id=${props.id})`
        )();
      }
    }

    return cacheItem.resource;
  }

  /**
   * Asynchronously returns a compute pipeline matching the supplied props.
   *
   * Concurrent requests for an equivalent pipeline share one pending backend compilation.
   */
  async createComputePipelineAsync(props: ComputePipelineProps): Promise<ComputePipeline> {
    if (!this.device.props._cachePipelines) {
      return await this.device.createComputePipelineAsync(props);
    }

    const allProps: Required<ComputePipelineProps> = {...ComputePipeline.defaultProps, ...props};
    const hash = this._hashComputePipeline(allProps);
    return await this._createPipelineAsync(
      this._computePipelineCache,
      this._pendingComputePipelineCache,
      hash,
      () =>
        this.device.createComputePipelineAsync({
          ...allProps,
          id: allProps.id ? `${allProps.id}-cached` : undefined
        })
    );
  }

  release(pipeline: RenderPipeline | ComputePipeline): void {
    if (!this.device.props._cachePipelines) {
      pipeline.destroy();
      return;
    }

    const cache = this._getCache(pipeline);
    const cacheItem = cache?.[pipeline.hash];
    if (!cacheItem || cacheItem.resource !== pipeline) {
      if (!pipeline.destroyed) {
        pipeline.destroy();
      }
      return;
    }

    cacheItem.useCount--;
    cacheItem.lastUsed = this._nextUse();
    if (cacheItem.useCount === 0) {
      this._destroyPipeline(pipeline);
      this._enforceCacheLimitForPipeline(pipeline);
      if (this.device.props.debugFactories) {
        log.log(3, `${this}: ${pipeline} released and destroyed`)();
      }
    } else if (cacheItem.useCount < 0) {
      log.error(`${this}: ${pipeline} released, useCount < 0, resetting`)();
      cacheItem.useCount = 0;
      this._destroyPipeline(pipeline);
      this._enforceCacheLimitForPipeline(pipeline);
    } else if (this.device.props.debugFactories) {
      log.log(3, `${this}: ${pipeline} released, count=${cacheItem.useCount}`)();
    }
  }

  createSharedRenderPipeline(props: RenderPipelineProps): SharedRenderPipeline {
    const sharedPipelineHash = this._hashSharedRenderPipeline(props);
    let sharedCacheItem = this._sharedRenderPipelineCache[sharedPipelineHash];
    if (!sharedCacheItem) {
      const sharedRenderPipeline = this.device._createSharedRenderPipelineWebGL(props);
      sharedCacheItem = {resource: sharedRenderPipeline, useCount: 0, lastUsed: this._nextUse()};
      this._sharedRenderPipelineCache[sharedPipelineHash] = sharedCacheItem;
    }
    sharedCacheItem.useCount++;
    sharedCacheItem.lastUsed = this._nextUse();
    return sharedCacheItem.resource;
  }

  releaseSharedRenderPipeline(pipeline: RenderPipeline): void {
    if (!pipeline.sharedRenderPipeline) {
      return;
    }

    const sharedPipelineHash = this._hashSharedRenderPipeline(pipeline.sharedRenderPipeline.props);
    this._releaseSharedRenderPipelineAtKey(sharedPipelineHash);
  }

  // PRIVATE

  private async _createRenderPipelineAsyncInternal(
    props: Required<RenderPipelineProps>
  ): Promise<RenderPipeline> {
    let sharedRenderPipeline: SharedRenderPipeline | undefined;
    try {
      sharedRenderPipeline =
        this.device.type === 'webgl' && this.device.props._sharePipelines
          ? this.createSharedRenderPipeline(props)
          : undefined;
      return await this.device.createRenderPipelineAsync({
        ...props,
        id: props.id ? `${props.id}-cached` : uid('unnamed-cached'),
        _sharedRenderPipeline: sharedRenderPipeline
      });
    } catch (error) {
      if (sharedRenderPipeline) {
        this._releaseSharedRenderPipelineResource(sharedRenderPipeline);
      }
      throw error;
    }
  }

  private _nextUse(): number {
    this._useCounter++;
    return this._useCounter;
  }

  private _isCacheItemUsable(pipeline: RenderPipeline | ComputePipeline): boolean {
    if (pipeline.destroyed) {
      return false;
    }
    if (pipeline instanceof RenderPipeline && pipeline.isErrored) {
      return false;
    }
    return true;
  }

  private _removeCacheItem<ResourceT extends RenderPipeline | ComputePipeline>(
    cache: Record<string, CacheItem<ResourceT>>,
    hash: string,
    destroy: boolean,
    counter: 'pipelineCacheEvictions' | 'pipelineCacheInvalidations' = 'pipelineCacheEvictions'
  ): boolean {
    const cacheItem = cache[hash];
    if (!cacheItem) {
      return false;
    }
    delete cache[hash];
    if (destroy) {
      cacheItem.resource.destroy();
      if (cacheItem.resource instanceof RenderPipeline) {
        this.releaseSharedRenderPipeline(cacheItem.resource);
      }
      incrementFactoryCounter(this.device, counter);
    }
    return true;
  }

  private _invalidateCache<ResourceT extends RenderPipeline | ComputePipeline>(
    cache: Record<string, CacheItem<ResourceT>>
  ): void {
    for (const key of Object.keys(cache)) {
      this._removeCacheItem(cache, key, true, 'pipelineCacheInvalidations');
    }
  }

  private _releaseSharedRenderPipelineResource(sharedRenderPipeline: SharedRenderPipeline): void {
    const sharedPipelineHash = this._hashSharedRenderPipeline(sharedRenderPipeline.props);
    this._releaseSharedRenderPipelineAtKey(sharedPipelineHash);
  }

  private _releaseSharedRenderPipelineAtKey(sharedPipelineHash: string): void {
    const sharedCacheItem = this._sharedRenderPipelineCache[sharedPipelineHash];
    if (!sharedCacheItem) {
      return;
    }

    sharedCacheItem.useCount--;
    sharedCacheItem.lastUsed = this._nextUse();
    if (sharedCacheItem.useCount === 0) {
      sharedCacheItem.resource.destroy();
      delete this._sharedRenderPipelineCache[sharedPipelineHash];
    }
  }

  private _enforceCacheLimit<ResourceT extends RenderPipeline | ComputePipeline>(
    cache: Record<string, CacheItem<ResourceT>>
  ): void {
    const cacheLimit = this._getPipelineCacheLimit();
    if (cacheLimit === Number.POSITIVE_INFINITY) {
      return;
    }

    while (Object.keys(cache).length > cacheLimit) {
      const unusedEntry = Object.entries(cache)
        .filter(([, cacheItem]) => cacheItem.useCount === 0)
        .sort((left, right) => left[1].lastUsed - right[1].lastUsed)[0];
      if (!unusedEntry) {
        return;
      }
      this._removeCacheItem(cache, unusedEntry[0], true);
    }
  }

  private _enforceCacheLimitForPipeline(pipeline: RenderPipeline | ComputePipeline): void {
    if (pipeline instanceof ComputePipeline) {
      this._enforceCacheLimit(this._computePipelineCache);
    } else {
      this._enforceCacheLimit(this._renderPipelineCache);
    }
  }

  private _getPipelineCacheLimit(): number {
    const cacheLimit =
      this.device.props._pipelineCacheLimit ?? this.device.props._pipelineCacheBudget;
    if (cacheLimit === undefined || cacheLimit === null || !Number.isFinite(cacheLimit)) {
      return Number.POSITIVE_INFINITY;
    }
    return Math.max(0, Math.floor(cacheLimit));
  }

  private async _createPipelineAsync<ResourceT extends RenderPipeline | ComputePipeline>(
    cache: Record<string, CacheItem<ResourceT>>,
    pendingCache: Record<string, PendingCacheItem<ResourceT>>,
    hash: string,
    create: () => Promise<ResourceT>
  ): Promise<ResourceT> {
    let cachedItem: CacheItem<ResourceT> | undefined = cache[hash];
    if (cachedItem && !this._isCacheItemUsable(cachedItem.resource)) {
      this._removeCacheItem(cache, hash, true);
      cachedItem = undefined;
    }
    if (cachedItem) {
      cachedItem.useCount++;
      cachedItem.lastUsed = this._nextUse();
      incrementFactoryCounter(this.device, 'pipelineCacheHits');
      return cachedItem.resource;
    }
    let pendingItem: PendingCacheItem<ResourceT> | undefined = pendingCache[hash];
    if (pendingItem && pendingItem.generation !== this._cacheGeneration) {
      delete pendingCache[hash];
      pendingItem = undefined;
    }
    if (pendingItem) {
      pendingItem.useCount++;
      pendingItem.lastUsed = this._nextUse();
      incrementFactoryCounter(this.device, 'pipelineCachePendingHits');
      return await pendingItem.promise;
    }

    const item: PendingCacheItem<ResourceT> = {
      promise: undefined!,
      useCount: 1,
      lastUsed: this._nextUse(),
      generation: this._cacheGeneration
    };
    item.promise = create()
      .then(pipeline => {
        if (pendingCache[hash] === item) {
          delete pendingCache[hash];
        }
        if (item.generation !== this._cacheGeneration) {
          pipeline.destroy();
          if (pipeline instanceof RenderPipeline) {
            this.releaseSharedRenderPipeline(pipeline);
          }
          incrementFactoryCounter(this.device, 'pipelineCacheInvalidations');
          throw new Error(`${this}: pipeline compilation was invalidated`);
        }
        if (!this._isCacheItemUsable(pipeline)) {
          pipeline.destroy();
          if (pipeline instanceof RenderPipeline) {
            this.releaseSharedRenderPipeline(pipeline);
          }
          throw new Error(`${this}: compiled pipeline is not usable`);
        }
        const synchronousItem = cache[hash];
        if (synchronousItem) {
          if (this._isCacheItemUsable(synchronousItem.resource)) {
            synchronousItem.useCount += item.useCount;
            synchronousItem.lastUsed = this._nextUse();
            pipeline.destroy();
            if (pipeline instanceof RenderPipeline) {
              this.releaseSharedRenderPipeline(pipeline);
            }
            return synchronousItem.resource;
          }
          this._removeCacheItem(cache, hash, true);
        }
        pipeline.hash = hash;
        cache[hash] = {
          resource: pipeline,
          useCount: item.useCount,
          lastUsed: this._nextUse()
        };
        incrementFactoryCounter(this.device, 'pipelineCacheMisses');
        this._enforceCacheLimit(cache);
        return pipeline;
      })
      .catch(error => {
        if (pendingCache[hash] === item) {
          delete pendingCache[hash];
        }
        if (item.generation === this._cacheGeneration) {
          incrementFactoryCounter(this.device, 'pipelineCompilationErrors');
        }
        throw error;
      });
    pendingCache[hash] = item;
    return await item.promise;
  }

  /** Destroy a cached pipeline, removing it from the cache if configured to do so. */
  private _destroyPipeline(pipeline: RenderPipeline | ComputePipeline): boolean {
    if (!this.device.props._destroyPipelines) {
      return false;
    }

    if (pipeline instanceof ComputePipeline) {
      return this._removeCacheItem(this._computePipelineCache, pipeline.hash, true);
    }
    return this._removeCacheItem(this._renderPipelineCache, pipeline.hash, true);
  }

  /** Get the appropriate cache for the type of pipeline */
  private _getCache(
    pipeline: RenderPipeline | ComputePipeline
  ): Record<string, CacheItem<RenderPipeline>> | Record<string, CacheItem<ComputePipeline>> | null {
    let cache:
      | Record<string, CacheItem<RenderPipeline>>
      | Record<string, CacheItem<ComputePipeline>>
      | undefined;
    if (pipeline instanceof ComputePipeline) {
      cache = this._computePipelineCache;
    }
    if (pipeline instanceof RenderPipeline) {
      cache = this._renderPipelineCache;
    }
    if (!cache) {
      return null;
    }
    return cache;
  }

  /** Calculate a hash based on all the inputs for a compute pipeline */
  private _hashComputePipeline(props: ComputePipelineProps): string {
    // This cache is device-local and separate from render pipeline caches.
    const shaderHash = this._getHash(props.shader.source);
    const shaderSignatureHash = this._getHash(
      JSON.stringify({
        stage: props.shader.stage,
        language: props.shader.props.language
      })
    );
    const configurationHash = this._getHash(
      JSON.stringify([props.shaderLayout, props.entryPoint ?? 'main', props.constants])
    );
    return `G${this._cacheGeneration}/C${this._getCapabilityHash()}/S${shaderSignatureHash}/${shaderHash}/${configurationHash}`;
  }

  /** Calculate a hash based on all the inputs for a render pipeline */
  private _hashRenderPipeline(props: RenderPipelineProps): string {
    // Backend-specific hashing requirements:
    // - WebGPU hash keys must include every immutable descriptor-shaping input that can
    //   change the created `GPURenderPipeline`, including attachment formats.
    // - WebGL hash keys only govern wrapper reuse. They must remain compatible with
    //   shared-program reuse and must not depend on linked-program introspection just
    //   to decide whether a shared `WebGLProgram` can be reused.
    //
    // General exclusions:
    // - `id`, `handle`: resource identity / caller-supplied handles, not cache shape
    // - `bindings`: mutable per-pipeline compatibility state
    // - `disableWarnings`: logging only, no rendering impact
    const vsHash = props.vs
      ? this._getHash(`${props.vs.stage}:${props.vs.props.language}:${props.vs.source}`)
      : 0;
    const fsHash = props.fs
      ? this._getHash(`${props.fs.stage}:${props.fs.props.language}:${props.fs.source}`)
      : 0;
    const varyingHash = this._getWebGLVaryingHash(props);
    const shaderLayoutHash = this._getHash(JSON.stringify(props.shaderLayout));
    const uniformBlockLayoutHash = this._getHash(JSON.stringify(props._uniformBlockLayouts));
    const bufferLayoutHash = this._getHash(JSON.stringify(props.bufferLayout));
    const pipelineCapabilityHash = this._getCapabilityHash();
    const constantsHash = this._getHash(
      JSON.stringify({vsConstants: props.vsConstants, fsConstants: props.fsConstants})
    );

    const {type} = this.device;
    switch (type) {
      case 'webgl':
        // WebGL wrappers preserve default topology and parameter semantics for direct
        // callers, even though the underlying linked program may be shared separately.
        // Future WebGL-only additions here must not turn wrapper reuse into a prerequisite
        // for shared `WebGLProgram` reuse.
        const webglParameterHash = this._getHash(JSON.stringify(props.parameters));
        return `G${this._cacheGeneration}/${type}/C${pipelineCapabilityHash}/R/${vsHash}/${fsHash}V${varyingHash}T${props.topology}P${webglParameterHash}K${constantsHash}SL${shaderLayoutHash}UBL${uniformBlockLayoutHash}BL${bufferLayoutHash}`;

      case 'webgpu':
      default:
        // On WebGPU we need to rebuild the pipeline if topology, entry points,
        // shader/layout data, parameters, bufferLayout or attachment formats change.
        // Attachment formats must stay in the key so screen and offscreen passes do not
        // accidentally alias the same cached `GPURenderPipeline`.
        const entryPointHash = this._getHash(
          JSON.stringify({
            vertexEntryPoint: props.vertexEntryPoint,
            fragmentEntryPoint: props.fragmentEntryPoint
          })
        );
        const parameterHash = this._getHash(JSON.stringify(props.parameters));
        const attachmentHash = this._getWebGPUAttachmentHash(props);
        // TODO - Can json.stringify() generate different strings for equivalent objects if order of params is different?
        // create a deepHash() to deduplicate?
        return `G${this._cacheGeneration}/${type}/C${pipelineCapabilityHash}/R/${vsHash}/${fsHash}V${varyingHash}T${props.topology}EP${entryPointHash}P${parameterHash}K${constantsHash}SL${shaderLayoutHash}BL${bufferLayoutHash}A${attachmentHash}`;
    }
  }

  // This is the only gate for shared `WebGLProgram` reuse.
  // Only include inputs that affect program linking or transform-feedback linkage.
  // Wrapper-only concerns such as topology, parameters, attachment formats and layout
  // overrides must not be added here.
  private _hashSharedRenderPipeline(props: RenderPipelineProps): string {
    const vsHash = props.vs ? this._getHash(props.vs.source) : 0;
    const fsHash = props.fs ? this._getHash(props.fs.source) : 0;
    const varyingHash = this._getWebGLVaryingHash(props);
    return `G${this._cacheGeneration}/webgl/C${this._getCapabilityHash()}/S/${vsHash}/${fsHash}V${varyingHash}`;
  }

  private _getHash(key: string): number {
    if (this._hashes[key] === undefined) {
      this._hashes[key] = this._hashCounter++;
    }
    return this._hashes[key];
  }

  private _getWebGLVaryingHash(props: RenderPipelineProps): number {
    const {varyings = [], bufferMode = null} = props;
    return this._getHash(JSON.stringify({varyings, bufferMode}));
  }

  /**
   * Capability fingerprint for one device generation.
   *
   * Factories are device-local, but including this in the key makes the separation explicit
   * and prevents a recycled factory from aliasing a pipeline after a feature level or limit
   * change (for example an attached-device upgrade).
   */
  private _getCapabilityHash(): number {
    if (this._capabilityHash !== null) {
      return this._capabilityHash;
    }
    const info = this.device.info;
    const limits = this.device.limits;
    const features = Array.from(this.device.features).sort();
    this._capabilityHash = this._getHash(
      JSON.stringify({
        type: this.device.type,
        featureLevel: info?.featureLevel ?? null,
        shadingLanguage: info?.shadingLanguage ?? null,
        shadingLanguageVersion: info?.shadingLanguageVersion ?? null,
        preferredColorFormat: this.device.preferredColorFormat,
        preferredDepthFormat: this.device.preferredDepthFormat,
        features,
        limits: {
          maxBindGroups: limits?.maxBindGroups,
          maxBindingsPerBindGroup: limits?.maxBindingsPerBindGroup,
          maxColorAttachments: limits?.maxColorAttachments,
          maxVertexAttributes: limits?.maxVertexAttributes,
          maxSamplersPerShaderStage: limits?.maxSamplersPerShaderStage,
          maxUniformBufferBindingSize: limits?.maxUniformBufferBindingSize,
          maxStorageBufferBindingSize: limits?.maxStorageBufferBindingSize
        }
      })
    );
    return this._capabilityHash;
  }

  private _getWebGPUAttachmentHash(props: RenderPipelineProps): number {
    const colorAttachmentFormats = props.colorAttachmentFormats ?? [
      this.device.preferredColorFormat
    ];
    const depthStencilAttachmentFormat =
      props.depthStencilAttachmentFormat ??
      (props.parameters?.depthWriteEnabled ? this.device.preferredDepthFormat : null);

    return this._getHash(
      JSON.stringify({
        colorAttachmentFormats,
        depthStencilAttachmentFormat
      })
    );
  }
}
