// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Sampler, SamplerProps} from '../../../index';
import type {WebGPUDevice} from '../webgpu-device';

export type WebGPUSamplerProps = SamplerProps & {
  handle?: GPUSampler;
  /** @internal Cache key retained by the native handle cache. */
  _samplerCacheKey?: string;
};

/**
 * A WebGPU sampler object
 */
export class WebGPUSampler extends Sampler {
  readonly device: WebGPUDevice;
  readonly handle: GPUSampler;
  private readonly _samplerCacheKey?: string;

  constructor(device: WebGPUDevice, props: WebGPUSamplerProps) {
    super(device, props);
    this.device = device;
    this._samplerCacheKey = props._samplerCacheKey;

    this.handle = props.handle || createWebGPUSamplerHandle(device, this.props);
    this.handle.label = this.props.id;
  }

  protected override destroyHandle(): void {
    this.device._releaseCachedSampler(this._samplerCacheKey);
    // GPUSampler does not have a destroy method
    // this.handle.destroy();
    // @ts-expect-error readonly
    this.handle = null;
  }
}

/** Creates the native sampler descriptor and handle used by cached and direct wrappers. */
export function createWebGPUSamplerHandle(device: WebGPUDevice, props: SamplerProps): GPUSampler {
  const samplerDescriptor: Partial<GPUSamplerDescriptor> = {
    ...props,
    mipmapFilter: undefined
  };

  if (props.type !== 'comparison-sampler') {
    delete samplerDescriptor.compare;
  }
  if (props.mipmapFilter && props.mipmapFilter !== 'none') {
    samplerDescriptor.mipmapFilter = props.mipmapFilter;
  }

  return device.handle.createSampler(samplerDescriptor);
}
