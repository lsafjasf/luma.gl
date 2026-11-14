// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  Buffer,
  PipelineFactory,
  Texture,
  _getDefaultBindGroupFactory,
  type Device,
  type RenderPipeline,
  type Shader,
  type SamplerProps
} from '@luma.gl/core';
import {getTestDevices, getWebGLTestDevice, getWebGPUTestDevice} from '@luma.gl/core/test-utils';

const CPU_HOTSPOT_PROFILER_MODULE = 'cpu-hotspot-profiler';

const samplerProps: SamplerProps = {
  minFilter: 'linear',
  magFilter: 'linear',
  mipmapFilter: 'linear'
};

const EMPTY_SHADER_LAYOUT = {attributes: [], bindings: []};

const GLSL_VERTEX_SOURCE = `#version 300 es
const vec2 positions[3] = vec2[3](
  vec2(-1.0, -1.0),
  vec2(3.0, -1.0),
  vec2(-1.0, 3.0)
);
void main() {
  gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
}
`;

const GLSL_FRAGMENT_SOURCE = `#version 300 es
precision highp float;
out vec4 fragmentColor;
void main() {
  fragmentColor = vec4(1.0, 0.0, 0.0, 1.0);
}
`;

const WGSL_RENDER_SOURCE = `
@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  return vec4<f32>(positions[vertexIndex], 0.0, 1.0);
}

@fragment fn fragmentMain() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`;

it('Device sampler cache shares native handles across wrappers and textures', async () => {
  const devices = [await getWebGLTestDevice(), await getWebGPUTestDevice()].filter(
    (device): device is Device => Boolean(device)
  );

  for (const device of devices) {
    const firstSampler = device.createSampler({...samplerProps, id: 'first-sampler'});
    const secondSampler = device.createSampler({...samplerProps, id: 'second-sampler'});
    const differentSampler = device.createSampler({
      ...samplerProps,
      minFilter: 'nearest',
      id: 'different-sampler'
    });

    expect(secondSampler.handle, `${device.type} reuses an identical native sampler`).toBe(
      firstSampler.handle
    );
    expect(differentSampler.handle, `${device.type} separates different sampler params`).not.toBe(
      firstSampler.handle
    );

    const firstTexture = device.createTexture({width: 1, height: 1, sampler: samplerProps});
    const secondTexture = device.createTexture({width: 1, height: 1, sampler: samplerProps});
    expect(
      secondTexture.sampler.handle,
      `${device.type} textures share one native sampler for identical props`
    ).toBe(firstTexture.sampler.handle);

    firstTexture.destroy();
    expect(
      secondTexture.sampler.destroyed,
      `${device.type} texture destruction does not invalidate another wrapper`
    ).toBe(false);

    secondTexture.destroy();
    firstSampler.destroy();
    secondSampler.destroy();
    differentSampler.destroy();
  }
});

it('Multi-pass rendering reuses one cached pipeline and preserves output', async () => {
  for (const device of await getTestDevices(['webgl', 'webgpu'])) {
    const {vertexShader, fragmentShader} = await createRenderShaders(device);
    const pipelineFactory = new PipelineFactory(device);
    const pipeline = pipelineFactory.createRenderPipeline({
      vs: vertexShader,
      fs: fragmentShader,
      shaderLayout: EMPTY_SHADER_LAYOUT,
      colorAttachmentFormats: ['rgba8unorm']
    });
    const vertexArray = device.createVertexArray({
      shaderLayout: EMPTY_SHADER_LAYOUT,
      bufferLayout: []
    });
    const textures = [0, 1].map(() =>
      device.createTexture({
        width: 1,
        height: 1,
        format: 'rgba8unorm',
        usage: Texture.RENDER | Texture.COPY_SRC
      })
    );
    const framebuffers = textures.map(texture =>
      device.createFramebuffer({colorAttachments: [texture]})
    );

    for (const framebuffer of framebuffers) {
      const renderPass = device.beginRenderPass({framebuffer, clearColor: [0, 0, 0, 1]});
      renderPass.setPipeline(pipeline);
      renderPass.setBindings({});
      renderPass.setVertexArray(vertexArray);
      renderPass.draw({vertexCount: 3});
      renderPass.end();
    }
    device.submit();

    for (const texture of textures) {
      expect(
        await readTexturePixels(device, texture),
        `${device.type} multi-pass output matches across render passes`
      ).toEqual([255, 0, 0, 255]);
    }

    for (const framebuffer of framebuffers) {
      framebuffer.destroy();
    }
    for (const texture of textures) {
      texture.destroy();
    }
    vertexArray.destroy();
    pipelineFactory.release(pipeline);
    for (const shader of new Set([vertexShader, fragmentShader])) {
      shader.destroy();
    }
  }
});

it('BindGroupFactory rebuilds a bind group when its resource changes or is destroyed', async () => {
  const webgpuDevice = await getWebGPUTestDevice();
  if (!webgpuDevice) {
    return;
  }

  const shader = webgpuDevice.createShader({
    source: `
      @group(0) @binding(0) var<storage, read_write> data: array<u32>;
      @compute @workgroup_size(1) fn main() {
        data[0] = data[0] + 1u;
      }
    `
  });
  const pipeline = webgpuDevice.createComputePipeline({
    shader,
    shaderLayout: {
      bindings: [{name: 'data', type: 'storage', group: 0, location: 0}]
    }
  });
  const firstBuffer = webgpuDevice.createBuffer({
    byteLength: 4,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const secondBuffer = webgpuDevice.createBuffer({
    byteLength: 4,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const bindGroupFactory = _getDefaultBindGroupFactory(webgpuDevice);
  const cacheKeys = {0: {}};

  const firstBindGroup = bindGroupFactory.getBindGroups(
    pipeline,
    {data: firstBuffer},
    cacheKeys
  )[0];
  expect(
    bindGroupFactory.getBindGroups(pipeline, {data: firstBuffer}, cacheKeys)[0],
    'unchanged live bindings reuse the cached bind group'
  ).toBe(firstBindGroup);

  const secondBindGroup = bindGroupFactory.getBindGroups(
    pipeline,
    {data: secondBuffer},
    cacheKeys
  )[0];
  expect(secondBindGroup, 'a new binding identity invalidates the cached bind group').not.toBe(
    firstBindGroup
  );

  firstBuffer.destroy();
  secondBuffer.destroy();
  pipeline.destroy();
  shader.destroy();
});

it('PipelineFactory evicts unused pipelines over budget and records counters', async () => {
  const webglDevice = await getWebGLTestDevice();
  if (!webglDevice?.props._cachePipelines) {
    return;
  }

  const previousLimit = webglDevice.props._pipelineCacheLimit;
  webglDevice.props._pipelineCacheLimit = 1;
  const profiler = getProfiler(webglDevice);
  const initialHits = profiler.pipelineCacheHits || 0;
  const initialEvictions = profiler.pipelineCacheEvictions || 0;
  const factory = new PipelineFactory(webglDevice);
  const vertexShader = webglDevice.createShader({
    stage: 'vertex',
    source: `
      in vec4 positions;
      void main() {
        gl_Position = positions;
      }
    `
  });
  const fragmentShader = webglDevice.createShader({
    stage: 'fragment',
    source: `
      precision highp float;
      out vec4 fragColor;
      void main() {
        fragColor = vec4(1.0);
      }
    `
  });

  try {
    const pipeline1 = factory.createRenderPipeline({
      vs: vertexShader,
      fs: fragmentShader,
      parameters: {cullMode: 'back'}
    });
    const repeatedPipeline = factory.createRenderPipeline({
      vs: vertexShader,
      fs: fragmentShader,
      parameters: {cullMode: 'back'}
    });
    const pipeline2 = factory.createRenderPipeline({
      vs: vertexShader,
      fs: fragmentShader,
      parameters: {cullMode: 'front'}
    });

    expect(repeatedPipeline).toBe(pipeline1);
    expect(profiler.pipelineCacheHits || 0).toBeGreaterThan(initialHits);

    factory.release(pipeline1);
    factory.release(repeatedPipeline);
    factory.release(pipeline2);

    expect(factory.cacheSize.renderPipelines).toBeLessThanOrEqual(1);
    expect(profiler.pipelineCacheEvictions || 0).toBeGreaterThan(initialEvictions);
    expect(
      [pipeline1, pipeline2].some(pipeline => pipeline.destroyed),
      'at least one unused pipeline is destroyed to satisfy the budget'
    ).toBe(true);
  } finally {
    webglDevice.props._pipelineCacheLimit = previousLimit;
    vertexShader.destroy();
    fragmentShader.destroy();
  }
});

it('PipelineFactory invalidation prevents stale pipeline reuse', async () => {
  const webglDevice = await getWebGLTestDevice();
  if (!webglDevice?.props._cachePipelines) {
    return;
  }

  const factory = new PipelineFactory(webglDevice);
  const vertexShader = webglDevice.createShader({
    stage: 'vertex',
    source: `
      in vec4 positions;
      void main() {
        gl_Position = positions;
      }
    `
  });
  const fragmentShader = webglDevice.createShader({
    stage: 'fragment',
    source: `
      precision highp float;
      out vec4 fragColor;
      void main() {
        fragColor = vec4(0.0);
      }
    `
  });
  const pipeline1: RenderPipeline = factory.createRenderPipeline({
    vs: vertexShader,
    fs: fragmentShader
  });

  try {
    const generation = factory.cacheGeneration;
    factory.invalidate('test context loss');
    expect(factory.cacheGeneration).toBe(generation + 1);
    expect(factory.cacheSize.renderPipelines).toBe(0);

    const pipeline2 = factory.createRenderPipeline({vs: vertexShader, fs: fragmentShader});
    expect(pipeline2, 'the next request builds a fresh pipeline').not.toBe(pipeline1);
    expect(pipeline1.destroyed).toBe(true);
    factory.release(pipeline2);
  } finally {
    vertexShader.destroy();
    fragmentShader.destroy();
  }
});

function getProfiler(device: Device): {
  enabled?: boolean;
  pipelineCacheHits?: number;
  pipelineCacheEvictions?: number;
} {
  device.userData[CPU_HOTSPOT_PROFILER_MODULE] ||= {};
  const profiler = device.userData[CPU_HOTSPOT_PROFILER_MODULE] as {
    enabled?: boolean;
    pipelineCacheHits?: number;
    pipelineCacheEvictions?: number;
  };
  profiler.enabled = true;
  return profiler;
}

async function createRenderShaders(
  device: Device
): Promise<{vertexShader: Shader; fragmentShader: Shader}> {
  if (device.info.shadingLanguage === 'wgsl') {
    const shader = device.createShader({source: WGSL_RENDER_SOURCE});
    await shader.asyncCompilationStatus;
    return {vertexShader: shader, fragmentShader: shader};
  }
  const vertexShader = device.createShader({stage: 'vertex', source: GLSL_VERTEX_SOURCE});
  const fragmentShader = device.createShader({
    stage: 'fragment',
    source: GLSL_FRAGMENT_SOURCE
  });
  await Promise.all([vertexShader.asyncCompilationStatus, fragmentShader.asyncCompilationStatus]);
  return {vertexShader, fragmentShader};
}

async function readTexturePixels(device: Device, texture: Texture): Promise<number[]> {
  const layout = texture.computeMemoryLayout({width: 1, height: 1});
  const readBuffer = device.createBuffer({
    byteLength: layout.byteLength,
    usage: Buffer.COPY_DST | Buffer.MAP_READ
  });
  try {
    texture.readBuffer({width: 1, height: 1}, readBuffer);
    return Array.from((await readBuffer.readAsync()).slice(0, 4));
  } finally {
    readBuffer.destroy();
  }
}
