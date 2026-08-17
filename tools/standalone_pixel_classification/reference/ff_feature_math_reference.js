/**
 * fastfilters3
 * Tony -- 20260602
 * 
 * A WebGPU-accelerated port of the foundational filters from fastfilters2/Vigra.
 * Designed for sub-100ms interactive pixel classification in the browser.
 */

export const fastfilters3 = {
  /**
   * Official ilastik feature identifiers in the order they are concatenated
   * in the output buffer for a single scale.
   */
  FEATURE_IDS: [
    "GaussianSmoothing",
    "LaplacianOfGaussian",
    "GaussianGradientMagnitude",
    "DifferenceOfGaussians",
    "StructureTensorEigenvalues",   // 2 channels (Largest, Smallest)
    "HessianOfGaussianEigenvalues"  // 2 channels (Largest, Smallest)
  ],

  /**
   * Generates a 1D Gaussian kernel or its analytical derivatives.
   * Replicates Vigra's dynamic radius and strict normalization requirements.
   * 
   * @param {number} scale - Sigma value (e.g. 0.3, 1.0, 3.5).
   * @param {number} order - 0 (Smoothing), 1 (1st Derivative), or 2 (2nd Derivative).
   * @returns {Float32Array} The right-half of the symmetric/anti-symmetric kernel [0...radius].
   */
  gaussian_kernel(scale, order = 0) {
    if (scale <= 0) throw new Error("scale should be greater than 0");

    // Vigra dynamic radius: ensures energy isn't truncated at high orders/scales
    const radius = Math.ceil((3.0 + 0.5 * order) * scale);
    const kernel = new Float32Array(radius + 1);
    const twoSigmaSq = 2.0 * scale * scale;
    const fullSize = 2 * radius + 1;
    const fullKernel = new Float32Array(fullSize);

    // 1. Calculate raw analytical values
    let sum = 0;
    for (let i = -radius; i <= radius; i++) {
      const x = i;
      let val = Math.exp(-(x * x) / twoSigmaSq);
      if (order === 1) val = (-x / (scale * scale)) * val;
      else if (order === 2) val = (((x * x) / (scale * scale * scale * scale)) - (1.0 / (scale * scale))) * val;
      fullKernel[i + radius] = val;
      if (order === 0) sum += val;
    }

    // 2. Strict Normalization (required for mathematical identicality with Vigra)
    if (order === 0) {
      // Smoothing sums to 1.0
      for (let i = 0; i < fullSize; i++) fullKernel[i] /= sum;
    }
    else if (order === 1) {
      // 1st Derivative: 1st moment must be exactly 1.0 (forces correlation behavior)
      let sumX = 0;
      for (let i = -radius; i <= radius; i++) sumX += i * fullKernel[i + radius];
      for (let i = 0; i < fullSize; i++) fullKernel[i] /= sumX;
    }
    else if (order === 2) {
      // 2nd Derivative: strict zero-mean and variance normalization
      let sum0 = 0;
      for (let i = -radius; i <= radius; i++) sum0 += fullKernel[i + radius];
      const mean = sum0 / fullSize;
      for (let i = 0; i < fullSize; i++) fullKernel[i] -= mean;
      let sumX2 = 0;
      for (let i = -radius; i <= radius; i++) sumX2 += 0.5 * i * i * fullKernel[i + radius];
      for (let i = 0; i < fullSize; i++) fullKernel[i] /= sumX2;
    }

    // Return the half-kernel to save GPU uniform space
    for (let i = 0; i <= radius; i++) kernel[i] = fullKernel[radius + i];
    return kernel;
  },

  /**
   * Executes the full feature extraction pipeline entirely on the GPU.
   * Uses a multi-pass approach (Horizontal 1D -> Vertical 1D) to compute 
   * 8 feature channels in parallel.
   * 
   * @param {GPUDevice} device - The active WebGPU device.
   * @param {Float32Array | GPUBuffer} data - Raw image intensity data.
   * @param {number} width - Image width.
   * @param {number} height - Image height.
   * @param {number} scale - Primary sigma for the Gaussian kernels.
   * @returns {Promise<GPUBuffer>} Storage buffer containing interleaved features [8, height, width].
   */
  async extract_features(device, data, width, height, scale) {
    const NUM_CHANNELS = 8;
    const outSize = NUM_CHANNELS * width * height;

    const k0 = this.gaussian_kernel(scale, 0);
    const k1 = this.gaussian_kernel(scale, 1);
    const k2 = this.gaussian_kernel(scale, 2);
    const k0sub = this.gaussian_kernel(scale * 0.66, 0); // For Difference of Gaussians

    // --- GPU Resource Allocation ---
    const maxRadius = 32;
    const kernelBuffer = device.createBuffer({
      size: (maxRadius * 4 * 4) + 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const kernelData = new Float32Array(maxRadius * 4 + 4);
    kernelData.set(k0, 0);
    kernelData.set(k1, maxRadius);
    kernelData.set(k2, maxRadius * 2);
    kernelData.set(k0sub, maxRadius * 3);
    const radiiView = new Int32Array(kernelData.buffer, maxRadius * 4 * 4);
    radiiView[0] = k0.length - 1;
    radiiView[1] = k1.length - 1;
    radiiView[2] = k2.length - 1;
    radiiView[3] = k0sub.length - 1;
    device.queue.writeBuffer(kernelBuffer, 0, kernelData);

    let gpuInput;
    if (data instanceof GPUBuffer) {
        gpuInput = data;
    } else {
        gpuInput = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(gpuInput, 0, data);
    }

    const gpuHoriz = device.createBuffer({ size: width * height * 4 * 4, usage: GPUBufferUsage.STORAGE });
    const gpuOutput = device.createBuffer({ size: outSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    // --- Pipeline Setup ---
    const commonWGSL = `
      struct Kernels {
        k0: array<vec4<f32>, 8>,
        k1: array<vec4<f32>, 8>,
        k2: array<vec4<f32>, 8>,
        k0sub: array<vec4<f32>, 8>,
        r0: i32, r1: i32, r2: i32, r0sub: i32,
      }
      @group(0) @binding(2) var<uniform> kernels: Kernels;
      fn get_k(k_idx: u32, i: i32) -> f32 {
        if (k_idx == 0u) { return kernels.k0[u32(i)/4u][u32(i)%4u]; }
        if (k_idx == 1u) { return kernels.k1[u32(i)/4u][u32(i)%4u]; }
        if (k_idx == 2u) { return kernels.k2[u32(i)/4u][u32(i)%4u]; }
        return kernels.k0sub[u32(i)/4u][u32(i)%4u];
      }
    `;

    const horizShader = `
      ${commonWGSL}
      @group(0) @binding(0) var<storage, read> input_data: array<f32>;
      @group(0) @binding(1) var<storage, read_write> horiz_data: array<f32>;

      fn get_val(ix: i32, iy: i32) -> f32 {
        return input_data[u32(clamp(iy, 0, ${height - 1})) * ${width}u + u32(clamp(ix, 0, ${width - 1}))];
      }

      @compute @workgroup_size(16, 16)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let x = i32(id.x); let y = i32(id.y);
        if (x >= ${width} || y >= ${height}) { return; }

        var h0 = get_k(0u, 0) * get_val(x, y);
        for (var i = 1; i <= kernels.r0; i++) { h0 += get_k(0u, i) * (get_val(x - i, y) + get_val(x + i, y)); }

        var h1 = 0.0;
        for (var i = 1; i <= kernels.r1; i++) { h1 += get_k(1u, i) * (get_val(x + i, y) - get_val(x - i, y)); }

        var h2 = get_k(2u, 0) * get_val(x, y);
        for (var i = 1; i <= kernels.r2; i++) { h2 += get_k(2u, i) * (get_val(x - i, y) + get_val(x + i, y)); }

        var h0s = get_k(3u, 0) * get_val(x, y);
        for (var i = 1; i <= kernels.r0sub; i++) { h0s += get_k(3u, i) * (get_val(x - i, y) + get_val(x + i, y)); }

        let out_idx = (u32(y) * ${width}u + u32(x)) * 4u;
        horiz_data[out_idx] = h0; horiz_data[out_idx + 1] = h1; horiz_data[out_idx + 2] = h2; horiz_data[out_idx + 3] = h0s;
      }
    `;

    const vertShader = `
      ${commonWGSL}
      @group(0) @binding(0) var<storage, read> horiz_data: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

      fn get_h(ix: i32, iy: i32) -> vec4<f32> {
        let idx = (u32(clamp(iy, 0, ${height - 1})) * ${width}u + u32(clamp(ix, 0, ${width - 1}))) * 4u;
        return vec4<f32>(horiz_data[idx], horiz_data[idx+1], horiz_data[idx+2], horiz_data[idx+3]);
      }

      @compute @workgroup_size(16, 16)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let x = i32(id.x); let y = i32(id.y);
        if (x >= ${width} || y >= ${height}) { return; }

        var l_vec = get_h(x, y) * get_k(0u, 0);
        for (var i = 1; i <= kernels.r0; i++) { l_vec += (get_h(x, y-i) + get_h(x, y+i)) * get_k(0u, i); }
        let L = l_vec.x; let Lx = l_vec.y; let Lxx = l_vec.z;

        var ly_vec = vec2<f32>(0.0);
        for (var i = 1; i <= kernels.r1; i++) { ly_vec += (get_h(x, y+i) - get_h(x, y-i)).xy * get_k(1u, i); }
        let Ly = ly_vec.x; let Lxy = ly_vec.y;

        var lyy = get_h(x, y).x * get_k(2u, 0);
        for (var i = 1; i <= kernels.r2; i++) { lyy += (get_h(x, y-i).x + get_h(x, y+i).x) * get_k(2u, i); }

        var lsub_vec = get_h(x, y).w * get_k(3u, 0);
        for (var i = 1; i <= kernels.r0sub; i++) { lsub_vec += (get_h(x, y-i).w + get_h(x, y+i).w) * get_k(3u, i); }

        let out_idx = (u32(y) * ${width}u + u32(x)) * 8u;
        output_data[out_idx] = L;                               // GaussianSmoothing
        output_data[out_idx + 1] = Lxx + lyy;                   // LaplacianOfGaussian
        output_data[out_idx + 2] = sqrt(Lx*Lx + Ly*Ly);         // GaussianGradientMagnitude
        output_data[out_idx + 3] = L - lsub_vec;                // DifferenceOfGaussians

        let s_a = Lx*Lx; let s_b = Lx*Ly; let s_c = Ly*Ly;
        let s_term = sqrt((s_a - s_c)*(s_a - s_c) * 0.25 + s_b*s_b);
        output_data[out_idx + 4] = (s_a + s_c) * 0.5 + s_term;  // StructureTensorEigenvalues (smallest)
        output_data[out_idx + 5] = (s_a + s_c) * 0.5 - s_term;  // StructureTensorEigenvalues (largest)
        let h_term = sqrt((Lxx - lyy)*(Lxx - lyy) * 0.25 + Lxy*Lxy);
        output_data[out_idx + 6] = (Lxx + lyy) * 0.5 + h_term;  // HessianOfGaussianEigenvalues (smallest)
        output_data[out_idx + 7] = (Lxx + lyy) * 0.5 - h_term;  // HessianOfGaussianEigenvalues (largest)
      }
    `;

    const modH = device.createShaderModule({ code: horizShader });
    const modV = device.createShaderModule({ code: vertShader });
    const pipeH = device.createComputePipeline({ layout: 'auto', compute: { module: modH, entryPoint: 'main' } });
    const pipeV = device.createComputePipeline({ layout: 'auto', compute: { module: modV, entryPoint: 'main' } });

    const bgH = device.createBindGroup({
      layout: pipeH.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: gpuInput } }, { binding: 1, resource: { buffer: gpuHoriz } }, { binding: 2, resource: { buffer: kernelBuffer } }]
    });
    const bgV = device.createBindGroup({
      layout: pipeV.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: gpuHoriz } }, { binding: 1, resource: { buffer: gpuOutput } }, { binding: 2, resource: { buffer: kernelBuffer } }]
    });

    const enc = device.createCommandEncoder();
    const p1 = enc.beginComputePass(); p1.setPipeline(pipeH); p1.setBindGroup(0, bgH); p1.dispatchWorkgroups(Math.ceil(width/16), Math.ceil(height/16)); p1.end();
    const p2 = enc.beginComputePass(); p2.setPipeline(pipeV); p2.setBindGroup(0, bgV); p2.dispatchWorkgroups(Math.ceil(width/16), Math.ceil(height/16)); p2.end();
    device.queue.submit([enc.finish()]);

    return gpuOutput;
  },

  /**
   * Convenience wrapper that pulls feature data back to system RAM.
   * Useful for CPU-side training or unit testing.
   * 
   * @param {GPUDevice} device 
   * @param {Float32Array} data 
   * @param {number} width 
   * @param {number} height 
   * @param {number} scale 
   * @returns {Promise<Float32Array>} Flat array of shape [8, height, width].
   */
  async compute_filters(device, data, width, height, scale) {
    const gpuOutput = await this.extract_features(device, data, width, height, scale);
    const outSize = 8 * width * height;
    const rb = device.createBuffer({ size: outSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(gpuOutput, 0, rb, 0, outSize * 4);
    device.queue.submit([enc.finish()]);

    await rb.mapAsync(GPUMapMode.READ);
    const res = new Float32Array(rb.getMappedRange().slice());
    rb.unmap();
    rb.destroy();
    return res;
  }
};