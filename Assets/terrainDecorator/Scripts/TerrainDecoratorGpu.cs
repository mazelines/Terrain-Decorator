#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEditor;

public static class TerrainDecoratorGpu
{
    const int ThreadGroupSize = 8;
    const int MaxMaskTextures = 32;
    const int MaxTextureLayers = 32;
    const float ParityEpsilon = 2e-2f;

    // HLSL RuleData / LayerData 와 동일 순서·크기 (StructuredBuffer stride 64 / 32)
    [StructLayout(LayoutKind.Sequential, Pack = 4, Size = 64)]
    struct RuleDataGpu
    {
        public int filter;
        public int blend;
        public int active;
        public int imageChannel;
        public int targetLayerIndex;
        public int maskTextureIndex;
        public float minVal;
        public float maxVal;
        public float frequency;
        public float lacunarity;
        public float intensity;
        public float contrast;
        public int paintedLayerIndex;
        public int pad0;
        public int pad1;
        public int pad2;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4, Size = 32)]
    struct LayerDataGpu
    {
        public int active;
        public int layerIndex;
        public int activeRuleCount;
        public int ruleStart;
        public int ruleCount;
        public int pad0;
        public int pad1;
        public int pad2;
    }

    const int RuleStrideBytes = 64;
    const int LayerStrideBytes = 32;

    static ComputeShader _compute;
    static int _kernelEvaluate;
    static int _kernelScanline;

    static bool EnsureComputeLoaded()
    {
        if (_compute != null)
            return true;

        _compute = AssetDatabase.LoadAssetAtPath<ComputeShader>(
            "Assets/terrainDecorator/Shaders/TerrainDecoratorSplat.compute");

        if (_compute == null)
            return false;

        _kernelEvaluate = _compute.FindKernel("EvaluateLayerWeights");
        _kernelScanline = _compute.FindKernel("EvaluateDecorScanline");
        if (_kernelEvaluate < 0 || _kernelScanline < 0)
        {
            Debug.LogError("TerrainDecorator GPU: compute kernels failed to compile. Check TerrainDecoratorSplat.compute in Console.");
            _compute = null;
            return false;
        }
        return true;
    }

    public static bool HasActiveLayerFilterRule(TerrainDecorator decorator)
    {
        for (int layerNo = 0; layerNo < decorator.layers.Count; layerNo++)
        {
            if (!decorator.layers[layerNo].active)
                continue;
            for (int i = 0; i < decorator.layers[layerNo].rules.Count; i++)
            {
                var rule = decorator.layers[layerNo].rules[i];
                if (rule.active && rule.filter == TerrainDecorator.FilterType.layer)
                    return true;
            }
        }
        return false;
    }

    public static bool TryDecorateSplat(TerrainDecorator decorator, out float[,,] splatMap, out float[] layerResultFlat)
    {
        splatMap = null;
        layerResultFlat = null;

        Terrain terrain = decorator.t;
        TerrainData data = terrain.terrainData;
        int alphamapWidth = data.alphamapWidth;
        int alphamapHeight = data.alphamapHeight;

        if (!decorator.useGpuRuleEvaluate)
            return TryDecorateSplatCpuEvaluate(decorator, data, alphamapWidth, alphamapHeight, out splatMap, out layerResultFlat);

        if (!SystemInfo.supportsComputeShaders || !EnsureComputeLoaded())
            return false;

        int heightmapHeight = data.heightmapResolution;
        int textureLayerCount = data.terrainLayers.Length;
        int decorLayerCount = decorator.layers.Count;

        if (textureLayerCount > MaxTextureLayers)
        {
            Debug.LogWarning($"TerrainDecorator GPU: texture layer count ({textureLayerCount}) exceeds max ({MaxTextureLayers}). Using CPU.");
            return false;
        }

        TerrainDecoratorBakeCache.TryAcquire(decorator, data, decorator.useBakeCache, out TerrainDecoratorBakeContext bake);

        int pixelCount = alphamapWidth * alphamapHeight;
        if (!BuildRulesAndLayers(decorator, bake, alphamapWidth, alphamapHeight, pixelCount,
                out List<RuleDataGpu> rulesGpu, out List<LayerDataGpu> layersGpu, out Texture2DArray maskArray))
            return false;

        if (Marshal.SizeOf<RuleDataGpu>() != RuleStrideBytes || Marshal.SizeOf<LayerDataGpu>() != LayerStrideBytes)
        {
            Debug.LogError(
                $"TerrainDecorator GPU: struct size mismatch (Rule={Marshal.SizeOf<RuleDataGpu>()}/{RuleStrideBytes}, Layer={Marshal.SizeOf<LayerDataGpu>()}/{LayerStrideBytes}).");
            return false;
        }

        ComputeBuffer ruleBuffer = new ComputeBuffer(Mathf.Max(rulesGpu.Count, 1), RuleStrideBytes);
        ComputeBuffer layerBuffer = new ComputeBuffer(Mathf.Max(layersGpu.Count, 1), LayerStrideBytes);
        UploadRuleBuffer(ruleBuffer, rulesGpu);
        UploadLayerBuffer(layerBuffer, layersGpu);

        Texture2DArray inputAlphamap = bake.PaintedAlphamapTexture
            ?? CreateDummyTexture2DArray(alphamapWidth, alphamapHeight, 1);
        maskArray = maskArray ?? CreateDummyTexture2DArray(alphamapWidth, alphamapHeight, 1);

        int layerFlatCount = Mathf.Max(decorLayerCount, 1) * pixelCount;
        ComputeBuffer layerFlatBuffer = new ComputeBuffer(layerFlatCount, sizeof(float));
        int splatScratchCount = Mathf.Max(pixelCount * textureLayerCount, 1);
        ComputeBuffer splatScratchBuffer = new ComputeBuffer(splatScratchCount, sizeof(float));

        var layerWeightsArray = new RenderTexture(alphamapWidth, alphamapHeight, 0, RenderTextureFormat.RFloat)
        {
            dimension = TextureDimension.Tex2DArray,
            volumeDepth = Mathf.Max(decorLayerCount, 1),
            enableRandomWrite = true
        };
        layerWeightsArray.Create();

        bool useScanline = HasActiveLayerFilterRule(decorator);
        bool ownsMaskArray = maskArray != null;

        try
        {
            int kernel = useScanline ? _kernelScanline : _kernelEvaluate;
            BindEvaluate(_compute, kernel, bake.HeightBuffer, bake.SlopeBuffer, bake.FalloffNoiseBuffer,
                inputAlphamap, maskArray, ruleBuffer, layerBuffer, layerWeightsArray, layerFlatBuffer,
                splatScratchBuffer, alphamapWidth, alphamapHeight, heightmapHeight, textureLayerCount,
                decorLayerCount, decorator);

            if (useScanline)
            {
                splatScratchBuffer.SetData(new float[splatScratchCount]);
                _compute.Dispatch(_kernelScanline, 1, alphamapHeight, 1);
            }
            else
            {
                int gx = Mathf.CeilToInt(alphamapWidth / (float)ThreadGroupSize);
                int gy = Mathf.CeilToInt(alphamapHeight / (float)ThreadGroupSize);
                _compute.Dispatch(_kernelEvaluate, gx, gy, Mathf.Max(decorLayerCount, 1));
            }

            layerResultFlat = new float[layerFlatCount];
            layerFlatBuffer.GetData(layerResultFlat);

            if (decorator.verifyGpuParity)
            {
                if (!VerifyLayerWeightParity(decorator, data, layerResultFlat, out float[] cpuFlat))
                {
                    Debug.LogWarning("TerrainDecorator GPU: parity failed — using CPU rule evaluate for this Decorate.");
                    layerResultFlat = cpuFlat;
                }
            }

            splatMap = new float[alphamapWidth, alphamapHeight, textureLayerCount];
            for (int y = 0; y < alphamapHeight; y++)
            {
                for (int x = 0; x < alphamapWidth; x++)
                {
                    decorator.ApplyLayerWeightsForPixel(x, y, layerResultFlat);
                    decorator.WriteSplatPixel(x, y, splatMap);
                }
            }

            return true;
        }
        finally
        {
            ruleBuffer.Release();
            layerBuffer.Release();
            layerFlatBuffer.Release();
            splatScratchBuffer.Release();
            layerWeightsArray.Release();
            if (ownsMaskArray)
                UnityEngine.Object.DestroyImmediate(maskArray);
        }
    }

    static bool VerifyLayerWeightParity(
        TerrainDecorator decorator,
        TerrainData data,
        float[] gpuFlat,
        out float[] cpuFlat)
    {
        int w = data.alphamapWidth;
        int h = data.alphamapHeight;
        int textureLayerCount = data.terrainLayers.Length;
        float[,,] refMap = new float[w, h, textureLayerCount];
        decorator.DecorateSplatCpu(refMap, out cpuFlat);

        float maxDiff = 0f;
        int maxIdx = -1;
        int pixelCount = w * h;
        int decorLayerCount = decorator.layers.Count;
        int n = Mathf.Min(gpuFlat.Length, cpuFlat.Length);
        for (int i = 0; i < n; i++)
        {
            int layerNo = i / pixelCount;
            if (layerNo >= decorLayerCount || !decorator.layers[layerNo].active)
                continue;

            float d = Mathf.Abs(gpuFlat[i] - cpuFlat[i]);
            if (d > maxDiff)
            {
                maxDiff = d;
                maxIdx = i;
            }
        }

        if (maxDiff > ParityEpsilon)
        {
            LogFirstParityMismatch(gpuFlat, cpuFlat, w, h, decorLayerCount, decorator);
            if (maxIdx >= 0)
            {
                int layerNo = maxIdx / pixelCount;
                int rem = maxIdx % pixelCount;
                int mx = rem % w;
                int my = rem / w;
                Debug.LogWarning(
                    $"TerrainDecorator GPU parity worst: layer={layerNo} ({mx},{my}) cpu={cpuFlat[maxIdx]:F4} gpu={gpuFlat[maxIdx]:F4} diff={maxDiff:F4}");
            }
            Debug.LogWarning($"TerrainDecorator GPU parity: max layer-weight diff vs CPU = {maxDiff:E3}.");
            return false;
        }

        Debug.Log($"TerrainDecorator GPU parity: OK (max diff {maxDiff:E3}).");
        return true;
    }

    static bool BuildRulesAndLayers(
        TerrainDecorator decorator,
        TerrainDecoratorBakeContext bake,
        int alphamapWidth,
        int alphamapHeight,
        int pixelCount,
        out List<RuleDataGpu> rulesGpu,
        out List<LayerDataGpu> layersGpu,
        out Texture2DArray maskArray)
    {
        rulesGpu = new List<RuleDataGpu>();
        layersGpu = new List<LayerDataGpu>();
        maskArray = null;

        var maskTextures = new List<Texture2D>();
        var noiseMaskIndex = new Dictionary<string, int>();
        var textureMaskIndex = new Dictionary<string, int>();
        int decorLayerCount = decorator.layers.Count;

        for (int layerNo = 0; layerNo < decorLayerCount; layerNo++)
        {
            var layer = decorator.layers[layerNo];
            int ruleStart = rulesGpu.Count;
            int activeRuleCount = 0;

            for (int i = 0; i < layer.rules.Count; i++)
            {
                var rule = layer.rules[i];
                if (rule.active)
                    activeRuleCount++;

                int maskIndex = -1;
                if (rule.filter == TerrainDecorator.FilterType.texture && rule.active && rule.map != null)
                {
                    string texKey = TerrainDecoratorBakeContext.TextureMaskKey(layerNo, i);
                    if (!textureMaskIndex.TryGetValue(texKey, out maskIndex))
                    {
                        if (maskTextures.Count >= MaxMaskTextures)
                        {
                            Debug.LogWarning("TerrainDecorator GPU: too many mask textures. Using CPU.");
                            CleanupMaskTextures(maskTextures);
                            return false;
                        }
                        if (!bake.TextureMaskMaps.TryGetValue(texKey, out float[] maskFlat))
                        {
                            maskFlat = new float[pixelCount];
                            for (int y = 0; y < alphamapHeight; y++)
                            for (int x = 0; x < alphamapWidth; x++)
                                maskFlat[y * alphamapWidth + x] = TerrainDecoratorSampling.SampleTextureMask(
                                    rule.map, x, y, alphamapWidth, rule.imageChannel);
                        }
                        maskIndex = AddMaskFromFloatMap(maskTextures, maskFlat, alphamapWidth, alphamapHeight);
                        textureMaskIndex[texKey] = maskIndex;
                    }
                }
                else if (rule.filter == TerrainDecorator.FilterType.noise && rule.active)
                {
                    string noiseKey = TerrainDecoratorBakeContext.NoiseKey(rule.frequency, rule.lacunarity);
                    if (!noiseMaskIndex.TryGetValue(noiseKey, out maskIndex))
                    {
                        if (maskTextures.Count >= MaxMaskTextures)
                        {
                            Debug.LogWarning("TerrainDecorator GPU: too many mask textures. Using CPU.");
                            CleanupMaskTextures(maskTextures);
                            return false;
                        }
                        if (!bake.NoiseMaps.TryGetValue(noiseKey, out float[] noiseFlat))
                        {
                            noiseFlat = new float[pixelCount];
                            TerrainDecoratorSampling.BakeNoiseMap(decorator, alphamapWidth, alphamapHeight,
                                rule.frequency, rule.lacunarity, noiseFlat);
                        }
                        maskIndex = AddMaskFromFloatMap(maskTextures, noiseFlat, alphamapWidth, alphamapHeight);
                        noiseMaskIndex[noiseKey] = maskIndex;
                    }
                }

                rulesGpu.Add(new RuleDataGpu
                {
                    filter = (int)rule.filter,
                    blend = (int)rule.blend,
                    active = rule.active ? 1 : 0,
                    imageChannel = (int)rule.imageChannel,
                    targetLayerIndex = rule.targetLayerIndex,
                    maskTextureIndex = maskIndex,
                    minVal = rule.min,
                    maxVal = rule.max,
                    frequency = rule.frequency,
                    lacunarity = rule.lacunarity,
                    intensity = rule.intensity,
                    contrast = rule.contrast,
                    paintedLayerIndex = layer.layerIndex
                });
            }

            layersGpu.Add(new LayerDataGpu
            {
                active = layer.active ? 1 : 0,
                layerIndex = layer.layerIndex,
                activeRuleCount = activeRuleCount,
                ruleStart = ruleStart,
                ruleCount = layer.rules.Count
            });
        }

        if (maskTextures.Count > 0)
        {
            maskArray = BuildMaskTextureArray(maskTextures, alphamapWidth, alphamapHeight);
            CleanupMaskTextures(maskTextures);
        }

        return true;
    }

    static void UploadRuleBuffer(ComputeBuffer buffer, List<RuleDataGpu> rules)
    {
        if (rules.Count == 0)
            return;

        const int intsPerRule = RuleStrideBytes / sizeof(int);
        var packed = new int[rules.Count * intsPerRule];
        for (int i = 0; i < rules.Count; i++)
        {
            var r = rules[i];
            int o = i * intsPerRule;
            packed[o + 0] = r.filter;
            packed[o + 1] = r.blend;
            packed[o + 2] = r.active;
            packed[o + 3] = r.imageChannel;
            packed[o + 4] = r.targetLayerIndex;
            packed[o + 5] = r.maskTextureIndex;
            WriteFloatAsInt(packed, o + 6, r.minVal);
            WriteFloatAsInt(packed, o + 7, r.maxVal);
            WriteFloatAsInt(packed, o + 8, r.frequency);
            WriteFloatAsInt(packed, o + 9, r.lacunarity);
            WriteFloatAsInt(packed, o + 10, r.intensity);
            WriteFloatAsInt(packed, o + 11, r.contrast);
            packed[o + 12] = r.paintedLayerIndex;
            packed[o + 13] = r.pad0;
            packed[o + 14] = r.pad1;
            packed[o + 15] = r.pad2;
        }
        buffer.SetData(packed);
    }

    static void UploadLayerBuffer(ComputeBuffer buffer, List<LayerDataGpu> layers)
    {
        if (layers.Count == 0)
            return;

        const int intsPerLayer = LayerStrideBytes / sizeof(int);
        var packed = new int[layers.Count * intsPerLayer];
        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            int o = i * intsPerLayer;
            packed[o + 0] = layer.active;
            packed[o + 1] = layer.layerIndex;
            packed[o + 2] = layer.activeRuleCount;
            packed[o + 3] = layer.ruleStart;
            packed[o + 4] = layer.ruleCount;
            packed[o + 5] = layer.pad0;
            packed[o + 6] = layer.pad1;
            packed[o + 7] = layer.pad2;
        }
        buffer.SetData(packed);
    }

    static void WriteFloatAsInt(int[] packed, int index, float value)
    {
        packed[index] = BitConverter.SingleToInt32Bits(value);
    }

    static void LogFirstParityMismatch(
        float[] gpuFlat,
        float[] cpuFlat,
        int w,
        int h,
        int decorLayerCount,
        TerrainDecorator decorator)
    {
        int pixelCount = w * h;
        for (int layerNo = 0; layerNo < decorLayerCount; layerNo++)
        {
            if (!decorator.layers[layerNo].active)
                continue;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int idx = layerNo * pixelCount + y * w + x;
                    if (idx >= gpuFlat.Length || idx >= cpuFlat.Length)
                        return;
                    float d = Mathf.Abs(gpuFlat[idx] - cpuFlat[idx]);
                    if (d > ParityEpsilon)
                    {
                        Debug.LogWarning(
                            $"TerrainDecorator GPU parity sample: layer={layerNo} ({x},{y}) cpu={cpuFlat[idx]:F4} gpu={gpuFlat[idx]:F4}");
                        return;
                    }
                }
            }
        }
    }

    static void CleanupMaskTextures(List<Texture2D> masks)
    {
        for (int i = 0; i < masks.Count; i++)
            UnityEngine.Object.DestroyImmediate(masks[i]);
        masks.Clear();
    }

    static bool TryDecorateSplatCpuEvaluate(
        TerrainDecorator decorator,
        TerrainData data,
        int alphamapWidth,
        int alphamapHeight,
        out float[,,] splatMap,
        out float[] layerResultFlat)
    {
        splatMap = null;
        layerResultFlat = null;

        int textureLayerCount = data.terrainLayers.Length;
        splatMap = new float[alphamapWidth, alphamapHeight, textureLayerCount];
        decorator.DecorateSplatCpu(splatMap, out layerResultFlat);
        return true;
    }

    static int AddMaskFromColorMap(List<Texture2D> masks, Color[] map, int width, int height)
    {
        var tex = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
        var pixels = new Color[width * height];
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            pixels[y * width + x] = map[y * width + (width - x - 1)];
        tex.SetPixels(pixels);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Point;
        masks.Add(tex);
        return masks.Count - 1;
    }

    static int AddMaskFromFloatMap(List<Texture2D> masks, float[] values, int width, int height)
    {
        var tex = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
        var pixels = new Color[values.Length];
        for (int i = 0; i < values.Length; i++)
            pixels[i] = new Color(values[i], values[i], values[i], 1f);
        tex.SetPixels(pixels);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Point;
        masks.Add(tex);
        return masks.Count - 1;
    }

    static void BindEvaluate(
        ComputeShader compute,
        int kernel,
        ComputeBuffer heightBuffer,
        ComputeBuffer slopeBuffer,
        ComputeBuffer falloffBuffer,
        Texture2DArray inputAlphamap,
        Texture2DArray maskArray,
        ComputeBuffer ruleBuffer,
        ComputeBuffer layerBuffer,
        RenderTexture layerWeightsArray,
        ComputeBuffer layerFlatBuffer,
        ComputeBuffer splatScratchBuffer,
        int alphamapWidth,
        int alphamapHeight,
        int heightmapHeight,
        int textureLayerCount,
        int decorLayerCount,
        TerrainDecorator decorator)
    {
        compute.SetBuffer(kernel, "_HeightWorldFlat", heightBuffer);
        compute.SetBuffer(kernel, "_SlopeFlat", slopeBuffer);
        compute.SetBuffer(kernel, "_FalloffNoiseFlat", falloffBuffer);
        compute.SetTexture(kernel, "_InputAlphamap", inputAlphamap);
        compute.SetTexture(kernel, "_MaskTextures", maskArray);
        compute.SetBuffer(kernel, "_Rules", ruleBuffer);
        compute.SetBuffer(kernel, "_Layers", layerBuffer);
        compute.SetTexture(kernel, "_LayerResultWeights", layerWeightsArray);
        compute.SetBuffer(kernel, "_LayerResultFlat", layerFlatBuffer);
        compute.SetBuffer(kernel, "_SplatScratch", splatScratchBuffer);
        compute.SetInt("_AlphamapWidth", alphamapWidth);
        compute.SetInt("_AlphamapHeight", alphamapHeight);
        compute.SetInt("_TextureLayerCount", textureLayerCount);
        compute.SetInt("_DecorLayerCount", decorLayerCount);
        compute.SetInt("_FalloffNoise", decorator.fallOfNoise ? 1 : 0);
        compute.SetFloat("_FalloffNoiseAmplitude", decorator.fallofNoiseAmplitude);
        compute.SetFloat("_HeightFalloff", heightmapHeight / 512f * decorator.fallOffDistance);
        compute.SetFloat("_SlopeFalloff", heightmapHeight / 256f);
    }

    static Texture2DArray BuildMaskTextureArray(List<Texture2D> masks, int targetWidth, int targetHeight)
    {
        if (masks.Count == 0)
            return null;

        var array = new Texture2DArray(targetWidth, targetHeight, masks.Count, TextureFormat.RGBA32, false, true);
        for (int i = 0; i < masks.Count; i++)
            array.SetPixels(masks[i].GetPixels(), i);
        array.Apply(false, false);
        array.wrapMode = TextureWrapMode.Clamp;
        array.filterMode = FilterMode.Point;
        return array;
    }

    static Texture2DArray CreateDummyTexture2DArray(int width, int height, int depth)
    {
        var array = new Texture2DArray(Mathf.Max(width, 1), Mathf.Max(height, 1), depth, TextureFormat.RGBA32, false, true);
        var pixel = new Color[] { Color.black };
        for (int i = 0; i < depth; i++)
            array.SetPixels(pixel, i);
        array.Apply(false, false);
        return array;
    }
}
#endif
