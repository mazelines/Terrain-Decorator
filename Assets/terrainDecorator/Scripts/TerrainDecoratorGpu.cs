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

    [StructLayout(LayoutKind.Sequential, Size = 64)]
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

    [StructLayout(LayoutKind.Sequential, Size = 32)]
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

        decorator.ProcessTextureFilters();

        int pixelCount = alphamapWidth * alphamapHeight;
        var heightFlat = new float[pixelCount];
        var slopeFlat = new float[pixelCount];
        var falloffFlat = new float[pixelCount];
        TerrainDecoratorSampling.BakeHeightMap(decorator, data, alphamapWidth, alphamapHeight, heightFlat);
        TerrainDecoratorSampling.BakeSlopeMap(data, alphamapWidth, alphamapHeight, slopeFlat);
        if (decorator.fallOfNoise)
            TerrainDecoratorSampling.BakeFalloffNoiseMap(decorator, alphamapWidth, alphamapHeight, falloffFlat);
        else
            Array.Fill(falloffFlat, 0.5f);

        var rulesGpu = new List<RuleDataGpu>();
        var layersGpu = new List<LayerDataGpu>();
        var maskTextures = new List<Texture2D>();
        var noiseMaskIndex = new Dictionary<string, int>();

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
                    if (maskTextures.Count >= MaxMaskTextures)
                    {
                        Debug.LogWarning("TerrainDecorator GPU: too many mask textures. Using CPU.");
                        return false;
                    }
                    maskIndex = AddMaskFromColorMap(maskTextures, rule.map, alphamapWidth, alphamapHeight);
                }
                else if (rule.filter == TerrainDecorator.FilterType.noise && rule.active)
                {
                    string noiseKey = rule.frequency.ToString("R") + "_" + rule.lacunarity.ToString("R");
                    if (!noiseMaskIndex.TryGetValue(noiseKey, out maskIndex))
                    {
                        if (maskTextures.Count >= MaxMaskTextures)
                        {
                            Debug.LogWarning("TerrainDecorator GPU: too many mask textures. Using CPU.");
                            return false;
                        }
                        var noiseFlat = new float[pixelCount];
                        TerrainDecoratorSampling.BakeNoiseMap(decorator, alphamapWidth, alphamapHeight, rule.frequency, rule.lacunarity, noiseFlat);
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

        const int ruleStride = 64;
        const int layerStride = 32;
        ComputeBuffer ruleBuffer = new ComputeBuffer(Mathf.Max(rulesGpu.Count, 1), ruleStride);
        ComputeBuffer layerBuffer = new ComputeBuffer(Mathf.Max(layersGpu.Count, 1), layerStride);
        if (rulesGpu.Count > 0)
            ruleBuffer.SetData(rulesGpu);
        if (layersGpu.Count > 0)
            layerBuffer.SetData(layersGpu);

        ComputeBuffer heightBuffer = new ComputeBuffer(pixelCount, sizeof(float));
        ComputeBuffer slopeBuffer = new ComputeBuffer(pixelCount, sizeof(float));
        ComputeBuffer falloffBuffer = new ComputeBuffer(pixelCount, sizeof(float));
        heightBuffer.SetData(heightFlat);
        slopeBuffer.SetData(slopeFlat);
        falloffBuffer.SetData(falloffFlat);

        Texture2DArray inputAlphamap = TerrainDecoratorSampling.BakePaintedAlphamapTexture(decorator, data, alphamapWidth, alphamapHeight)
            ?? CreateDummyTexture2DArray(alphamapWidth, alphamapHeight, 1);
        Texture2DArray maskArray = BuildMaskTextureArray(maskTextures, alphamapWidth, alphamapHeight)
            ?? CreateDummyTexture2DArray(alphamapWidth, alphamapHeight, 1);

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

        try
        {
            int kernel = useScanline ? _kernelScanline : _kernelEvaluate;
            BindEvaluate(_compute, kernel, heightBuffer, slopeBuffer, falloffBuffer, inputAlphamap, maskArray,
                ruleBuffer, layerBuffer, layerWeightsArray, layerFlatBuffer, splatScratchBuffer,
                alphamapWidth, alphamapHeight, heightmapHeight, textureLayerCount, decorLayerCount, decorator);

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

            LogParityVsCpu(decorator, data, alphamapWidth, alphamapHeight, layerResultFlat);

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
            heightBuffer.Release();
            slopeBuffer.Release();
            falloffBuffer.Release();
            layerWeightsArray.Release();
            for (int i = 0; i < maskTextures.Count; i++)
                UnityEngine.Object.DestroyImmediate(maskTextures[i]);
            UnityEngine.Object.DestroyImmediate(inputAlphamap);
            UnityEngine.Object.DestroyImmediate(maskArray);
        }
    }

    static void LogParityVsCpu(
        TerrainDecorator decorator,
        TerrainData data,
        int w,
        int h,
        float[] gpuFlat)
    {
        int decorCount = decorator.layers.Count;
        int pixelCount = w * h;
        int textureLayerCount = data.terrainLayers.Length;
        float maxDiff = 0f;
        bool useLayerFilter = HasActiveLayerFilterRule(decorator);

        if (useLayerFilter)
        {
            float[,,] cpuMap = new float[w, h, textureLayerCount];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    Vector2 nPos = TerrainDecoratorSampling.AlphamapNorm(x, y, w, h);
                    decorator.EvaluateLayersAtPixel(nPos, x, y);
                    decorator.WriteSplatPixel(x, y, cpuMap);

                    int pixelIndex = y * w + x;
                    for (int layerNo = 0; layerNo < decorCount; layerNo++)
                    {
                        float cpu = decorator.layers[layerNo].resultWeight;
                        float gpu = gpuFlat[layerNo * pixelCount + pixelIndex];
                        maxDiff = Mathf.Max(maxDiff, Mathf.Abs(cpu - gpu));
                    }
                }
            }
        }
        else
        {
            int samples = Mathf.Min(64, pixelCount);
            var rng = new System.Random(42);
            for (int s = 0; s < samples; s++)
            {
                int x = rng.Next(0, w);
                int y = rng.Next(0, h);
                Vector2 nPos = TerrainDecoratorSampling.AlphamapNorm(x, y, w, h);
                decorator.EvaluateLayersAtPixel(nPos, x, y);

                int pixelIndex = y * w + x;
                for (int layerNo = 0; layerNo < decorCount; layerNo++)
                {
                    float cpu = decorator.layers[layerNo].resultWeight;
                    float gpu = gpuFlat[layerNo * pixelCount + pixelIndex];
                    maxDiff = Mathf.Max(maxDiff, Mathf.Abs(cpu - gpu));
                }
            }
        }

        if (maxDiff > 1e-4f)
            Debug.LogWarning($"TerrainDecorator GPU parity: max layer-weight diff vs CPU = {maxDiff:E3}. Filters: Height/Slope/Painted/Noise/Texture/Layer.");
        else
            Debug.Log($"TerrainDecorator GPU parity: OK (max diff {maxDiff:E3}). All filter types use CPU-matched bake/eval.");
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
        {
            for (int x = 0; x < width; x++)
                pixels[y * width + x] = map[y * width + (width - x - 1)];
        }
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
        var pixels = new Color[width * height];
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
