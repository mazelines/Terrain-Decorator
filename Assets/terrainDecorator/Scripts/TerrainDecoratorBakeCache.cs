#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>Height/Slope/Painted/Noise 등 입력 맵 1회 베이크. CPU·GPU 공용.</summary>
public sealed class TerrainDecoratorBakeContext : IDisposable
{
    public int AlphamapWidth;
    public int AlphamapHeight;
    public int PixelCount;
    public int TerrainLayerCount;

    public float[] HeightWorld;
    public float[] Slope;
    public float[] FalloffNoise;
    public float[] PaintedAlphamap;
    public Dictionary<string, float[]> NoiseMaps;
    public Dictionary<string, float[]> TextureMaskMaps;

    public Texture2DArray PaintedAlphamapTexture;

    public ComputeBuffer HeightBuffer;
    public ComputeBuffer SlopeBuffer;
    public ComputeBuffer FalloffNoiseBuffer;

    int _cacheKey;

    public int CacheKey => _cacheKey;

    public float SampleHeight(int x, int y) => HeightWorld[PixelIndex(x, y)];

    public float SampleSlope(int x, int y) => Slope[PixelIndex(x, y)];

    public float SampleFalloffNoise(int x, int y) => FalloffNoise[PixelIndex(x, y)];

    public float SamplePainted(int terrainLayerIndex, int x, int y)
    {
        if (terrainLayerIndex < 0 || terrainLayerIndex >= TerrainLayerCount)
            return 0f;
        return PaintedAlphamap[terrainLayerIndex * PixelCount + PixelIndex(x, y)];
    }

    public bool TrySampleNoise(float frequency, float lacunarity, int x, int y, out float value)
    {
        string key = NoiseKey(frequency, lacunarity);
        if (NoiseMaps.TryGetValue(key, out float[] map))
        {
            value = map[PixelIndex(x, y)];
            return true;
        }
        value = 0f;
        return false;
    }

    public static string NoiseKey(float frequency, float lacunarity) =>
        frequency.ToString("R") + "_" + lacunarity.ToString("R");

    public static string TextureMaskKey(int decorLayerNo, int ruleIndex) =>
        "t_" + decorLayerNo + "_" + ruleIndex;

    public bool TrySampleTextureMask(int decorLayerNo, int ruleIndex, int x, int y, out float value)
    {
        string key = TextureMaskKey(decorLayerNo, ruleIndex);
        if (TextureMaskMaps != null && TextureMaskMaps.TryGetValue(key, out float[] map))
        {
            value = map[PixelIndex(x, y)];
            return true;
        }
        value = 0f;
        return false;
    }

    int PixelIndex(int x, int y) => y * AlphamapWidth + x;

    public static TerrainDecoratorBakeContext Build(TerrainDecorator decorator, TerrainData data)
    {
        decorator.ProcessTextureFilters();

        int w = data.alphamapWidth;
        int h = data.alphamapHeight;
        int pixelCount = w * h;
        int terrainLayerCount = data.terrainLayers.Length;

        var ctx = new TerrainDecoratorBakeContext
        {
            AlphamapWidth = w,
            AlphamapHeight = h,
            PixelCount = pixelCount,
            TerrainLayerCount = terrainLayerCount,
            HeightWorld = new float[pixelCount],
            Slope = new float[pixelCount],
            FalloffNoise = new float[pixelCount],
            PaintedAlphamap = new float[Mathf.Max(terrainLayerCount, 1) * pixelCount],
            NoiseMaps = new Dictionary<string, float[]>(),
            TextureMaskMaps = new Dictionary<string, float[]>(),
            _cacheKey = ComputeCacheKey(decorator, data)
        };

        TerrainDecoratorSampling.BakeHeightMap(decorator, data, w, h, ctx.HeightWorld);
        TerrainDecoratorSampling.BakeSlopeMap(data, w, h, ctx.Slope);
        if (decorator.fallOfNoise)
            TerrainDecoratorSampling.BakeFalloffNoiseMap(decorator, w, h, ctx.FalloffNoise);
        else
            Array.Fill(ctx.FalloffNoise, 0.5f);

        for (int layer = 0; layer < terrainLayerCount; layer++)
        {
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                ctx.PaintedAlphamap[layer * pixelCount + y * w + x] =
                    TerrainDecoratorSampling.SamplePainted(decorator, layer, x, y);
        }

        CollectNoiseKeys(decorator, ctx);
        CollectTextureMasks(decorator, ctx);
        ctx.PaintedAlphamapTexture = BuildPaintedTexture(ctx);

        ctx.HeightBuffer = new ComputeBuffer(pixelCount, sizeof(float));
        ctx.SlopeBuffer = new ComputeBuffer(pixelCount, sizeof(float));
        ctx.FalloffNoiseBuffer = new ComputeBuffer(pixelCount, sizeof(float));
        ctx.HeightBuffer.SetData(ctx.HeightWorld);
        ctx.SlopeBuffer.SetData(ctx.Slope);
        ctx.FalloffNoiseBuffer.SetData(ctx.FalloffNoise);

        return ctx;
    }

    static void CollectTextureMasks(TerrainDecorator decorator, TerrainDecoratorBakeContext ctx)
    {
        int w = ctx.AlphamapWidth;
        int h = ctx.AlphamapHeight;
        for (int layerNo = 0; layerNo < decorator.layers.Count; layerNo++)
        {
            var layer = decorator.layers[layerNo];
            if (!layer.active)
                continue;
            for (int i = 0; i < layer.rules.Count; i++)
            {
                var rule = layer.rules[i];
                if (!rule.active || rule.filter != TerrainDecorator.FilterType.texture || rule.map == null)
                    continue;

                string key = TextureMaskKey(layerNo, i);
                if (ctx.TextureMaskMaps.ContainsKey(key))
                    continue;

                var mask = new float[ctx.PixelCount];
                for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    mask[y * w + x] = TerrainDecoratorSampling.SampleTextureMask(rule.map, x, y, w, rule.imageChannel);
                ctx.TextureMaskMaps[key] = mask;
            }
        }
    }

    static void CollectNoiseKeys(TerrainDecorator decorator, TerrainDecoratorBakeContext ctx)
    {
        for (int layerNo = 0; layerNo < decorator.layers.Count; layerNo++)
        {
            var layer = decorator.layers[layerNo];
            if (!layer.active)
                continue;
            for (int i = 0; i < layer.rules.Count; i++)
            {
                var rule = layer.rules[i];
                if (!rule.active || rule.filter != TerrainDecorator.FilterType.noise)
                    continue;
                string key = NoiseKey(rule.frequency, rule.lacunarity);
                if (ctx.NoiseMaps.ContainsKey(key))
                    continue;
                var noiseFlat = new float[ctx.PixelCount];
                TerrainDecoratorSampling.BakeNoiseMap(decorator, ctx.AlphamapWidth, ctx.AlphamapHeight,
                    rule.frequency, rule.lacunarity, noiseFlat);
                ctx.NoiseMaps[key] = noiseFlat;
            }
        }
    }

    static Texture2DArray BuildPaintedTexture(TerrainDecoratorBakeContext ctx)
    {
        if (ctx.TerrainLayerCount <= 0)
            return null;

        var array = new Texture2DArray(ctx.AlphamapWidth, ctx.AlphamapHeight, ctx.TerrainLayerCount,
            TextureFormat.RGBA32, false, true);
        for (int layer = 0; layer < ctx.TerrainLayerCount; layer++)
        {
            var pixels = new Color[ctx.PixelCount];
            int offset = layer * ctx.PixelCount;
            for (int i = 0; i < ctx.PixelCount; i++)
                pixels[i] = new Color(ctx.PaintedAlphamap[offset + i], 0f, 0f, 1f);
            array.SetPixels(pixels, layer);
        }
        array.Apply(false, false);
        array.wrapMode = TextureWrapMode.Clamp;
        array.filterMode = FilterMode.Point;
        return array;
    }

    static int ComputeCacheKey(TerrainDecorator decorator, TerrainData data)
    {
        unchecked
        {
            int h = 17;
            h = h * 31 + data.alphamapWidth;
            h = h * 31 + data.alphamapHeight;
            h = h * 31 + data.heightmapResolution;
            h = h * 31 + decorator.smoothcount;
            h = h * 31 + (decorator.fallOfNoise ? 1 : 0);
            h = h * 31 + decorator.fallofNoiseFrequency.GetHashCode();
            h = h * 31 + decorator.fallofNoiseAmplitude.GetHashCode();
            h = h * 31 + decorator.fallOffDistance.GetHashCode();
            h = h * 31 + data.terrainLayers.Length;
            return h;
        }
    }

    public void Dispose()
    {
        HeightBuffer?.Release();
        SlopeBuffer?.Release();
        FalloffNoiseBuffer?.Release();
        HeightBuffer = null;
        SlopeBuffer = null;
        FalloffNoiseBuffer = null;

        if (PaintedAlphamapTexture != null)
            UnityEngine.Object.DestroyImmediate(PaintedAlphamapTexture);
        PaintedAlphamapTexture = null;
    }
}

public static class TerrainDecoratorBakeCache
{
    static TerrainDecoratorBakeContext _cached;
    static int _cachedKey;

    public static bool TryAcquire(TerrainDecorator decorator, TerrainData data, bool useCache, out TerrainDecoratorBakeContext ctx)
    {
        var built = TerrainDecoratorBakeContext.Build(decorator, data);
        int key = built.CacheKey;

        if (useCache && _cached != null && _cachedKey == key)
        {
            built.Dispose();
            ctx = _cached;
            return true;
        }

        _cached?.Dispose();
        _cached = built;
        _cachedKey = key;
        ctx = built;
        return true;
    }

    public static void Invalidate()
    {
        _cached?.Dispose();
        _cached = null;
    }
}
#endif
