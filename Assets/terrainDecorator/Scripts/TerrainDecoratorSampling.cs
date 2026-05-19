#if UNITY_EDITOR
using UnityEngine;

/// <summary>CPU EvaluateLayersAtPixel 과 동일한 좌표·수식으로 맵 샘플 (GPU 베이크용).</summary>
public static class TerrainDecoratorSampling
{
    public static Vector2 AlphamapNorm(int x, int y, int alphamapWidth, int alphamapHeight)
    {
        float denomX = alphamapWidth > 1 ? alphamapWidth - 1f : 1f;
        float denomY = alphamapHeight > 1 ? alphamapHeight - 1f : 1f;
        return new Vector2(x / denomX, y / denomY);
    }

    public static float SampleHeightWorld(
        TerrainDecorator decorator,
        TerrainData data,
        int x,
        int y,
        int alphamapWidth,
        int alphamapHeight)
    {
        Vector2 nPos = AlphamapNorm(x, y, alphamapWidth, alphamapHeight);
        int heightMapHeight = data.heightmapResolution;
        int heightMapWidth = data.heightmapResolution;
        int hx = Mathf.FloorToInt(nPos.y * (heightMapHeight - 1));
        int hy = Mathf.FloorToInt(nPos.x * (heightMapWidth - 1));

        if (decorator.smoothcount <= 0)
            return data.GetHeight(hx, hy);

        float totalHeight = 0f;
        float valueCount = 0f;
        for (int sx = hx - decorator.smoothcount; sx <= hx + decorator.smoothcount; sx++)
        {
            if (sx < 0 || sx >= heightMapHeight)
                continue;
            for (int sy = hy - decorator.smoothcount; sy <= hy + decorator.smoothcount; sy++)
            {
                if (sy < 0 || sy >= heightMapWidth)
                    continue;
                totalHeight += data.GetHeight(sx, sy);
                valueCount++;
            }
        }

        return valueCount > 0f ? totalHeight / valueCount : data.GetHeight(hx, hy);
    }

    public static float SampleSlope(
        TerrainData data,
        int x,
        int y,
        int alphamapWidth,
        int alphamapHeight)
    {
        Vector2 nPos = AlphamapNorm(x, y, alphamapWidth, alphamapHeight);
        return data.GetSteepness(nPos.y, nPos.x);
    }

    public static float SampleFalloffNoise(TerrainDecorator decorator, int x, int y, int alphamapWidth, int alphamapHeight)
    {
        Vector2 nPos = AlphamapNorm(x, y, alphamapWidth, alphamapHeight);
        return Mathf.PerlinNoise(nPos.x * decorator.fallofNoiseFrequency, nPos.y * decorator.fallofNoiseFrequency);
    }

    public static float SampleNoise(
        TerrainDecorator decorator,
        int x,
        int y,
        int alphamapWidth,
        int alphamapHeight,
        float frequency,
        float lacunarity)
    {
        Vector2 nPos = AlphamapNorm(x, y, alphamapWidth, alphamapHeight);
        return decorator.ProPerlin(nPos.x, nPos.y, frequency, lacunarity);
    }

    public static float SamplePainted(TerrainDecorator decorator, int layerIndex, int x, int y)
    {
        return decorator.GetWeight(layerIndex, y, x);
    }

    /// <summary>GetAlphamaps(0,0,w,h) 1회로 Painted 베이크 (픽셀마다 GetAlphamaps 방지).</summary>
    public static void BakePaintedAlphamap(TerrainData data, float[] output, int width, int height)
    {
        int layerCount = data.terrainLayers.Length;
        if (layerCount <= 0 || output == null || output.Length == 0)
            return;

        float[,,] maps = data.GetAlphamaps(0, 0, width, height);
        int pixelCount = width * height;
        for (int layer = 0; layer < layerCount; layer++)
        {
            int layerOffset = layer * pixelCount;
            for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                output[layerOffset + y * width + x] = maps[y, x, layer];
        }
    }

    public static float SampleTextureMask(Color[] map, int x, int y, int alphamapWidth, TerrainDecorator.ImageChannel channel)
    {
        int idx = y * alphamapWidth + (alphamapWidth - x - 1);
        Color c = map[idx];
        switch (channel)
        {
            case TerrainDecorator.ImageChannel.g: return c.g;
            case TerrainDecorator.ImageChannel.b: return c.b;
            case TerrainDecorator.ImageChannel.a: return c.a;
            default: return c.r;
        }
    }

    public static void BakeHeightMap(
        TerrainDecorator decorator,
        TerrainData data,
        int width,
        int height,
        float[] output)
    {
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            output[y * width + x] = SampleHeightWorld(decorator, data, x, y, width, height);
    }

    public static void BakeSlopeMap(TerrainData data, int width, int height, float[] output)
    {
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            output[y * width + x] = SampleSlope(data, x, y, width, height);
    }

    public static void BakeFalloffNoiseMap(TerrainDecorator decorator, int width, int height, float[] output)
    {
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            output[y * width + x] = SampleFalloffNoise(decorator, x, y, width, height);
    }

    public static void BakeNoiseMap(
        TerrainDecorator decorator,
        int width,
        int height,
        float frequency,
        float lacunarity,
        float[] output)
    {
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            output[y * width + x] = SampleNoise(decorator, x, y, width, height, frequency, lacunarity);
    }

    /// <summary>GetWeight(layer, y, x) 와 동일 — Painted 필터 GPU용.</summary>
    public static Texture2DArray BakePaintedAlphamapTexture(
        TerrainDecorator decorator,
        TerrainData data,
        int width,
        int height)
    {
        int layerCount = data.terrainLayers.Length;
        if (layerCount <= 0)
            return null;

        int pixelCount = width * height;
        var flat = new float[layerCount * pixelCount];
        BakePaintedAlphamap(data, flat, width, height);

        var array = new Texture2DArray(width, height, layerCount, TextureFormat.RGBA32, false, true);
        for (int layer = 0; layer < layerCount; layer++)
        {
            var pixels = new Color[pixelCount];
            int offset = layer * pixelCount;
            for (int i = 0; i < pixelCount; i++)
                pixels[i] = new Color(flat[offset + i], 0f, 0f, 1f);
            array.SetPixels(pixels, layer);
        }
        array.Apply(false, false);
        array.wrapMode = TextureWrapMode.Clamp;
        array.filterMode = FilterMode.Point;
        return array;
    }
}
#endif
