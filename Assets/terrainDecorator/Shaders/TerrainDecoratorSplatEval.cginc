#ifndef TERRAIN_DECORATOR_SPLAT_EVAL_INCLUDED
#define TERRAIN_DECORATOR_SPLAT_EVAL_INCLUDED

#define FILTER_HEIGHT 0
#define FILTER_SLOPE 1
#define FILTER_PAINTED 2
#define FILTER_NOISE 3
#define FILTER_TEXTURE 4
#define FILTER_LAYER 5

#define BLEND_ADD 0
#define BLEND_SUB 1
#define BLEND_MUL 2
#define BLEND_MAX 3
#define BLEND_MIN 4

struct RuleData
{
    int filter;
    int blend;
    int active;
    int imageChannel;
    int targetLayerIndex;
    int maskTextureIndex;
    float minVal;
    float maxVal;
    float frequency;
    float lacunarity;
    float intensity;
    float contrast;
    int paintedLayerIndex;
    int pad0;
    int pad1;
    int pad2;
};

struct LayerData
{
    int active;
    int layerIndex;
    int activeRuleCount;
    int ruleStart;
    int ruleCount;
    int pad0;
    int pad1;
    int pad2;
};

Texture2DArray<float4> _InputAlphamap;
Texture2DArray<float4> _MaskTextures;

StructuredBuffer<float> _HeightWorldFlat;
StructuredBuffer<float> _SlopeFlat;
StructuredBuffer<float> _FalloffNoiseFlat;
StructuredBuffer<RuleData> _Rules;
StructuredBuffer<LayerData> _Layers;

int _AlphamapWidth;
int _AlphamapHeight;
int _TextureLayerCount;
int _DecorLayerCount;
float _HeightFalloff;
float _SlopeFalloff;
int _FalloffNoise;
float _FalloffNoiseAmplitude;

float CalculateWeight(float val, float minVal, float maxVal, float falloff)
{
    float weight = 0.0;
    float halfFalloff = falloff * 0.5;
    if (val > minVal + halfFalloff && val <= maxVal - halfFalloff)
        weight = 1.0;
    else if (val < minVal + halfFalloff)
        weight = (val - (minVal - halfFalloff)) / falloff;
    else if (val > maxVal - halfFalloff)
        weight = 1.0 - (val - (maxVal - halfFalloff)) / falloff;
    return weight;
}

int PixelIndex(int x, int y)
{
    return y * _AlphamapWidth + x;
}

float SampleHeightAt(int x, int y)
{
    return _HeightWorldFlat[PixelIndex(x, y)];
}

float SampleSlopeAt(int x, int y)
{
    return _SlopeFlat[PixelIndex(x, y)];
}

float SampleFalloffNoiseAt(int x, int y)
{
    return _FalloffNoiseFlat[PixelIndex(x, y)];
}

float SampleMask(int textureIndex, int x, int y, int channel)
{
    if (textureIndex < 0)
        return 0.0;

    // Mask textures are built from CPU-baked float[] in alphamap (x,y) order (see TerrainDecoratorBakeCache).
    float4 c = _MaskTextures.Load(int4(x, y, textureIndex, 0));
    if (channel == 0) return c.r;
    if (channel == 1) return c.g;
    if (channel == 2) return c.b;
    return c.a;
}

float SamplePainted(int layerIndex, int x, int y)
{
    return _InputAlphamap.Load(int4(x, y, layerIndex, 0)).r;
}

float SampleLayerSplat(RWStructuredBuffer<float> splatScratch, int x, int y, int targetLayer)
{
    if (x <= 1 || y <= 1 || targetLayer < 0 || targetLayer >= _TextureLayerCount)
        return 0.0;
    int prevIndex = PixelIndex(x - 1, y - 1);
    return splatScratch[prevIndex * _TextureLayerCount + targetLayer];
}

float EvaluateRuleWeight(RuleData rule, int x, int y, RWStructuredBuffer<float> splatScratch, bool useLayerSplat)
{
    float weight = 0.0;

    if (rule.filter == FILTER_SLOPE)
        weight = CalculateWeight(SampleSlopeAt(x, y), rule.minVal, rule.maxVal, _SlopeFalloff);
    else if (rule.filter == FILTER_HEIGHT)
    {
        weight = CalculateWeight(SampleHeightAt(x, y), rule.minVal, rule.maxVal, _HeightFalloff);
        if (_FalloffNoise > 0 && weight < 1.0 && weight > 0.0)
        {
            float noise = SampleFalloffNoiseAt(x, y);
            weight = clamp((noise - 0.5) * 2.0 * _FalloffNoiseAmplitude * (1.0 - weight) + weight, 0.0, 1.0);
        }
    }
    else if (rule.filter == FILTER_PAINTED)
        weight = SamplePainted(rule.paintedLayerIndex, x, y);
    else if (rule.filter == FILTER_TEXTURE)
        weight = SampleMask(rule.maskTextureIndex, x, y, rule.imageChannel);
    else if (rule.filter == FILTER_NOISE)
        weight = SampleMask(rule.maskTextureIndex, x, y, 0);
    else if (rule.filter == FILTER_LAYER && useLayerSplat)
        weight = SampleLayerSplat(splatScratch, x, y, rule.targetLayerIndex);

    if (rule.intensity < 0.0)
        weight = 1.0 - weight;

    weight = (weight - 0.5) * (rule.contrast + 1.0) + 0.5;
    weight *= abs(rule.intensity);
    return saturate(weight);
}

float BlendRuleWeight(float layerResultWeight, float weight, int blend)
{
    if (blend == BLEND_ADD) return layerResultWeight + weight;
    if (blend == BLEND_SUB) return layerResultWeight - weight;
    if (blend == BLEND_MUL) return layerResultWeight * weight;
    if (blend == BLEND_MIN) return min(layerResultWeight, weight);
    if (blend == BLEND_MAX) return max(layerResultWeight, weight);
    return layerResultWeight;
}

float EvaluateDecorLayerAtPixel(int decorLayer, int x, int y, RWStructuredBuffer<float> splatScratch, bool useLayerSplat)
{
    LayerData layer = _Layers[decorLayer];
    if (layer.active == 0)
        return 0.0;
    if (layer.ruleCount == 0 || layer.activeRuleCount == 0)
        return 1.0;

    float layerResultWeight = 0.0;
    for (int r = 0; r < layer.ruleCount; r++)
    {
        RuleData rule = _Rules[layer.ruleStart + r];
        if (rule.active == 0)
            continue;
        float weight = EvaluateRuleWeight(rule, x, y, splatScratch, useLayerSplat);
        layerResultWeight = BlendRuleWeight(layerResultWeight, weight, rule.blend);
    }
    return saturate(layerResultWeight);
}

void WriteSplatScratchFromLayerFlat(int x, int y, RWStructuredBuffer<float> layerResultFlat, RWStructuredBuffer<float> splatScratch)
{
    float weights[32];
    for (int i = 0; i < 32; i++)
        weights[i] = 0.0;

    float kalan = 1.0;
    int pixelCount = _AlphamapWidth * _AlphamapHeight;
    int pixelIndex = PixelIndex(x, y);

    for (int r = _DecorLayerCount - 1; r >= 0; r--)
    {
        LayerData layer = _Layers[r];
        if (layer.active == 0)
            continue;

        float resultWeight = layerResultFlat[r * pixelCount + pixelIndex];
        float corrected = 0.0;
        if (layer.layerIndex < _TextureLayerCount)
        {
            corrected = resultWeight * kalan;
            kalan -= corrected;
        }

        if (layer.layerIndex >= 0 && layer.layerIndex < 32)
            weights[layer.layerIndex] += corrected;
    }

    float totalWeight = 0.0;
    for (int t = 0; t < _TextureLayerCount; t++)
        totalWeight += weights[t];

    int firstActiveTextureLayer = 0;
    for (int l = 0; l < _DecorLayerCount; l++)
    {
        LayerData layer = _Layers[l];
        if (layer.active != 0 || layer.layerIndex < _TextureLayerCount)
            firstActiveTextureLayer = layer.layerIndex;
    }

    if (firstActiveTextureLayer < _TextureLayerCount && firstActiveTextureLayer < 32 && totalWeight < 1.0)
    {
        weights[firstActiveTextureLayer] += 1.0 - totalWeight;
        totalWeight = 1.0;
    }

    totalWeight = max(totalWeight, 1e-5);
    int baseIndex = pixelIndex * _TextureLayerCount;
    for (int ti = 0; ti < _TextureLayerCount; ti++)
        splatScratch[baseIndex + ti] = weights[ti] / totalWeight;
}

#endif
