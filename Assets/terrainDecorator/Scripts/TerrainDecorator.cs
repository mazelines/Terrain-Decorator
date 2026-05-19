using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
//using System;

public class TerrainDecorator : MonoBehaviour
{
	
	public enum FilterType {height, slope, painted, noise, texture, layer }
	public enum BlendType {add, sub, mul,max,min }
	public enum ImageChannel {r, g, b, a }
	public enum LayerType {texture,tree }
	
	public Terrain t;
	[System.Serializable]

		public class Layers{

			public string name;
			public bool active;
			public int layerIndex;
			public LayerType layerType;
	
			public float probability;
			public int maximumTreeCount;
			public int treeCount;
			
			public float width;
			public float height;
		
			public float randomPosition;
			public float randomRotation;
			public float randomSize;
			public float randomHealth;

			public float offset;
			
			public float debugLayerWeight;
			public float resultWeight;
			public float debugResultWeight;
			public float correctedResultWeight;
			public float debugCorrectedResultWeight;
	
			public List<Rules> rules = new List<Rules> ();
			public int activeRuleCount =0;

	
	}
	
	[System.Serializable]
	public class Rules{
 
 	
		public bool active;
		public FilterType filter;
		public float min;
		public float max;
		public float frequency;
		public float lacunarity ;
		
		public float intensity=1;
		public float contrast;
		public Texture2D texture;
		public ImageChannel imageChannel;
		public int targetLayerIndex;
	
		[System.NonSerialized]
		public Color[] map;
		public BlendType blend;
		public float weight;
		
		
		public float debugWeight; 

	}
	
	[HideInInspector]
	public List<Layers> layers = new List<Layers> ();
	[SerializeField]
	
	public float fallOffDistance = 2;
	public bool fallOfNoise = true;
	public int perlinOctaves = 5;
	public bool clearTexture = true;
	public bool clearTrees = true;
	
	public Vector2Int debugPos;
	
	public float fallofNoiseFrequency = 150f;
	public float fallofNoiseContrast= 10f;
	public float fallofNoiseAmplitude= 1f;

	[HideInInspector]
	public int textureLayerCount = 0;
	private int treePrototypeCount =0;
	private int alphamapWidth = 0;
	private int alphamapHeight = 0;
	private int heightMapWidth = 0;
	private int heightMapHeight = 0;
	float[,,] map;// = new float[alphamapWidth, alphamapHeight, t.terrainData.terrainLayers.Length];
	public bool showProgress;
	public bool calculating= false;
	public float calculatingPercent;
	
	public bool debugMode = false;

	[Tooltip("Editor only. Uses GPU pipeline (rule evaluate + splat). Falls back to full CPU when Layer filter rules are active.")]
	public bool useGpuDecorate = true;

	[Tooltip("Experimental. GPU compute for rule weights. Off = CPU rule evaluate (matches full CPU decorate).")]
	public bool useGpuRuleEvaluate = false;
	
	void OnEnable() {
		
		t=GetComponent<Terrain>();
		
	}
	
    void Start()
	{
		t=GetComponent<Terrain>();
		
    }
   
	public void GetTerrain() {
		t=GetComponent<Terrain>();
	}
	
	public float CalculateWeigth( float val, float min, float max, float falloff  ) {
		
		float weight = 0;
		
		if ( val > min+falloff/2f && val <= max-falloff/2f ) {
			weight= 1f;
		}
		else if (val<min+falloff/2f ) {
		
			weight = ( val-(min-falloff/2))/falloff;
		
		}
		else if (val> max-falloff/2f ) {
			
			weight = 1-(val-(max-falloff/2f))/falloff;
			
		}

		return weight;
		
	}
	
	public float GetWeight(int index , int x, int y) {
	
		Terrain terrain = t;
		TerrainData terrainData = terrain.terrainData;
		Vector3 terrainPos = terrain.transform.position;
	
		float[,,] splatmapData = terrainData.GetAlphamaps(x,y,1,1);
	
		float[] cellMix = new float[splatmapData.GetUpperBound(2)+1];
		return splatmapData[0,0,index];        
	}
	
	public float ProPerlin(float x, float y, float frequency, float lacunarity ) {
		
		float total = 0;
		float _frequency = frequency;
		float amplitude = 1f;
		float persistence= 0.5f;
		float totalAmplitude = 0;  // Used for normalizing result to 0.0 - 1.0
		for(int i=0;i<perlinOctaves;i++) {
			total += Mathf.PerlinNoise(x * _frequency, y * _frequency) * amplitude;
			totalAmplitude += amplitude;
			amplitude *= persistence;
			_frequency *= lacunarity;
		}
		return total/totalAmplitude;
		
	}
	
	public void ProcessTextureFilters() {
#if UNITY_EDITOR
		GetTerrain();
		int width = t.terrainData.alphamapWidth;
		int height = t.terrainData.alphamapHeight;

		for ( int layerNo=0; layerNo<layers.Count;layerNo++)
			if ( layers[layerNo].active)
			for (int i = 0; i < layers[layerNo].rules.Count; i++)  {
			
				EditorUtility.DisplayProgressBar("Prepare", "preparing rule Textures", i*1.0f/( layers[layerNo].rules.Count-1));
		
				
				if (  layers[layerNo].active && layers[layerNo].rules[i].active &&  layers[layerNo].rules[i].filter == FilterType.texture )
			{
				int w =   layers[layerNo].rules[i].texture.width;
				int h =   layers[layerNo].rules[i].texture.height;
				
				Color[] mapColors =   layers[layerNo].rules[i].texture.GetPixels();
				 layers[layerNo].rules[i].map = new Color[width * height];
				
				if (width != w || height != h)
				{
					float ratioX = (1.0f / ((float) width / (w - 1))) ;
					float ratioY = (1.0f / ((float) width / (h - 1))) ;
					for (int y = 0; y < width; y++)
					{
						int yy = Mathf.FloorToInt(y * ratioY) ;
						int y1 = yy * w;
						int y2 = (yy + 1) * w;
						int yw = y * width;
						for (int x = 0; x < width; x++)
						{
							int xx = Mathf.FloorToInt(x * ratioX) ;
							Color bl = mapColors[y1 + xx];
							Color br = mapColors[y1 + xx + 1];
							Color tl = mapColors[y2 + xx];
							Color tr = mapColors[y2 + xx + 1];
							float xLerp = x * ratioX - xx;
							layers[layerNo].rules[i].map[(x)*width +(width-y-1)] = Color.Lerp(Color.Lerp(bl, br, xLerp), Color.Lerp(tl, tr, xLerp), y * ratioY - (float)yy);
						}
					}
				}
				else
				{
					 layers[layerNo].rules[i].map = mapColors;
				}
			}
			EditorUtility.ClearProgressBar();
		}
#endif

	}
	public int smoothcount=1;
	
	public void EvaluateLayersAtPixel(Vector2 nPos, int x, int y)
	{
		for (int layerNo = 0; layerNo < layers.Count; layerNo++)
		{
			float layerResultWeight = 0;

			if (layers[layerNo].active)
			{
				if (layers[layerNo].rules.Count == 0 ||
				    (layers[layerNo].rules.Count > 0 && layers[layerNo].activeRuleCount == 0))
					layerResultWeight = 1f;

				for (int i = 0; i < layers[layerNo].rules.Count; i++)
				{
					if (!layers[layerNo].rules[i].active)
						continue;

					float weight = 0;
					var rule = layers[layerNo].rules[i];

					if (rule.filter == FilterType.slope)
					{
						float angle = TerrainDecoratorSampling.SampleSlope(t.terrainData, x, y, alphamapWidth, alphamapHeight);
						weight = CalculateWeigth(angle, rule.min, rule.max, heightMapHeight / 256);
					}

					if (rule.filter == FilterType.height)
					{
						float _height = TerrainDecoratorSampling.SampleHeightWorld(this, t.terrainData, x, y, alphamapWidth, alphamapHeight);
						weight = CalculateWeigth(_height, rule.min, rule.max, heightMapHeight / 512f * fallOffDistance);
					}

					if (rule.filter == FilterType.painted)
						weight = TerrainDecoratorSampling.SamplePainted(this, layers[layerNo].layerIndex, x, y);

					if (rule.filter == FilterType.texture && rule.map != null)
						weight = TerrainDecoratorSampling.SampleTextureMask(rule.map, x, y, alphamapWidth, rule.imageChannel);

					if (rule.filter == FilterType.noise)
						weight = TerrainDecoratorSampling.SampleNoise(this, x, y, alphamapWidth, alphamapHeight, rule.frequency, rule.lacunarity);

					if (rule.filter == FilterType.layer)
					{
						if (rule.targetLayerIndex < textureLayerCount && x > 1 && y > 1)
							weight = map[x - 1, y - 1, rule.targetLayerIndex];
					}

					if (rule.filter == FilterType.height && fallOfNoise && weight < 1 && weight > 0)
					{
						float noise = TerrainDecoratorSampling.SampleFalloffNoise(this, x, y, alphamapWidth, alphamapHeight);
						weight = Mathf.Clamp((noise - 0.5f) * 2f * fallofNoiseAmplitude * (1 - weight) + weight, 0f, 1f);
					}

					if (rule.intensity < 0)
						weight = 1 - weight;

					weight = (weight - 0.5f) * (rule.contrast + 1) + 0.5f;
					weight *= Mathf.Abs(rule.intensity);
					weight = Mathf.Clamp(weight, 0, 1);

					if (debugPos.x == x && debugPos.y == y)
						rule.debugWeight = weight;

					rule.weight = weight;
					if (rule.blend == BlendType.add)
						layerResultWeight += weight;
					else if (rule.blend == BlendType.sub)
						layerResultWeight -= weight;
					else if (rule.blend == BlendType.mul)
						layerResultWeight *= weight;
					else if (rule.blend == BlendType.min)
						layerResultWeight = Mathf.Min(layerResultWeight, weight);
					else if (rule.blend == BlendType.max)
						layerResultWeight = Mathf.Max(layerResultWeight, weight);
				}

				layerResultWeight = Mathf.Clamp(layerResultWeight, 0f, 1f);

				if (debugPos.x == x && debugPos.y == y)
					layers[layerNo].debugResultWeight = Mathf.Clamp(layerResultWeight, 0, 1f);

				layers[layerNo].resultWeight = Mathf.Clamp(layerResultWeight, 0, 1f);
			}
		}
	}

	public void ProcessLayers(Vector2 nPos, int x, int y)
	{
		EvaluateLayersAtPixel(nPos, x, y);
		ApplyTreesAtPixel(nPos, x, y);
	}

	void ApplyTreesAtPixel(Vector2 nPos, int x, int y) {
		if (!clearTrees)
			return;

		for (int layerNo = 0; layerNo < layers.Count; layerNo++) {
			if (!layers[layerNo].active)
				continue;

			float probility = Random.Range(0f, 1f);
			int inc = Mathf.CeilToInt(alphamapWidth / Mathf.Sqrt(layers[layerNo].maximumTreeCount));
			int idx = layerNo;
			if (y % inc != 0 || x % inc != 0)
				continue;
			if (layers[layerNo].treeCount >= layers[layerNo].maximumTreeCount)
				continue;
			if (layers[idx].resultWeight <= 0)
				continue;
			if (layers[layerNo].layerIndex < textureLayerCount)
				continue;
			if (layers[layerNo].layerIndex - textureLayerCount < 0 ||
			    layers[layerNo].layerIndex - textureLayerCount >= treePrototypeCount)
				continue;
			if (probility >= layers[layerNo].probability * layers[idx].resultWeight)
				continue;

			TreeInstance tree = new TreeInstance();
			float rndPos = Random.Range(-1f, 1f) / alphamapWidth * layers[idx].randomPosition * inc;
			float rndRot = Random.Range(-1f, 1f) * layers[idx].randomRotation * 6.283f;
			float rndScale = 1f + Random.Range(0, 1f) * layers[idx].randomSize;
			float nx = nPos.x + rndPos;
			float ny = nPos.y + rndPos;
			float _height = t.terrainData.GetHeight(
				Mathf.FloorToInt(ny * (heightMapHeight - 1)),
				Mathf.FloorToInt(nx * (heightMapWidth - 1)));

			tree.position = new Vector3(ny, (_height + layers[idx].offset) / t.terrainData.heightmapScale.y, nx);
			tree.color = Color.Lerp(Color.white, new Color(1, 0.8f, 0), Random.Range(0f, layers[idx].randomHealth));
			tree.rotation = rndRot;
			tree.heightScale = layers[idx].height * rndScale;
			tree.widthScale = layers[idx].width * rndScale;
			tree.prototypeIndex = layers[idx].layerIndex - textureLayerCount;
			tree.lightmapColor = Color.white;
			layers[layerNo].treeCount++;
			t.AddTreeInstance(tree);
		}
	}

#if UNITY_EDITOR
	/// <summary>CPU Decorate와 동일한 룰 평가 + WriteSplatPixel. GPU fast path도 이 함수를 사용합니다.</summary>
	public void DecorateSplatCpu(float[,,] targetMap, out float[] layerResultFlat)
	{
		int decorLayerCount = layers.Count;
		int pixelCount = alphamapWidth * alphamapHeight;
		layerResultFlat = new float[Mathf.Max(decorLayerCount, 1) * pixelCount];

		ProcessTextureFilters();

		for (int y = 0; y < alphamapHeight; y++)
		{
			for (int x = 0; x < alphamapWidth; x++)
			{
				float normX = x * 1.0f / (alphamapWidth - 1);
				float normY = y * 1.0f / (alphamapHeight - 1);
				EvaluateLayersAtPixel(new Vector2(normX, normY), x, y);

				int pixelIndex = y * alphamapWidth + x;
				for (int layerNo = 0; layerNo < decorLayerCount; layerNo++)
					layerResultFlat[layerNo * pixelCount + pixelIndex] = layers[layerNo].resultWeight;

				WriteSplatPixel(x, y, targetMap);
			}
		}
	}

	public void ApplyLayerWeightsForPixel(int x, int y, float[] layerResultFlat)
	{
		int pixelCount = alphamapWidth * alphamapHeight;
		int pixelIndex = y * alphamapWidth + x;
		for (int layerNo = 0; layerNo < layers.Count; layerNo++)
			layers[layerNo].resultWeight = layerResultFlat[layerNo * pixelCount + pixelIndex];
	}

	public void WriteSplatPixel(int x, int y, float[,,] targetMap)
	{
		float[] weights = new float[textureLayerCount + treePrototypeCount];
		float totalWeight = 0f;
		float kalan = 1f;

		for (int r = layers.Count - 1; r >= 0; r--)
		{
			if (!layers[r].active)
				continue;

			float corrected = 0f;
			if (layers[r].layerIndex < textureLayerCount)
			{
				corrected = layers[r].resultWeight * kalan;
				kalan -= corrected;
				layers[r].correctedResultWeight = corrected;
			}

			if (r < textureLayerCount + treePrototypeCount)
				weights[layers[r].layerIndex] += corrected;
		}

		for (int i = 0; i < weights.Length; i++)
			if (i < textureLayerCount)
				totalWeight += weights[i];

		int firstActiveTextureLayer = 0;
		if (weights.Length > 0)
		{
			for (int layerNo = 0; layerNo < layers.Count; layerNo++)
			{
				if (layers[layerNo].active || layers[layerNo].layerIndex < textureLayerCount)
					firstActiveTextureLayer = layers[layerNo].layerIndex;
			}
		}

		if (firstActiveTextureLayer < weights.Length && totalWeight < 1f)
		{
			weights[firstActiveTextureLayer] = 1f - totalWeight;
			totalWeight = 1f;
		}

		totalWeight = Mathf.Max(totalWeight, 1e-5f);
		for (int i = 0; i < textureLayerCount; i++)
			targetMap[x, y, i] = weights[i] / totalWeight;
	}

	void PlaceTreesFromGpuLayerWeights(float[] layerWeights) {
		int decorCount = layers.Count;
		int pixelCount = alphamapWidth * alphamapHeight;
		for (int y = 0; y < alphamapHeight; y++) {
			for (int x = 0; x < alphamapWidth; x++) {
				int pixelIndex = y * alphamapWidth + x;
				for (int layerNo = 0; layerNo < decorCount; layerNo++) {
					float w = layerWeights[layerNo * pixelCount + pixelIndex];
					layers[layerNo].resultWeight = w;
					if (debugPos.x == x && debugPos.y == y)
						layers[layerNo].debugResultWeight = Mathf.Clamp(w, 0f, 1f);
				}

				float normX = x * 1.0f / (alphamapWidth - 1);
				float normY = y * 1.0f / (alphamapHeight - 1);
				ApplyTreesAtPixel(new Vector2(normX, normY), x, y);
			}
		}
	}
#endif
	public void Decorate() {
		StartCoroutine(DecorateNow());
	}
	
	public 	IEnumerator DecorateNow() {

#if UNITY_EDITOR

		if ( calculating)
			yield return new WaitForSeconds(0);
		else
			calculating=true;
			
		float startTime = Time.realtimeSinceStartup;
		 alphamapWidth = t.terrainData.alphamapWidth;
		 alphamapHeight = t.terrainData.alphamapHeight;
		 heightMapWidth = t.terrainData.heightmapResolution;
		 heightMapHeight = t.terrainData.heightmapResolution;
		
		textureLayerCount = t.terrainData.terrainLayers.Length;
		if ( clearTrees)
		treePrototypeCount =t.terrainData.treePrototypes.Length;
	
	
		if( !showProgress) 
			map = new float[alphamapWidth, alphamapHeight, t.terrainData.terrainLayers.Length];
		if ( map==null) 
			map = new float[alphamapWidth, alphamapHeight, t.terrainData.terrainLayers.Length];
		
		if ( clearTrees)
			t.terrainData.treeInstances = new TreeInstance[0];

		Random.InitState(42);

		//Init
		for ( int layerNo=0; layerNo<layers.Count;layerNo++) {
			layers[layerNo].activeRuleCount =0;
			layers[layerNo].debugResultWeight =0;
			layers[layerNo].resultWeight=0;
			layers[layerNo].correctedResultWeight=0;
			layers[layerNo].debugCorrectedResultWeight=0;
			layers[layerNo].treeCount = 0;
			for (int i = 0; i <  layers[layerNo].rules.Count; i++) {
				layers[layerNo].rules[i].debugWeight = 0f;
				if ( layers[layerNo].rules[i].active)
					layers[layerNo].activeRuleCount++;
				layers[layerNo].rules[i].weight = 0f;
				
			}
		}

		bool usedGpuSplat = false;
		if (useGpuDecorate)
		{
			if (TerrainDecoratorGpu.TryDecorateSplat(this, out float[,,] gpuMap, out float[] layerWeights))
			{
				map = gpuMap;
				PlaceTreesFromGpuLayerWeights(layerWeights);
				t.terrainData.SetAlphamaps(0, 0, map);
				usedGpuSplat = true;
			}
		}

		if (!usedGpuSplat)
		{
			if (showProgress)
			{
				ProcessTextureFilters();
				for (int y = 0; y < alphamapHeight; y++)
				{
					if (y % 50 == 0)
					{
						EditorUtility.DisplayProgressBar("Texture Rules", "Calculating Texture Rules ",
							Mathf.InverseLerp(0.0f, alphamapWidth, y));
						calculatingPercent = Mathf.InverseLerp(0.0f, alphamapWidth, y);
						t.terrainData.SetAlphamaps(0, 0, map);
						yield return 0;
					}

					for (int x = 0; x < alphamapWidth; x++)
					{
						float normX = x * 1.0f / (alphamapWidth - 1);
						float normY = y * 1.0f / (alphamapHeight - 1);
						ProcessLayers(new Vector2(normX, normY), x, y);
						WriteSplatPixel(x, y, map);
					}
				}
			}
			else
			{
				DecorateSplatCpu(map, out float[] layerWeights);
				PlaceTreesFromGpuLayerWeights(layerWeights);
			}

			t.terrainData.SetAlphamaps(0, 0, map);
		}
			
			
			
			
			
			
		
	
		EditorUtility.ClearProgressBar();
	
	
			t.Flush ();
	
		
		string decorateMode = " (CPU)";
		if (usedGpuSplat)
			decorateMode = useGpuRuleEvaluate ? " (GPU rule evaluate + CPU splat)" : " (CPU rule evaluate, GPU path)";
		Debug.Log("Decorating Time:" + (Time.realtimeSinceStartup - startTime) + decorateMode);
		
		
	
		yield return new WaitForSeconds(0);
		calculating = false;
#else
		yield return new WaitForSeconds(0);
		#endif
	}
	
	
	
	

    
}
