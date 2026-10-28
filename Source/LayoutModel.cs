using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using Microsoft.ML.OnnxRuntime;

namespace DiffusionMaps
{
    /// <summary>
    /// Sampler settings and category names written by train/export_onnx.py (model_meta.json).
    /// </summary>
    [DataContract]
    public class ModelMeta
    {
        [DataMember(Name = "window")] public int window;
        [DataMember(Name = "num_classes")] public int numClasses;
        [DataMember(Name = "mask_token")] public int maskToken;
        [DataMember(Name = "categories")] public Dictionary<string, string> categories;
        [DataMember(Name = "logit_bias")] public float[] logitBias;
        [DataMember(Name = "adjust")] public float adjust;
        [DataMember(Name = "sampler")] public SamplerMeta sampler;
        [DataMember(Name = "epoch")] public int epoch;

        [DataContract]
        public class SamplerMeta
        {
            [DataMember(Name = "steps")] public int steps;
            [DataMember(Name = "temperature")] public float temperature;
            [DataMember(Name = "conf_noise")] public float confNoise;
        }

        public static ModelMeta Load(string path)
        {
            var settings = new DataContractJsonSerializerSettings { UseSimpleDictionaryFormat = true };
            using (var stream = File.OpenRead(path))
                return (ModelMeta)new DataContractJsonSerializer(typeof(ModelMeta), settings).ReadObject(stream);
        }

        public string CategoryName(int id) =>
            categories != null && categories.TryGetValue(id.ToString(), out string name) ? name : id.ToString();
    }

    /// <summary>
    /// Runs the exported layout model with ONNX Runtime on the CPU and samples category grids.
    /// A port of the MaskGIT sampler in train/diffusion.py; deterministic mode matches export_onnx.ort_sample.
    /// No game dependencies, so it can be shared with a 1.7 build and test tools.
    /// Grids are flat arrays of window * window cells, row-major, row 0 = south.
    /// </summary>
    public class LayoutModel : IDisposable
    {
        public readonly ModelMeta meta;
        readonly InferenceSession session;
        readonly string[] outputNames = { "logits" };

        /// <param name="threads">ONNX Runtime intra-op threads; 0 lets it use all cores.</param>
        public LayoutModel(string onnxPath, string metaPath, int threads = 0)
        {
            meta = ModelMeta.Load(metaPath);
            var options = new SessionOptions { IntraOpNumThreads = threads };
            session = new InferenceSession(onnxPath, options);
        }

        public int Size => meta.window;

        public static string RuntimeVersion => OrtEnv.Instance().GetVersionString();

        /// <summary>One forward pass: logits laid out [class, row, column].</summary>
        public float[] Logits(long[] tokens, float maskedRatio)
        {
            int n = Size;
            using (var tokenValue = OrtValue.CreateTensorValueFromMemory(tokens, new long[] { 1, n, n }))
            using (var ratioValue = OrtValue.CreateTensorValueFromMemory(new[] { maskedRatio }, new long[] { 1 }))
            {
                var inputs = new Dictionary<string, OrtValue> { { "tokens", tokenValue }, { "ratio", ratioValue } };
                using (var runOptions = new RunOptions())
                using (var results = session.Run(runOptions, inputs, outputNames))
                    return results[0].GetTensorDataAsSpan<float>().ToArray();
            }
        }

        /// <summary>
        /// Generates one grid.
        /// fixedCells: optional, same size as the grid; a value of 0 or more pins that cell to that category,
        /// -1 leaves it free (inpainting). deterministic: argmax picks and no confidence noise.
        /// </summary>
        public int[] Sample(Random rng, bool deterministic = false, int[] fixedCells = null, int? steps = null)
        {
            int n = Size, cells = n * n, classes = meta.numClasses, mask = meta.maskToken;
            int stepCount = steps ?? meta.sampler.steps;
            float temperature = Math.Max(meta.sampler.temperature, 1e-6f);
            float[] bias = meta.logitBias ?? new float[classes];

            var x = new long[cells];
            for (int i = 0; i < cells; i++)
                x[i] = fixedCells != null && fixedCells[i] >= 0 ? fixedCells[i] : mask;
            int start = 0;
            for (int i = 0; i < cells; i++)
                if (x[i] == mask)
                    start++;

            var pick = new int[cells];
            var conf = new float[cells];
            var sorted = new float[cells];
            var logp = new float[classes];
            for (int k = 0; k < stepCount; k++)
            {
                int maskedCount = 0;
                for (int i = 0; i < cells; i++)
                    if (x[i] == mask)
                        maskedCount++;
                float[] logits = Logits(x, (float)maskedCount / cells);
                double noiseScale = meta.sampler.confNoise * (1.0 - (k + 1.0) / stepCount);

                for (int i = 0; i < cells; i++)
                {
                    if (x[i] != mask)
                    {
                        conf[i] = float.PositiveInfinity;
                        continue;
                    }
                    // log-softmax over classes of (logits + bias) / temperature, in float32 steps like the
                    // Python sampler, so near-ties in confidence order the same way
                    float max = float.NegativeInfinity;
                    for (int c = 0; c < classes; c++)
                    {
                        float l = (logits[c * cells + i] + bias[c]) / temperature;
                        logp[c] = l;
                        if (l > max)
                            max = l;
                    }
                    float sum = 0;
                    for (int c = 0; c < classes; c++)
                    {
                        logp[c] -= max;
                        sum += (float)Math.Exp(logp[c]);
                    }
                    float logSum = (float)Math.Log(sum);
                    int chosen = 0;
                    if (deterministic)
                    {
                        // first index of the maximum, as torch.max / numpy argmax
                        for (int c = 1; c < classes; c++)
                            if (logp[c] > logp[chosen])
                                chosen = c;
                    }
                    else
                    {
                        double u = rng.NextDouble(), acc = 0;
                        for (chosen = 0; chosen < classes - 1; chosen++)
                        {
                            acc += Math.Exp(logp[chosen] - logSum);
                            if (u < acc)
                                break;
                        }
                    }
                    pick[i] = chosen;
                    double c0 = logp[chosen] - logSum;
                    if (!deterministic)
                    {
                        double g = Math.Max(rng.NextDouble(), 1e-20);
                        c0 += noiseScale * -Math.Log(-Math.Log(g));   // Gumbel noise, fades out over the steps
                    }
                    conf[i] = (float)c0;
                }

                // keep the least confident cells hidden: cos schedule over the free cells
                long keepHidden = (long)Math.Floor(Math.Cos((k + 1.0) / stepCount * Math.PI / 2) * start);
                float kth = float.NegativeInfinity;
                if (keepHidden > 0)
                {
                    Array.Copy(conf, sorted, cells);
                    Array.Sort(sorted);
                    kth = sorted[Math.Min(keepHidden, cells - 1)];
                }
                for (int i = 0; i < cells; i++)
                    if (x[i] == mask && !(keepHidden > 0 && conf[i] < kth))
                        x[i] = pick[i];
            }

            var grid = new int[cells];
            for (int i = 0; i < cells; i++)
                grid[i] = (int)x[i];
            return grid;
        }

        public void Dispose() => session.Dispose();
    }
}
