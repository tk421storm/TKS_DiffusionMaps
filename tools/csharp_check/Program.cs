using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using DiffusionMaps;

// Usage: LayoutCheck <mod root> [reference.txt]
// Loads the native ONNX Runtime from <mod root>/Native/win-x64 the same way the mod does, samples deterministically,
// compares with the reference digits file, and times stochastic sampling.
static class Program
{
    static int Main(string[] args)
    {
        string root = args[0];
        string modelDir = Path.Combine(root, "Models", "v1_mdm");
        string reference = args.Length > 1 ? args[1] : Path.Combine(modelDir, "ref_deterministic.txt");

        OnnxNative.EnsureLoaded(Path.Combine(root, "Native", "win-x64"));
        Console.WriteLine($"native: {OnnxNative.LoadedPath}, ONNX Runtime {LayoutModel.RuntimeVersion}");
        using (var model = new LayoutModel(Path.Combine(modelDir, "model.onnx"), Path.Combine(modelDir, "model_meta.json")))
        {
            int n = model.Size;
            int[] grid = model.Sample(null, deterministic: true);
            string[] want = File.ReadAllLines(reference).Where(l => l.Length > 0).ToArray();
            int same = 0;
            for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                    if (want[z][x] - '0' == grid[z * n + x])
                        same++;
            Console.WriteLine($"deterministic sample vs reference: {same}/{n * n} cells identical ({100.0 * same / (n * n):F2}%)");

            var watch = Stopwatch.StartNew();
            const int runs = 5;
            for (int i = 0; i < runs; i++)
                model.Sample(new Random(i));
            Console.WriteLine($"stochastic sampling, all threads: {watch.ElapsedMilliseconds / runs} ms per layout");
            return same == n * n ? 0 : 1;
        }
    }
}
