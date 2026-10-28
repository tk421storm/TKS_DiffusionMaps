using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using LudeonTK;
using Verse;

namespace DiffusionMaps
{
    /// <summary>
    /// Dev-mode actions for checking the model inside the game. They work from the main menu as well as in a game
    /// (allowedGameStates = Invalid means no restriction).
    /// </summary>
    public static class DebugActions
    {
        const string Category = "DiffusionMaps";

        [DebugAction(Category, "Sample layout (log)", allowedGameStates = AllowedGameStates.Invalid)]
        static void SampleLayout()
        {
            Run("Sample layout", () =>
            {
                var report = new StringBuilder();
                var watch = Stopwatch.StartNew();
                LayoutModel m = DiffusionMapsMod.Model;
                report.AppendLine($"model ready in {watch.ElapsedMilliseconds} ms (ONNX Runtime {LayoutModel.RuntimeVersion}, native {OnnxNative.LoadedPath})");
                int seed = Rand.Int;
                watch.Restart();
                int[] grid = m.Sample(new Random(seed));
                report.AppendLine($"sampled 1 layout in {watch.ElapsedMilliseconds} ms ({m.meta.sampler.steps} steps, seed {seed})");
                report.Append(Render(grid, m.meta));
                Log.Message(report.ToString());
            });
        }

        [DebugAction(Category, "Benchmark (all threads vs 1 thread)", allowedGameStates = AllowedGameStates.Invalid)]
        static void Benchmark()
        {
            Run("Benchmark", () =>
            {
                var report = new StringBuilder();
                Time(report, "all threads", DiffusionMapsMod.Model);
                using (LayoutModel single = DiffusionMapsMod.Instance.LoadModel(threads: 1))
                    Time(report, "1 thread", single);
                Log.Message(report.ToString());
            });
        }

        /// <summary>
        /// Deterministic sample (argmax, no noise), written as digits to runs/ingame_deterministic.txt so it can be
        /// compared with the Python reference sampler (train/export_onnx.py ort_sample).
        /// </summary>
        [DebugAction(Category, "Deterministic sample (write file)", allowedGameStates = AllowedGameStates.Invalid)]
        static void DeterministicSample()
        {
            Run("Deterministic sample", () =>
            {
                LayoutModel m = DiffusionMapsMod.Model;
                int[] grid = m.Sample(null, deterministic: true);
                string dir = Path.Combine(DiffusionMapsMod.Instance.Content.RootDir, "runs");
                Directory.CreateDirectory(dir);
                string path = Path.Combine(dir, "ingame_deterministic.txt");
                File.WriteAllText(path, Digits(grid, m.Size));
                Log.Message($"[DiffusionMaps] wrote {path}\n{Render(grid, m.meta)}");
            });
        }

        static void Time(StringBuilder report, string label, LayoutModel m)
        {
            var watch = Stopwatch.StartNew();
            m.Sample(new Random(1));   // warm-up
            long first = watch.ElapsedMilliseconds;
            const int runs = 3;
            watch.Restart();
            for (int i = 0; i < runs; i++)
                m.Sample(new Random(i + 2));
            report.AppendLine($"{label}: first sample {first} ms, then {watch.ElapsedMilliseconds / runs} ms per sample ({m.meta.sampler.steps} steps)");
        }

        static void Run(string name, Action action)
        {
            try
            {
                action();
            }
            catch (Exception e)
            {
                Log.Error($"[DiffusionMaps] {name} failed: {e}");
            }
        }

        /// <summary>ASCII picture, north at the top. Same characters as tools/report.py.</summary>
        public static string Render(int[] grid, ModelMeta meta)
        {
            int n = meta.window;
            var sb = new StringBuilder();
            for (int z = n - 1; z >= 0; z--)
            {
                sb.Append('|');
                for (int x = 0; x < n; x++)
                    sb.Append(Char(meta.CategoryName(grid[z * n + x])));
                sb.AppendLine("|");
            }
            return sb.ToString();
        }

        static char Char(string category)
        {
            switch (category)
            {
                case "empty": return ' ';
                case "wall": return '#';
                case "door": return '+';
                case "floor": return '.';
                case "bridge": return '=';
                case "fence": return ':';
                case "wall_mounted": return '%';
                default: return '?';
            }
        }

        /// <summary>One line per row, row 0 (south) first, one digit per cell.</summary>
        public static string Digits(int[] grid, int n)
        {
            var sb = new StringBuilder();
            for (int z = 0; z < n; z++)
            {
                for (int x = 0; x < n; x++)
                    sb.Append(grid[z * n + x]);
                sb.Append('\n');
            }
            return sb.ToString();
        }
    }
}
