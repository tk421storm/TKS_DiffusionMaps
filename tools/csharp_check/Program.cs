using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using DiffusionMaps;

// Usage:
//   LayoutCheck <mod root> [reference.txt]   deterministic sample vs the Python reference, plus timing
//   LayoutCheck <mod root> repair [n]        generate n layouts per seed setting, measures before and after repair
//   LayoutCheck <mod root> rock [n]          n layouts (2 mixed seeds) beside a wavy cliff, rock shell on and off
// Loads the native ONNX Runtime from <mod root>/Native/win-x64 the same way the mod does.
static class Program
{
    static int Main(string[] args)
    {
        string root = args[0];
        string modelDir = Path.Combine(root, "Models", "v1_mdm");
        OnnxNative.EnsureLoaded(Path.Combine(root, "Native", "win-x64"));
        Console.WriteLine($"native: {OnnxNative.LoadedPath}, ONNX Runtime {LayoutModel.RuntimeVersion}");
        using (var model = new LayoutModel(Path.Combine(modelDir, "model.onnx"), Path.Combine(modelDir, "model_meta.json")))
        {
            if (args.Length > 1 && args[1] == "rock")
                return RockFace(model, args.Length > 2 ? int.Parse(args[2]) : 8);
            if (args.Length > 1 && args[1] == "repair")
                return Repair(model, args.Length > 2 ? int.Parse(args[2]) : 12);
            return Deterministic(model, args.Length > 1 ? args[1] : Path.Combine(modelDir, "ref_deterministic.txt"));
        }
    }

    static int Deterministic(LayoutModel model, string reference)
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

    /// <summary>Averages of the layout measures before and after repair, per seed setting; prints the first layout of each.</summary>
    static int Repair(LayoutModel model, int count)
    {
        var cats = new Categories(model.meta);
        var settings = new[]
        {
            new GenerateOptions { seeds = 0 },
            new GenerateOptions { seeds = 2, shape = SeedShape.Corner },
            new GenerateOptions { seeds = 2, shape = SeedShape.Diagonal },
            new GenerateOptions { seeds = 1, shape = SeedShape.Staircase },
            new GenerateOptions { seeds = 2, shape = SeedShape.Mixed },
        };
        int problems = 0;
        foreach (GenerateOptions options in settings)
        {
            var rng = new Random(11);
            double[] before = new double[6], after = new double[6];
            var total = new RepairReport();
            long ms = 0;
            for (int i = 0; i < count; i++)
            {
                GeneratedLayout layout = LayoutGenerator.Generate(model, cats, options, rng);
                Add(before, layout.before);
                Add(after, layout.after);
                ms += layout.sampleMs;
                RepairReport r = layout.repairs;
                total.edgeWalls += r.edgeWalls; total.slivers += r.slivers; total.diagonalFills += r.diagonalFills; total.straysRemoved += r.straysRemoved;
                total.gapDoors += r.gapDoors; total.pocketsFilled += r.pocketsFilled; total.addedDoors += r.addedDoors;
                total.unresolvedRooms += r.unresolvedRooms;
                problems += layout.after.diagonalJoints + layout.after.strayWalls;
                if (i == 0)
                    Console.WriteLine($"\n== {options}: first layout, before | after repair\n" +
                        LayoutGrid.RenderSideBySide(new[] { layout.raw, layout.grid }, layout.size, cats));
            }
            Console.WriteLine($"== {options}: {count} layouts, {ms / count} ms sampling each");
            Console.WriteLine("            wall-like  floor  rooms  with door  unreachable  diag joints  stray walls");
            Console.WriteLine($"  before  {Row(before, count)}");
            Console.WriteLine($"  after   {Row(after, count)}");
            Console.WriteLine($"  repairs (totals): {total}");
        }
        return problems == 0 ? 0 : 1;
    }

    /// <summary>A cliff on the east side of the window with a wavy face; rock must survive repair except for mined doors.</summary>
    static int RockFace(LayoutModel model, int count)
    {
        var cats = new Categories(model.meta);
        int w = model.Size;
        int problems = 0;
        foreach (bool shell in new[] { false, true })
        {
            var rng = new Random(5);
            double rooms = 0, rockRooms = 0, built = 0;
            int rockDoors = 0, seedsPlaced = 0;
            for (int i = 0; i < count; i++)
            {
                double phase = rng.NextDouble() * 6;
                var rock = new bool[w * w];
                for (int z = 0; z < w; z++)
                    for (int x = 0; x < w; x++)
                        rock[z * w + x] = x > 32 + 4 * Math.Sin(z / 4.0 + phase);
                var options = new GenerateOptions { seeds = 2, shape = SeedShape.Mixed, rockShell = shell };
                GeneratedLayout layout = LayoutGenerator.Generate(model, cats, options, rng, rock: rock);
                rooms += layout.after.rooms;
                built += layout.after.wallLike;
                rockDoors += layout.repairs.rockDoors;
                seedsPlaced += layout.seedsPlaced;
                // rooms with rock in their outline
                var regions = LayoutGrid.Label(layout.grid, layout.size, v => !cats.WallLike(v));
                var touching = new System.Collections.Generic.HashSet<int>();
                for (int c = 0; c < layout.grid.Length; c++)
                    if (layout.grid[c] == Categories.Rock)
                        foreach (int nb in LayoutGrid.Neighbours4(c, layout.size))
                            if (regions.Enclosed(nb))
                                touching.Add(regions.label[nb]);
                rockRooms += touching.Count;
                // every rock cell of the crop must still be rock, or a door mined through it
                for (int z = 0; z < layout.size; z++)
                    for (int x = 0; x < layout.size; x++)
                    {
                        int v = layout.grid[z * layout.size + x];
                        if (rock[(layout.oz + z) * w + layout.ox + x] && v != Categories.Rock && v != cats.door)
                            problems++;
                    }
                problems += layout.after.inaccessibleRooms;
                if (i == 0)
                    Console.WriteLine($"\n== rock shell {(shell ? "on" : "off")}: first layout, before | after repair\n" +
                        LayoutGrid.RenderSideBySide(new[] { layout.raw, layout.grid }, layout.size, cats));
            }
            Console.WriteLine($"== rock shell {(shell ? "on" : "off")}: {count} layouts: {built / count:F0} built wall-like cells, " +
                $"{rooms / count:F1} rooms, {rockRooms / count:F1} rooms using rock, {rockDoors} doors mined through rock, {seedsPlaced} seeds placed");
        }
        Console.WriteLine(problems == 0 ? "rock check: ok" : $"rock check: {problems} problems");
        return problems == 0 ? 0 : 1;
    }

    static void Add(double[] sum, LayoutStats s)
    {
        sum[0] += s.wallLike; sum[1] += s.floor; sum[2] += s.rooms; sum[3] += s.roomsWithDoor;
        sum[4] += s.inaccessibleRooms; sum[5] += s.diagonalJoints;
    }

    static string Row(double[] sum, int n) =>
        $"{sum[0] / n,9:F1}  {sum[1] / n,5:F0}  {sum[2] / n,5:F1}  {sum[3] / n,9:F1}  {sum[4] / n,11:F1}  {sum[5] / n,11:F1}";
}
