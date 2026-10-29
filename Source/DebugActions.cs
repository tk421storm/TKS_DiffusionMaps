using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using LudeonTK;
using RimWorld;
using RimWorld.BaseGen;
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

        /// <summary>Same as vanilla's SettlementSizeRange midpoint (34-38); both sides use it so they compare fairly.</summary>
        const int CompareSize = 36;
        /// <summary>Room for our square to shift (up to 6 cells) while staying clear of the vanilla side.</summary>
        const int CompareGap = 12;
        const int MinSeparation = 3;

        /// <summary>Whether layouts give rock to the model as a shell (GenerateOptions.rockShell). On by default.</summary>
        static bool rockShell = true;

        [DebugAction(Category, "Toggle rock shell", allowedGameStates = AllowedGameStates.Invalid)]
        static void ToggleRockShell()
        {
            rockShell = !rockShell;
            Messages.Message($"DiffusionMaps: rock shell {(rockShell ? "on (build against mountains)" : "off (rock is empty ground)")}",
                MessageTypeDefOf.NeutralEvent, false);
        }

        /// <summary>
        /// Pick a faction, a seed count and a seed shape, then click on the map: vanilla's settlement generator builds a
        /// 36x36 base to the left of the click, and the model builds one to the right, same faction. No pawns. Both are
        /// logged side by side with their measures. Click again for more pairs; right-click to stop.
        /// </summary>
        [DebugAction(Category, "Compare with vanilla settlement...", allowedGameStates = AllowedGameStates.PlayingOnMap)]
        static List<DebugActionNode> CompareWithVanilla() => FactionNodes(compare: true);

        /// <summary>Like the compare tool, but only builds the model's layout, centred on the click.</summary>
        [DebugAction(Category, "Place diffusion layout...", allowedGameStates = AllowedGameStates.PlayingOnMap)]
        static List<DebugActionNode> PlaceLayout() => FactionNodes(compare: false);

        /// <summary>Faction, then seed count, then (with seeds) seed shape; each leaf is a map tool.</summary>
        static List<DebugActionNode> FactionNodes(bool compare)
        {
            var nodes = new List<DebugActionNode>();
            foreach (Faction faction in Find.FactionManager.AllFactionsListForReading)
            {
                if (faction.IsPlayer || faction.def.hidden || !faction.def.humanlikeFaction || faction.temporary)
                    continue;
                Faction local = faction;
                var factionNode = new DebugActionNode($"{faction.Name} ({faction.def.label}, {faction.def.techLevel})");
                factionNode.AddChild(Tool("0 seeds (best crop)", local, compare, new GenerateOptions()));
                for (int k = 1; k <= 4; k++)
                {
                    var countNode = new DebugActionNode($"{k} seed{(k > 1 ? "s" : "")}");
                    foreach (SeedShape shape in Enum.GetValues(typeof(SeedShape)))
                        countNode.AddChild(Tool(ShapeLabel(shape), local, compare, new GenerateOptions { seeds = k, shape = shape }));
                    factionNode.AddChild(countNode);
                }
                nodes.Add(factionNode);
            }
            return nodes;
        }

        static string ShapeLabel(SeedShape shape)
        {
            switch (shape)
            {
                case SeedShape.Corner: return "corner (L-shape)";
                case SeedShape.Diagonal: return "diagonal pair (cut corners)";
                case SeedShape.Staircase: return "staircase (curved walls)";
                default: return "mixed (corner + diagonal)";
            }
        }

        static DebugActionNode Tool(string label, Faction faction, bool compare, GenerateOptions options) =>
            new DebugActionNode(label, DebugActionType.ToolMap) { action = () => OnClick(UI.MouseCell(), faction, compare, options) };

        static void OnClick(IntVec3 click, Faction faction, bool compare, GenerateOptions options)
        {
            Run("Compare", () =>
            {
                Map map = Find.CurrentMap;
                int half = CompareSize / 2;
                var ours = compare
                    ? new CellRect(click.x + CompareGap / 2, click.z - half, CompareSize, CompareSize)
                    : new CellRect(click.x - half, click.z - half, CompareSize, CompareSize);
                var vanilla = new CellRect(click.x - CompareGap / 2 - CompareSize, click.z - half, CompareSize, CompareSize);
                CellRect inside = CellRect.WholeMap(map).ContractedBy(1);
                if (!ours.FullyContainedWithin(inside) || (compare && !vanilla.FullyContainedWithin(inside)))
                {
                    Messages.Message("Not enough room here: needs " + (compare ? $"{2 * CompareSize + CompareGap}x{CompareSize}" : $"{CompareSize}x{CompareSize}") + " cells inside the map.", MessageTypeDefOf.RejectInput, false);
                    return;
                }

                LayoutModel model = DiffusionMapsMod.Model;
                var cats = new Categories(model.meta);
                options.rockShell = rockShell;
                var report = new StringBuilder();
                report.AppendLine($"[DiffusionMaps] {faction.Name} ({faction.def.label}), {options}");
                int[] vanillaGrid = null;
                if (compare)
                {
                    var watch = Stopwatch.StartNew();
                    LayoutPlacer.ClearRect(map, vanilla);
                    GenerateVanilla(map, vanilla, faction);
                    vanillaGrid = LayoutPlacer.ReadBack(map, vanilla, cats);
                    report.AppendLine($"vanilla  {vanilla} in {watch.ElapsedMilliseconds} ms: {LayoutStats.Of(vanillaGrid, CompareSize, cats)}");
                    MoteMaker.ThrowText(Label(vanilla), map, "vanilla", 8f);
                }

                int seed = Rand.Int;
                CellRect avoid = vanilla.ExpandedBy(MinSeparation);
                LayoutPlacer.Result result = LayoutPlacer.Generate(map, ours, faction, model, options, seed,
                    allowed: compare ? (Func<CellRect, bool>)(r => !r.Overlaps(avoid)) : null);
                GeneratedLayout layout = result.layout;
                report.AppendLine($"ours     {result.rect} sampled in {layout.sampleMs} ms, seed {seed}, {layout.seedsPlaced} seeds placed, rock shell {(options.rockShell ? "on" : "off")}, {result.rock} rock and {result.obstacles} other obstacle cells");
                report.AppendLine($"  model:  {layout.before}");
                report.AppendLine($"  repair: {layout.repairs}");
                report.AppendLine($"  built:  {layout.after}");
                report.AppendLine("  placed: " + string.Join(", ", result.placed.Select(p => $"{p.Key} {p.Value}")));
                report.Append(compare
                    ? LayoutGrid.RenderSideBySide(new[] { vanillaGrid, layout.grid }, CompareSize, cats)
                    : LayoutGrid.Render(layout.grid, CompareSize, cats));
                MoteMaker.ThrowText(Label(result.rect), map, "diffusion", 8f);
                Log.Message(report.ToString());
            });
        }

        static UnityEngine.Vector3 Label(CellRect rect) => new IntVec3(rect.CenterCell.x, 0, rect.maxZ + 2).ToVector3Shifted();

        /// <summary>The same call GenStep_Settlement makes, without pawns.</summary>
        static void GenerateVanilla(Map map, CellRect rect, Faction faction)
        {
            var parms = new ResolveParams
            {
                rect = rect,
                faction = faction,
                settlementDontGeneratePawns = true,
            };
            BaseGen.globalSettings.map = map;
            BaseGen.globalSettings.minBuildings = 1;
            BaseGen.globalSettings.minBarracks = 1;
            if (faction == Faction.OfEmpire)
            {
                BaseGen.globalSettings.minThroneRooms = 1;
                BaseGen.globalSettings.minLandingPads = 1;
            }
            BaseGen.symbolStack.Push("settlement", parms);
            BaseGen.Generate();
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
