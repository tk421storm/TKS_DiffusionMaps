using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace DiffusionMaps
{
    public enum SeedShape
    {
        /// <summary>L-shaped corner, arms of 3-5 cells.</summary>
        Corner,
        /// <summary>Two walls touching corner to corner; the model tends to finish them as cut room corners.</summary>
        Diagonal,
        /// <summary>A 4-connected diagonal staircase of 5-7 cells; gives curved walls.</summary>
        Staircase,
        /// <summary>Corner and diagonal seeds alternating.</summary>
        Mixed,
    }

    public class GenerateOptions
    {
        public int size = 36;
        public int seeds;
        public SeedShape shape = SeedShape.Corner;
        /// <summary>
        /// The model has no idea where water is, so on dry ground its bridges are blobs of noise. When false, bridges are
        /// ruled out during sampling and the model picks floor or empty instead.
        /// </summary>
        public bool allowBridges;

        public override string ToString() => seeds == 0 ? "no seeds" : $"{seeds} {shape} seed{(seeds > 1 ? "s" : "")}";
    }

    public class GeneratedLayout
    {
        public int size;
        /// <summary>Crop offset inside the model window.</summary>
        public int ox, oz;
        public int[] window;
        /// <summary>The crop before and after repair.</summary>
        public int[] raw, grid;
        public readonly RepairReport repairs = new RepairReport();
        public LayoutStats before, after;
        public long sampleMs;
        public int seedsPlaced;
    }

    /// <summary>
    /// Generates a size x size layout with the model (see docs/HISTORY.md, 2026-10-28):
    /// - Only real obstacles are fixed (as empty); there is no fixed-empty ring around the square.
    /// - Optional seeds: a few wall fragments fixed in the middle of the window for the model to complete; the main
    ///   density control.
    /// - The square is cut from the model window where it cuts the fewest walls, always containing all seeds.
    /// - LayoutRepair then fixes the structure.
    /// No game dependencies.
    /// </summary>
    public static class LayoutGenerator
    {
        const int SeedSpacing = 8;
        /// <summary>Seeds stay this far inside the centred square, so the crop can still shift to avoid cutting buildings.</summary>
        const int SeedMargin = 6;

        /// <param name="blocked">window cells that must stay empty (obstacles, off the map); null for none</param>
        /// <param name="cropFits">whether a crop at window offset (ox, oz) may be used; null for any</param>
        public static GeneratedLayout Generate(LayoutModel model, Categories cats, GenerateOptions options, Random rng,
            bool[] blocked = null, Func<int, int, bool> cropFits = null)
        {
            int w = model.Size, size = options.size, lo = (w - size) / 2;
            var fixedCells = new int[w * w];
            for (int i = 0; i < fixedCells.Length; i++)
                fixedCells[i] = blocked != null && blocked[i] ? cats.empty : -1;

            var layout = new GeneratedLayout { size = size };
            int minX = w, minZ = w, maxX = -1, maxZ = -1;
            foreach (List<int> seed in ScatterSeeds(rng, options, w, lo + SeedMargin, lo + size - SeedMargin, blocked))
            {
                foreach (int cell in seed)
                {
                    fixedCells[cell] = cats.wall;
                    minX = Math.Min(minX, cell % w); maxX = Math.Max(maxX, cell % w);
                    minZ = Math.Min(minZ, cell / w); maxZ = Math.Max(maxZ, cell / w);
                }
                layout.seedsPlaced++;
            }

            float[] extraBias = null;
            if (!options.allowBridges && cats.bridge >= 0)
            {
                extraBias = new float[model.meta.numClasses];
                extraBias[cats.bridge] = -30f;
            }
            var watch = Stopwatch.StartNew();
            layout.window = model.Sample(rng, fixedCells: fixedCells, extraBias: extraBias);
            layout.sampleMs = watch.ElapsedMilliseconds;

            // the crop must contain every seed cell
            bool Fits(int ox, int oz) => (cropFits == null || cropFits(ox, oz))
                && (layout.seedsPlaced == 0 || (ox <= minX && oz <= minZ && ox + size > maxX && oz + size > maxZ));
            layout.ox = layout.oz = lo;
            BestCrop(layout.window, w, size, cats, Fits, ref layout.ox, ref layout.oz);
            layout.raw = LayoutGrid.Crop(layout.window, w, layout.ox, layout.oz, size);
            layout.before = LayoutStats.Of(layout.raw, size, cats);
            layout.grid = LayoutRepair.Repair(layout.window, w, layout.ox, layout.oz, size, cats, layout.repairs);
            layout.after = LayoutStats.Of(layout.grid, size, cats);
            return layout;
        }

        /// <summary>The crop with the most structure and the fewest wall lines cut by its edge (as train/inpaint_test.py).</summary>
        static void BestCrop(int[] window, int w, int size, Categories cats, Func<int, int, bool> cropFits, ref int bestX, ref int bestZ)
        {
            // keeps the given (centred) offset if no crop fits
            int bestScore = int.MinValue;
            for (int oz = 0; oz <= w - size; oz++)
                for (int ox = 0; ox <= w - size; ox++)
                {
                    if (!cropFits(ox, oz))
                        continue;
                    int inside = 0;
                    for (int z = 0; z < size; z++)
                        for (int x = 0; x < size; x++)
                            if (cats.WallLike(window[(oz + z) * w + ox + x]))
                                inside++;
                    int score = inside - 3 * CutCells(window, w, ox, oz, size, cats);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestX = ox;
                        bestZ = oz;
                    }
                }
        }

        /// <summary>Wall-like cells on the crop border with a wall-like neighbour just outside the crop.</summary>
        static int CutCells(int[] window, int w, int ox, int oz, int size, Categories cats)
        {
            bool WL(int x, int z) => cats.WallLike(LayoutGrid.At(window, w, x, z));
            int n = 0;
            for (int i = 0; i < size; i++)
            {
                if (WL(ox + i, oz) && WL(ox + i, oz - 1)) n++;
                if (WL(ox + i, oz + size - 1) && WL(ox + i, oz + size)) n++;
                if (WL(ox, oz + i) && WL(ox - 1, oz + i)) n++;
                if (WL(ox + size - 1, oz + i) && WL(ox + size, oz + i)) n++;
            }
            return n;
        }

        /// <summary>Seeds (lists of window cells) inside [lo, hi), anchors at least SeedSpacing apart, never on blocked cells.</summary>
        static IEnumerable<List<int>> ScatterSeeds(Random rng, GenerateOptions options, int w, int lo, int hi, bool[] blocked)
        {
            var anchors = new List<(int x, int z)>();
            for (int tries = 0; anchors.Count < options.seeds && tries < 1000; tries++)
            {
                int cx = rng.Next(lo + 1, hi - 1), cz = rng.Next(lo + 1, hi - 1);
                if (anchors.Exists(a => Math.Abs(a.x - cx) + Math.Abs(a.z - cz) < SeedSpacing))
                    continue;
                SeedShape shape = options.shape == SeedShape.Mixed
                    ? (anchors.Count % 2 == 0 ? SeedShape.Corner : SeedShape.Diagonal)
                    : options.shape;
                List<(int x, int z)> cells = Shape(rng, shape, cx, cz);
                if (!cells.TrueForAll(c => c.x >= lo && c.z >= lo && c.x < hi && c.z < hi && (blocked == null || !blocked[c.z * w + c.x])))
                    continue;
                anchors.Add((cx, cz));
                yield return cells.ConvertAll(c => c.z * w + c.x);
            }
        }

        static List<(int x, int z)> Shape(Random rng, SeedShape shape, int cx, int cz)
        {
            int dx = rng.Next(2) == 0 ? -1 : 1, dz = rng.Next(2) == 0 ? -1 : 1;
            var cells = new List<(int x, int z)> { (cx, cz) };
            switch (shape)
            {
                case SeedShape.Corner:
                    int armZ = rng.Next(3, 6), armX = rng.Next(3, 6);
                    for (int i = 1; i < armZ; i++)
                        cells.Add((cx, cz + dz * i));
                    for (int i = 1; i < armX; i++)
                        cells.Add((cx + dx * i, cz));
                    break;
                case SeedShape.Diagonal:
                    cells.Add((cx + dx, cz + dz));
                    break;
                case SeedShape.Staircase:
                    int x = cx, z = cz, length = rng.Next(5, 8);
                    for (int i = 0; i < length - 1; i++)
                    {
                        if (i % 2 == 0)
                            x += dx;
                        else
                            z += dz;
                        cells.Add((x, z));
                    }
                    break;
            }
            return cells;
        }
    }
}
