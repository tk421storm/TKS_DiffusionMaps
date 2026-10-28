using System;
using System.Collections.Generic;
using RimWorld;
using RimWorld.BaseGen;
using Verse;

namespace DiffusionMaps
{
    /// <summary>
    /// Generates a layout for a map rectangle and builds it: clears the area, fixes cells that can't be built on as
    /// empty (inpainting), samples the model, then places floors, walls, doors, fences and vents and roofs the enclosed
    /// rooms. Materials follow the faction the same way vanilla BaseGen picks them.
    /// </summary>
    public static class LayoutPlacer
    {
        public class Result
        {
            public int seed;
            public int obstacles;
            public long sampleMs;
            public int[] grid;
            public readonly Dictionary<string, int> placed = new Dictionary<string, int>();

            public void Count(string what) => placed[what] = placed.TryGetValue(what, out int n) ? n + 1 : 1;
        }

        /// <summary>
        /// rect must be no larger than the model window. It is centred in the window, and the window cells around it
        /// are fixed as empty, so the model has to fit its buildings inside rect.
        /// </summary>
        public static Result Generate(Map map, CellRect rect, Faction faction, LayoutModel model, int seed)
        {
            int n = model.Size;
            if (rect.Width > n || rect.Height > n)
                throw new ArgumentException($"rect {rect.Width}x{rect.Height} is larger than the {n}x{n} model window");
            var result = new Result { seed = seed };
            IntVec3 origin = new IntVec3(rect.minX - (n - rect.Width) / 2, 0, rect.minZ - (n - rect.Height) / 2);

            ClearRect(map, rect);

            var fixedCells = new int[n * n];
            for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                {
                    IntVec3 c = origin + new IntVec3(x, 0, z);
                    bool free = rect.Contains(c) && !IsObstacle(c, map);
                    fixedCells[z * n + x] = free ? -1 : Empty(model);
                    if (rect.Contains(c) && !free)
                        result.obstacles++;
                }

            var watch = System.Diagnostics.Stopwatch.StartNew();
            result.grid = model.Sample(new Random(seed), fixedCells: fixedCells);
            result.sampleMs = watch.ElapsedMilliseconds;

            Place(map, origin, result, faction, model.meta);
            return result;
        }

        static int Empty(LayoutModel model) => CategoryId(model.meta, "empty");

        static int CategoryId(ModelMeta meta, string name)
        {
            foreach (var pair in meta.categories)
                if (pair.Value == name)
                    return int.Parse(pair.Key);
            throw new KeyNotFoundException($"model has no category '{name}'");
        }

        /// <summary>Cells the layout must leave alone: natural rock, mountain roof, and terrain that can't take walls (water, marsh).</summary>
        static bool IsObstacle(IntVec3 c, Map map)
        {
            if (map.roofGrid.RoofAt(c) is RoofDef roof && roof.isNatural)
                return true;
            foreach (Thing t in c.GetThingList(map))
                if (t.def.building != null && t.def.building.isNaturalRock)
                    return true;
            return !c.GetTerrain(map).affordances.Contains(TerrainAffordanceDefOf.Heavy);
        }

        /// <summary>
        /// Removes plants, items, chunks and buildings in rect. Pawns and natural rock stay. In real map generation
        /// plants come after the settlement, so the compare tool clears the vanilla side the same way.
        /// </summary>
        public static void ClearRect(Map map, CellRect rect)
        {
            var toDestroy = new List<Thing>();
            foreach (IntVec3 c in rect)
                foreach (Thing t in c.GetThingList(map))
                    if (!(t is Pawn) && t.def.destroyable && !(t.def.building != null && t.def.building.isNaturalRock))
                        toDestroy.Add(t);
            foreach (Thing t in toDestroy)
                if (!t.Destroyed)
                    t.Destroy();
        }

        static void Place(Map map, IntVec3 origin, Result result, Faction faction, ModelMeta meta)
        {
            int n = meta.window;
            ThingDef wallStuff = BaseGenUtility.RandomCheapWallStuff(faction);
            TerrainDef floor = BaseGenUtility.RandomBasicFloorDef(faction);
            ThingDef fence = DefDatabase<ThingDef>.GetNamedSilentFail("Fence");
            ThingDef vent = DefDatabase<ThingDef>.GetNamedSilentFail("Vent");

            for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                {
                    IntVec3 c = origin + new IntVec3(x, 0, z);
                    if (!c.InBounds(map))
                        continue;
                    switch (meta.CategoryName(result.grid[z * n + x]))
                    {
                        case "wall":
                            Spawn(ThingDefOf.Wall, wallStuff, c, map, faction, result);
                            break;
                        case "door":
                            Spawn(ThingDefOf.Door, wallStuff, c, map, faction, result);
                            break;
                        case "fence":
                            Spawn(fence, wallStuff, c, map, faction, result);
                            break;
                        case "wall_mounted":
                            // the category covers vents, coolers, heaters and windows; a vent stands in for all of them
                            Spawn(vent, wallStuff, c, map, faction, result);
                            break;
                        case "floor":
                        case "bridge":
                            // the model knows nothing about water and water cells are fixed empty, so bridges can
                            // only land on dry ground; they are laid as ordinary floor
                            SetFloor(floor, c, map, result);
                            break;
                    }
                }
            RoofEnclosed(map, origin, result, meta);
        }

        static void Spawn(ThingDef def, ThingDef wallStuff, IntVec3 c, Map map, Faction faction, Result result)
        {
            if (def == null)
                return;
            ThingDef stuff = null;
            if (def.MadeFromStuff)
                stuff = wallStuff.stuffProps.CanMake(def) ? wallStuff : GenStuff.DefaultStuffFor(def);
            Thing thing = ThingMaker.MakeThing(def, stuff);
            thing.SetFaction(faction);
            GenSpawn.Spawn(thing, c, map);
            result.Count(def.defName);
        }

        static void SetFloor(TerrainDef floor, IntVec3 c, Map map, Result result)
        {
            if (!c.GetTerrain(map).affordances.Contains(floor.terrainAffordanceNeeded))
                return;
            map.terrainGrid.SetTerrain(c, floor);
            result.Count(floor.defName);
        }

        /// <summary>
        /// Constructed roof over every cell the outside can't reach without passing a wall, door or vent:
        /// a flood fill from the window border through empty, floor, bridge and fence cells.
        /// </summary>
        static void RoofEnclosed(Map map, IntVec3 origin, Result result, ModelMeta meta)
        {
            int n = meta.window;
            var outside = new bool[n * n];
            var queue = new Queue<int>();
            bool Open(int i)
            {
                string cat = meta.CategoryName(result.grid[i]);
                return cat != "wall" && cat != "door" && cat != "wall_mounted";
            }
            for (int i = 0; i < n; i++)
                foreach (int cell in new[] { i, (n - 1) * n + i, i * n, i * n + n - 1 })
                    if (!outside[cell] && Open(cell))
                    {
                        outside[cell] = true;
                        queue.Enqueue(cell);
                    }
            while (queue.Count > 0)
            {
                int cell = queue.Dequeue();
                int x = cell % n, z = cell / n;
                foreach (int next in new[] { x > 0 ? cell - 1 : -1, x < n - 1 ? cell + 1 : -1, z > 0 ? cell - n : -1, z < n - 1 ? cell + n : -1 })
                    if (next >= 0 && !outside[next] && Open(next))
                    {
                        outside[next] = true;
                        queue.Enqueue(next);
                    }
            }
            for (int i = 0; i < n * n; i++)
            {
                if (outside[i] || !Open(i))
                    continue;
                IntVec3 c = origin + new IntVec3(i % n, 0, i / n);
                if (c.InBounds(map) && map.roofGrid.RoofAt(c) == null)
                {
                    map.roofGrid.SetRoof(c, RoofDefOf.RoofConstructed);
                    result.Count("roof");
                }
            }
        }
    }
}
