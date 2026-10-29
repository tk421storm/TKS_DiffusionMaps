using System;
using System.Collections.Generic;
using RimWorld;
using RimWorld.BaseGen;
using Verse;

namespace DiffusionMaps
{
    /// <summary>
    /// The game side of a generated layout: marks rock and other obstacles for the generator, then clears the chosen square
    /// and builds floors, walls, doors, fences and vents, and roofs the rooms. Materials follow the faction the same way vanilla
    /// BaseGen picks them.
    /// </summary>
    public static class LayoutPlacer
    {
        public class Result
        {
            public GeneratedLayout layout;
            /// <summary>Where the layout was built: the requested square, shifted by the crop.</summary>
            public CellRect rect;
            /// <summary>Cells in the requested square: natural rock, and other cells that can't be built on.</summary>
            public int rock, obstacles;
            public readonly Dictionary<string, int> placed = new Dictionary<string, int>();

            public void Count(string what) => placed[what] = placed.TryGetValue(what, out int n) ? n + 1 : 1;
        }

        /// <summary>Rooms larger than this stay unroofed (yards and courtyards).</summary>
        const int MaxRoofedRoom = 150;

        /// <param name="rect">requested square; the model window is centred on it and the result may shift by up to
        /// (window - size) / 2 cells to avoid cutting buildings</param>
        /// <param name="allowed">extra test for the final square (e.g. keep clear of the vanilla side); null for none</param>
        public static Result Generate(Map map, CellRect rect, Faction faction, LayoutModel model, GenerateOptions options, int seed,
            Func<CellRect, bool> allowed = null)
        {
            int w = model.Size, lo = (w - options.size) / 2;
            IntVec3 origin = new IntVec3(rect.minX - lo, 0, rect.minZ - lo);
            var result = new Result();

            var blocked = new bool[w * w];
            var rock = new bool[w * w];
            for (int z = 0; z < w; z++)
                for (int x = 0; x < w; x++)
                {
                    IntVec3 c = origin + new IntVec3(x, 0, z);
                    int i = z * w + x;
                    rock[i] = c.InBounds(map) && IsNaturalRock(c, map);
                    blocked[i] = !c.InBounds(map) || (!rock[i] && IsObstacle(c, map));
                    if (rect.Contains(c))
                    {
                        if (rock[i])
                            result.rock++;
                        else if (blocked[i])
                            result.obstacles++;
                    }
                }
            CellRect inside = CellRect.WholeMap(map).ContractedBy(1);
            CellRect Square(int ox, int oz) => new CellRect(origin.x + ox, origin.z + oz, options.size, options.size);
            bool Fits(int ox, int oz) => Square(ox, oz).FullyContainedWithin(inside) && (allowed == null || allowed(Square(ox, oz)));

            var cats = new Categories(model.meta);
            result.layout = LayoutGenerator.Generate(model, cats, options, new Random(seed), blocked, rock, Fits);
            result.rect = Square(result.layout.ox, result.layout.oz);

            ClearRect(map, result.rect);
            Place(map, result, faction, cats);
            return result;
        }

        static bool IsNaturalRock(IntVec3 c, Map map) => NaturalRockAt(c, map) != null;

        static Thing NaturalRockAt(IntVec3 c, Map map)
        {
            foreach (Thing t in c.GetThingList(map))
                if (t.def.building != null && t.def.building.isNaturalRock)
                    return t;
            return null;
        }

        /// <summary>Other cells the layout must leave empty: open ground under a mountain roof (caves), and terrain that
        /// can't take walls (water, marsh).</summary>
        static bool IsObstacle(IntVec3 c, Map map)
        {
            if (map.roofGrid.RoofAt(c) is RoofDef roof && roof.isNatural)
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

        static void Place(Map map, Result result, Faction faction, Categories cats)
        {
            int n = result.layout.size;
            int[] grid = result.layout.grid;
            ThingDef wallStuff = BaseGenUtility.RandomCheapWallStuff(faction);
            TerrainDef floor = BaseGenUtility.RandomBasicFloorDef(faction);
            ThingDef fence = DefDatabase<ThingDef>.GetNamedSilentFail("Fence");
            ThingDef vent = DefDatabase<ThingDef>.GetNamedSilentFail("Vent");

            for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                {
                    IntVec3 c = new IntVec3(result.rect.minX + x, 0, result.rect.minZ + z);
                    int v = grid[z * n + x];
                    if (v == cats.wall)
                        Spawn(ThingDefOf.Wall, wallStuff, c, map, faction, result);
                    else if (v == cats.door)
                        Spawn(ThingDefOf.Door, wallStuff, c, map, faction, result);
                    else if (v == cats.fence)
                        Spawn(fence, wallStuff, c, map, faction, result);
                    else if (v == cats.mounted)
                        // the category covers vents, coolers, heaters and windows; a vent stands in for all of them
                        Spawn(vent, wallStuff, c, map, faction, result);
                    else if (v == cats.floor || v == cats.bridge)
                        // bridges are ruled out while sampling unless allowed; if any come through they are laid as floor
                        SetFloor(floor, c, map, result);
                }
            RoofRooms(map, result, cats);
        }

        static void Spawn(ThingDef def, ThingDef wallStuff, IntVec3 c, Map map, Faction faction, Result result)
        {
            if (def == null)
                return;
            // repair only builds on rock to put a door through it when a room has no other way out: mine the cell first
            Thing rock = NaturalRockAt(c, map);
            if (rock != null)
            {
                rock.Destroy();
                result.Count("mined");
            }
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
        /// Constructed roof over enclosed rooms up to MaxRoofedRoom cells. Fences don't count as walls here, so animal
        /// pens stay open to the sky.
        /// </summary>
        static void RoofRooms(Map map, Result result, Categories cats)
        {
            int n = result.layout.size;
            int[] grid = result.layout.grid;
            var regions = LayoutGrid.Label(grid, n, v => v != cats.wall && v != cats.door && v != cats.mounted && v != Categories.Rock);
            for (int i = 0; i < grid.Length; i++)
            {
                if (!regions.Enclosed(i) || regions.area[regions.label[i]] > MaxRoofedRoom)
                    continue;
                IntVec3 c = new IntVec3(result.rect.minX + i % n, 0, result.rect.minZ + i / n);
                if (map.roofGrid.RoofAt(c) == null)
                {
                    map.roofGrid.SetRoof(c, RoofDefOf.RoofConstructed);
                    result.Count("roof");
                }
            }
        }

        /// <summary>
        /// Reads what is built in rect back into categories (for vanilla's side of the comparison): doors, fences, walls
        /// (anything impassable that fills its cell, natural rock excepted), and constructed floor.
        /// </summary>
        public static int[] ReadBack(Map map, CellRect rect, Categories cats)
        {
            var grid = new int[rect.Width * rect.Height];
            int i = 0;
            for (int z = rect.minZ; z <= rect.maxZ; z++)
                for (int x = rect.minX; x <= rect.maxX; x++, i++)
                {
                    var c = new IntVec3(x, 0, z);
                    Building edifice = c.GetEdifice(map);
                    ThingDef def = edifice?.def;
                    if (def != null && def.building != null && !def.building.isNaturalRock)
                    {
                        if (def.IsDoor)
                        {
                            grid[i] = cats.door;
                            continue;
                        }
                        if (def.IsFence)
                        {
                            grid[i] = cats.fence;
                            continue;
                        }
                        if (def.IsWall || (def.passability == Traversability.Impassable && def.fillPercent >= 0.99f))
                        {
                            grid[i] = cats.wall;
                            continue;
                        }
                    }
                    TerrainDef terrain = c.GetTerrain(map);
                    grid[i] = map.terrainGrid.CanRemoveTopLayerAt(c) || terrain.bridge ? cats.floor : cats.empty;
                }
            return grid;
        }
    }
}
