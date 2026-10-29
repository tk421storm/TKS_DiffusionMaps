using System.Collections.Generic;

namespace DiffusionMaps
{
    /// <summary>What a repair pass changed.</summary>
    public class RepairReport
    {
        public int edgeWalls, slivers, diagonalFills, straysRemoved, gapDoors, pocketsFilled, addedDoors, rockDoors, unresolvedRooms;

        public override string ToString() =>
            $"{edgeWalls} edge walls, {slivers} cut slivers cleared, {diagonalFills} diagonal joints filled, {straysRemoved} stray walls removed, " +
            $"{gapDoors} gaps made doors, {pocketsFilled} pockets filled, {addedDoors} doors added ({rockDoors} mined through rock), {unresolvedRooms} rooms still unreachable";
    }

    /// <summary>
    /// Fixes the model's structural slips on a cropped layout, the guarantees vanilla's generator gives by construction:
    /// rooms closed where the crop cuts them, no corner-to-corner wall joints (they don't seal a room in RimWorld),
    /// no lone wall pieces, no sealed pockets, and every room reachable through doors. No game dependencies.
    /// Natural rock (Categories.Rock) counts as wall and is never removed; a door goes through rock only when a room has
    /// no other way out (the cell is then mined).
    /// </summary>
    public static class LayoutRepair
    {
        /// <param name="window">the sampled window</param>
        /// <param name="ox">crop offset in the window, x</param>
        /// <param name="oz">crop offset in the window, z</param>
        /// <param name="n">crop size</param>
        public static int[] Repair(int[] window, int windowSize, int ox, int oz, int n, Categories cats, RepairReport report)
        {
            int[] g = LayoutGrid.Crop(window, windowSize, ox, oz, n);
            CloseCutRooms(g, window, windowSize, ox, oz, n, cats, report);
            FillDiagonalJoints(g, n, cats, report);
            RemoveStrays(g, n, cats, report);
            GapsToDoors(g, n, cats, report);
            FillPockets(g, n, cats, report);
            ConnectRooms(g, n, cats, report);
            report.unresolvedRooms = InaccessibleRegions(g, n, cats).Count;
            return g;
        }

        static bool Open(Categories cats, int v) => !cats.WallLike(v);

        /// <summary>Cut rooms with fewer cells than this inside the crop, or fewer off its border, are cleared rather than closed.</summary>
        const int MinCutRoomCells = 10, MinCutRoomInnerCells = 6;

        /// <summary>
        /// A crop border cell that lies inside a room of the full window means the crop cut that room off from its wall.
        /// If enough of the room is inside the crop, put walls on its border cells so that part is closed. A sliver
        /// (a room mostly outside the crop) is cleared to empty instead; its leftover wall pieces go in later steps.
        /// </summary>
        static void CloseCutRooms(int[] g, int[] window, int windowSize, int ox, int oz, int n, Categories cats, RepairReport report)
        {
            var regions = LayoutGrid.Label(window, windowSize, v => Open(cats, v));
            var inside = new Dictionary<int, int>();
            var onBorder = new Dictionary<int, int>();
            for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                {
                    int wi = (oz + z) * windowSize + ox + x;
                    if (!regions.Enclosed(wi))
                        continue;
                    int id = regions.label[wi];
                    inside[id] = inside.TryGetValue(id, out int a) ? a + 1 : 1;
                    if (x == 0 || z == 0 || x == n - 1 || z == n - 1)
                        onBorder[id] = onBorder.TryGetValue(id, out int b) ? b + 1 : 1;
                }
            foreach (var cut in onBorder)
            {
                int id = cut.Key;
                bool sliver = inside[id] < MinCutRoomCells || inside[id] - cut.Value < MinCutRoomInnerCells;
                for (int z = 0; z < n; z++)
                    for (int x = 0; x < n; x++)
                    {
                        if (regions.label[(oz + z) * windowSize + ox + x] != id)
                            continue;
                        bool border = x == 0 || z == 0 || x == n - 1 || z == n - 1;
                        if (sliver)
                            g[z * n + x] = cats.empty;
                        else if (border)
                        {
                            g[z * n + x] = cats.wall;
                            report.edgeWalls++;
                        }
                    }
                if (sliver)
                    report.slivers++;
            }
        }

        /// <summary>
        /// Two wall-like cells touching only corner to corner leave a diagonal leak. Fill one of the two open cells,
        /// preferring the one outside rooms so no room loses floor space; the fill matches fences when both are fences.
        /// </summary>
        static void FillDiagonalJoints(int[] g, int n, Categories cats, RepairReport report)
        {
            for (int pass = 0; pass < 3; pass++)
            {
                bool changed = false;
                var regions = LayoutGrid.Label(g, n, v => Open(cats, v));
                for (int z = 0; z < n - 1; z++)
                    for (int x = 0; x < n - 1; x++)
                    {
                        int i = z * n + x;
                        int a = i, b = i + 1, c = i + n, d = i + n + 1;
                        if (cats.WallLike(g[a]) && cats.WallLike(g[d]) && Open(cats, g[b]) && Open(cats, g[c]))
                            changed |= Fill(g, regions, b, c, g[a], g[d], cats, report);
                        else if (cats.WallLike(g[b]) && cats.WallLike(g[c]) && Open(cats, g[a]) && Open(cats, g[d]))
                            changed |= Fill(g, regions, a, d, g[b], g[c], cats, report);
                    }
                if (!changed)
                    break;
            }
        }

        static bool Fill(int[] g, LayoutGrid.Regions regions, int p, int q, int wallA, int wallB, Categories cats, RepairReport report)
        {
            int pick = regions.Enclosed(p) && !regions.Enclosed(q) ? q : p;
            g[pick] = wallA == cats.fence && wallB == cats.fence ? cats.fence : cats.wall;
            report.diagonalFills++;
            return true;
        }

        /// <summary>Wall-like cells with no wall-like neighbour become floor (if mostly surrounded by floor) or empty.</summary>
        static void RemoveStrays(int[] g, int n, Categories cats, RepairReport report)
        {
            for (int i = 0; i < g.Length; i++)
            {
                if (!cats.WallLike(g[i]) || g[i] == Categories.Rock)
                    continue;
                int walls = 0, floors = 0;
                foreach (int nb in LayoutGrid.Neighbours4(i, n))
                {
                    if (cats.WallLike(g[nb]))
                        walls++;
                    else if (g[nb] == cats.floor)
                        floors++;
                }
                if (walls > 0)
                    continue;
                g[i] = floors >= 2 ? cats.floor : cats.empty;
                report.straysRemoved++;
            }
        }

        /// <summary>
        /// A one-cell gap in a wall line whose closing would separate a room (of at least 3 cells) from the rest is
        /// usually a doorway: make it a door.
        /// </summary>
        static void GapsToDoors(int[] g, int n, Categories cats, RepairReport report)
        {
            for (int i = 0; i < g.Length; i++)
            {
                if (!Open(cats, g[i]) || !DoorSides(g, n, i, cats, c => cats.Blocking(c), out int s1, out int s2))
                    continue;
                int before = g[i];
                g[i] = cats.door;
                var regions = LayoutGrid.Label(g, n, v => Open(cats, v));
                int r1 = regions.label[s1], r2 = regions.label[s2];
                bool separates = r1 != r2 && ((regions.Enclosed(s1) && regions.area[r1] >= 3) || (regions.Enclosed(s2) && regions.area[r2] >= 3));
                if (separates)
                    report.gapDoors++;
                else
                    g[i] = before;
            }
        }

        /// <summary>Enclosed regions of 1-2 cells with no door (typically inside double walls) are filled with wall.</summary>
        static void FillPockets(int[] g, int n, Categories cats, RepairReport report)
        {
            var regions = LayoutGrid.Label(g, n, v => Open(cats, v));
            var withDoor = new HashSet<int>();
            for (int i = 0; i < g.Length; i++)
                if (g[i] == cats.door)
                    foreach (int nb in LayoutGrid.Neighbours4(i, n))
                        withDoor.Add(regions.label[nb]);
            for (int i = 0; i < g.Length; i++)
            {
                int id = regions.label[i];
                if (id > 0 && !regions.outside[id] && regions.area[id] <= 2 && !withDoor.Contains(id))
                {
                    g[i] = cats.wall;
                    report.pocketsFilled++;
                }
            }
        }

        /// <summary>
        /// Adds doors until every room can be reached from outside: one door at a time, in a wall or fence cell between an
        /// unreachable room and a reachable area (outside preferred). If a room borders no reachable area directly, it is
        /// connected to a neighbouring room instead, which a later iteration connects onwards. Rock is used last.
        /// </summary>
        static void ConnectRooms(int[] g, int n, Categories cats, RepairReport report)
        {
            for (int iteration = 0; iteration < 64; iteration++)
            {
                List<int> unreachable = InaccessibleRegions(g, n, cats, out LayoutGrid.Regions regions, out bool[] reachable);
                if (unreachable.Count == 0)
                    return;
                var stuck = new HashSet<int>(unreachable);
                int best = -1, bestRank = int.MaxValue;
                for (int i = 0; i < g.Length; i++)
                {
                    if (g[i] != cats.wall && g[i] != cats.fence && g[i] != Categories.Rock)
                        continue;
                    if (!DoorSides(g, n, i, cats, c => cats.WallLike(c), out int s1, out int s2))
                        continue;
                    int r1 = regions.label[s1], r2 = regions.label[s2];
                    if (r1 == r2 || (!stuck.Contains(r1) && !stuck.Contains(r2)))
                        continue;
                    int other = stuck.Contains(r1) ? r2 : r1;
                    // rank: 0 = opens to the outside, 1 = to a reachable room, 2 = to another unreachable room;
                    // the same through rock ranks after all built walls
                    int rank = (regions.outside[other] ? 0 : reachable[other] ? 1 : 2) + (g[i] == Categories.Rock ? 3 : 0);
                    if (rank < bestRank)
                    {
                        best = i;
                        bestRank = rank;
                        if (rank == 0)
                            break;
                    }
                }
                if (best < 0)
                    return;
                if (g[best] == Categories.Rock)
                    report.rockDoors++;
                g[best] = cats.door;
                report.addedDoors++;
            }
        }

        /// <summary>
        /// True when cell i could hold a door: wall on both sides along one axis (as judged by isWall), and open cells
        /// inside the grid on both sides along the other. s1 and s2 are those two open cells.
        /// </summary>
        static bool DoorSides(int[] g, int n, int i, Categories cats, System.Func<int, bool> isWall, out int s1, out int s2)
        {
            int x = i % n, z = i / n;
            s1 = s2 = -1;
            bool InGridOpen(int xx, int zz) => xx >= 0 && zz >= 0 && xx < n && zz < n && Open(cats, g[zz * n + xx]);
            bool W(int xx, int zz) => xx >= 0 && zz >= 0 && xx < n && zz < n && isWall(g[zz * n + xx]);
            if (W(x - 1, z) && W(x + 1, z) && InGridOpen(x, z - 1) && InGridOpen(x, z + 1))
            {
                s1 = i - n;
                s2 = i + n;
                return true;
            }
            if (W(x, z - 1) && W(x, z + 1) && InGridOpen(x - 1, z) && InGridOpen(x + 1, z))
            {
                s1 = i - 1;
                s2 = i + 1;
                return true;
            }
            return false;
        }

        /// <summary>Door test as in train/metrics.py: wall-like on both sides along one axis, open on both along the other.</summary>
        public static bool InWallLine(int[] g, int n, int x, int z, Categories cats)
        {
            bool WL(int xx, int zz) => cats.WallLike(LayoutGrid.At(g, n, xx, zz));
            return (WL(x - 1, z) && WL(x + 1, z) && !WL(x, z - 1) && !WL(x, z + 1))
                || (WL(x, z - 1) && WL(x, z + 1) && !WL(x - 1, z) && !WL(x + 1, z));
        }

        public static List<int> InaccessibleRegions(int[] g, int n, Categories cats) => InaccessibleRegions(g, n, cats, out _, out _);

        /// <summary>
        /// Enclosed regions that can't be reached from outside: regions are linked through doors (a door joins the regions
        /// on its open sides), starting from the regions that touch the grid edge.
        /// </summary>
        static List<int> InaccessibleRegions(int[] g, int n, Categories cats, out LayoutGrid.Regions regions, out bool[] reachable)
        {
            var r = LayoutGrid.Label(g, n, v => Open(cats, v));
            var links = new List<int>[r.count + 1];
            for (int id = 0; id <= r.count; id++)
                links[id] = new List<int>();
            var touching = new List<int>(4);
            for (int i = 0; i < g.Length; i++)
            {
                if (g[i] != cats.door)
                    continue;
                touching.Clear();
                foreach (int nb in LayoutGrid.Neighbours4(i, n))
                    if (r.label[nb] > 0 && !touching.Contains(r.label[nb]))
                        touching.Add(r.label[nb]);
                foreach (int a in touching)
                    foreach (int b in touching)
                        if (a != b)
                            links[a].Add(b);
            }
            var seen = new bool[r.count + 1];
            var queue = new Queue<int>();
            for (int id = 1; id <= r.count; id++)
                if (r.outside[id])
                {
                    seen[id] = true;
                    queue.Enqueue(id);
                }
            while (queue.Count > 0)
                foreach (int next in links[queue.Dequeue()])
                    if (!seen[next])
                    {
                        seen[next] = true;
                        queue.Enqueue(next);
                    }
            var result = new List<int>();
            for (int id = 1; id <= r.count; id++)
                if (!seen[id])
                    result.Add(id);
            regions = r;
            reachable = seen;
            return result;
        }
    }
}
