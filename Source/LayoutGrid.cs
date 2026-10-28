using System;
using System.Collections.Generic;
using System.Text;

namespace DiffusionMaps
{
    /// <summary>
    /// Category ids by name, from model_meta.json. Missing categories are -1.
    /// "Wall-like" = wall, door, fence, wall_mounted: anything that blocks a room or sits in a wall line (as train/metrics.py).
    /// </summary>
    public class Categories
    {
        public readonly int empty, wall, door, floor, bridge, fence, mounted;

        public Categories(ModelMeta meta)
        {
            int Id(string name)
            {
                foreach (var pair in meta.categories)
                    if (pair.Value == name)
                        return int.Parse(pair.Key);
                return -1;
            }
            empty = Id("empty");
            wall = Id("wall");
            door = Id("door");
            floor = Id("floor");
            bridge = Id("bridge");
            fence = Id("fence");
            mounted = Id("wall_mounted");
        }

        public bool WallLike(int v) => v == wall || v == door || v == fence || v == mounted;

        /// <summary>Wall-like but not a door: the cells a room's outline is made of.</summary>
        public bool Blocking(int v) => v == wall || v == fence || v == mounted;

        public char Char(int v) =>
            v == empty ? ' ' : v == wall ? '#' : v == door ? '+' : v == floor ? '.' : v == bridge ? '=' : v == fence ? ':' : v == mounted ? '%' : '?';
    }

    /// <summary>
    /// Square category grid helpers. Grids are flat, row-major, row 0 = south; cell (x, z) is at z * size + x.
    /// No game dependencies.
    /// </summary>
    public static class LayoutGrid
    {
        /// <summary>4-connected regions of open cells. Regions touching the grid edge count as outside.</summary>
        public class Regions
        {
            public int[] label;      // 0 = not open, else region id (1-based)
            public int count;
            public bool[] outside;   // by region id
            public int[] area;       // by region id

            public bool Enclosed(int cell) => label[cell] > 0 && !outside[label[cell]];
        }

        public static Regions Label(int[] grid, int size, Func<int, bool> open)
        {
            var r = new Regions { label = new int[grid.Length] };
            var outside = new List<bool> { false };
            var area = new List<int> { 0 };
            var queue = new Queue<int>();
            for (int start = 0; start < grid.Length; start++)
            {
                if (r.label[start] != 0 || !open(grid[start]))
                    continue;
                int id = outside.Count;
                bool edge = false;
                int n = 0;
                r.label[start] = id;
                queue.Enqueue(start);
                while (queue.Count > 0)
                {
                    int cell = queue.Dequeue();
                    n++;
                    int x = cell % size, z = cell / size;
                    if (x == 0 || z == 0 || x == size - 1 || z == size - 1)
                        edge = true;
                    foreach (int next in Neighbours4(cell, size))
                        if (r.label[next] == 0 && open(grid[next]))
                        {
                            r.label[next] = id;
                            queue.Enqueue(next);
                        }
                }
                outside.Add(edge);
                area.Add(n);
            }
            r.count = outside.Count - 1;
            r.outside = outside.ToArray();
            r.area = area.ToArray();
            return r;
        }

        /// <summary>The cell's 4-neighbours that lie inside the grid.</summary>
        public static IEnumerable<int> Neighbours4(int cell, int size)
        {
            int x = cell % size, z = cell / size;
            if (x > 0) yield return cell - 1;
            if (x < size - 1) yield return cell + 1;
            if (z > 0) yield return cell - size;
            if (z < size - 1) yield return cell + size;
        }

        /// <summary>Category at (x, z), or -1 outside the grid.</summary>
        public static int At(int[] grid, int size, int x, int z) =>
            x < 0 || z < 0 || x >= size || z >= size ? -1 : grid[z * size + x];

        public static int[] Crop(int[] grid, int size, int ox, int oz, int cropSize)
        {
            var crop = new int[cropSize * cropSize];
            for (int z = 0; z < cropSize; z++)
                Array.Copy(grid, (oz + z) * size + ox, crop, z * cropSize, cropSize);
            return crop;
        }

        /// <summary>ASCII picture, north at the top, framed with '|'.</summary>
        public static string Render(int[] grid, int size, Categories cats) => RenderSideBySide(new[] { grid }, size, cats);

        public static string RenderSideBySide(IList<int[]> grids, int size, Categories cats, string gap = "   ")
        {
            var sb = new StringBuilder();
            for (int z = size - 1; z >= 0; z--)
            {
                for (int g = 0; g < grids.Count; g++)
                {
                    if (g > 0)
                        sb.Append(gap);
                    sb.Append('|');
                    for (int x = 0; x < size; x++)
                        sb.Append(cats.Char(grids[g][z * size + x]));
                    sb.Append('|');
                }
                sb.AppendLine();
            }
            return sb.ToString();
        }
    }

    /// <summary>The structural measures of train/metrics.py that matter for a settlement, for one grid.</summary>
    public class LayoutStats
    {
        public int wallLike, floor, rooms, roomsWithDoor, inaccessibleRooms, strayWalls, diagonalJoints;
        public int doors, doorsInWallLine;

        public static LayoutStats Of(int[] grid, int size, Categories cats)
        {
            var s = new LayoutStats();
            var regions = LayoutGrid.Label(grid, size, v => !cats.WallLike(v));
            var doorTouch = new HashSet<int>();
            for (int i = 0; i < grid.Length; i++)
            {
                int v = grid[i];
                int x = i % size, z = i / size;
                if (cats.WallLike(v))
                    s.wallLike++;
                if (v == cats.floor)
                    s.floor++;
                int nWall = 0;
                foreach (int nb in LayoutGrid.Neighbours4(i, size))
                    if (cats.WallLike(grid[nb]))
                        nWall++;
                if (cats.Blocking(v) && nWall == 0)
                    s.strayWalls++;
                if (v == cats.door)
                {
                    s.doors++;
                    if (LayoutRepair.InWallLine(grid, size, x, z, cats))
                        s.doorsInWallLine++;
                    foreach (int nb in LayoutGrid.Neighbours4(i, size))
                        if (regions.label[nb] > 0)
                            doorTouch.Add(regions.label[nb]);
                }
                if (x < size - 1 && z < size - 1)
                {
                    bool a = cats.WallLike(v), b = cats.WallLike(grid[i + 1]), c = cats.WallLike(grid[i + size]), d = cats.WallLike(grid[i + size + 1]);
                    if ((a && d && !b && !c) || (b && c && !a && !d))
                        s.diagonalJoints++;
                }
            }
            for (int id = 1; id <= regions.count; id++)
                if (!regions.outside[id])
                {
                    s.rooms++;
                    if (doorTouch.Contains(id))
                        s.roomsWithDoor++;
                }
            s.inaccessibleRooms = LayoutRepair.InaccessibleRegions(grid, size, cats).Count;
            return s;
        }

        public override string ToString() =>
            $"{wallLike} wall-like cells, {floor} floor, {rooms} rooms ({roomsWithDoor} with a door, {inaccessibleRooms} unreachable), " +
            $"doors in a wall line {doorsInWallLine}/{doors}, {strayWalls} stray walls, {diagonalJoints} diagonal joints";
    }
}
