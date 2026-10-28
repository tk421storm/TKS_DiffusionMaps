using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace DiffusionMaps
{
    /// <summary>
    /// Loads the native ONNX Runtime library from the mod folder before the managed API first needs it.
    /// The managed API imports it by bare name ("onnxruntime"), which Windows resolves to an already loaded module
    /// of that name. Without this the game would not look in the mod folder, and could instead pick up the old
    /// Windows ML copy in System32. No game dependencies, so it can be shared with a 1.7 build and test tools.
    /// </summary>
    public static class OnnxNative
    {
        public const string LibraryName = "onnxruntime.dll";

        static IntPtr handle;

        /// <summary>Path of the loaded library, for logging.</summary>
        public static string LoadedPath { get; private set; }

        public static void EnsureLoaded(string nativeDir)
        {
            if (handle != IntPtr.Zero)
                return;
            if (Environment.OSVersion.Platform != PlatformID.Win32NT)
                throw new PlatformNotSupportedException("DiffusionMaps currently ships ONNX Runtime for Windows x64 only.");
            string path = Path.Combine(nativeDir, LibraryName);
            if (!File.Exists(path))
                throw new FileNotFoundException("ONNX Runtime native library not found", path);
            handle = LoadLibraryW(path);
            if (handle == IntPtr.Zero)
                throw new DllNotFoundException($"LoadLibrary failed for {path} (Win32 error {Marshal.GetLastWin32Error()})");
            var sb = new StringBuilder(1024);
            LoadedPath = GetModuleFileNameW(handle, sb, sb.Capacity) > 0 ? sb.ToString() : path;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        static extern IntPtr LoadLibraryW(string path);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        static extern int GetModuleFileNameW(IntPtr module, StringBuilder name, int size);
    }
}
